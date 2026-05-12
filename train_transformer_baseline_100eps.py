import pickle, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = "full_multitask_audiovima_100eps_dataset.pkl"
MAX_OBJECTS = 8
MAX_PROMPT_LEN = 40

random.seed(42)
torch.manual_seed(42)

data = pickle.load(open(DATA_PATH, "rb"))
samples = data["samples"]
metadata = data["metadata"]

tasks = metadata["tasks"]
roles = metadata["roles"]
obj_names = metadata["obj_names"]
obj_colors = metadata["obj_colors"]
prompt_vocab = metadata["prompt_vocab"]

task2idx = {x: i for i, x in enumerate(tasks)}
role2idx = {x: i for i, x in enumerate(roles)}
name2idx = {x: i for i, x in enumerate(obj_names)}
color2idx = {x: i for i, x in enumerate(obj_colors)}
word2idx = {x: i + 1 for i, x in enumerate(prompt_vocab)}  # 0 = pad

random.shuffle(samples)
split = int(0.8 * len(samples))
train_samples = samples[:split]
test_samples = samples[split:]


def load_image(path):
    img = Image.open(path).convert("RGB").resize((128, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def encode_prompt(tokens):
    ids = [word2idx.get(t, 0) for t in tokens[:MAX_PROMPT_LEN]]
    while len(ids) < MAX_PROMPT_LEN:
        ids.append(0)
    return ids


def encode_objects(placeholders):
    role_ids = []
    name_ids = []
    color_ids = []
    centroids = []
    audio_tokens = []
    mask = []

    for p in placeholders[:MAX_OBJECTS]:
        role_ids.append(role2idx[p["role"]])
        name_ids.append(name2idx[p["obj_name"]])
        color_ids.append(color2idx[p["obj_color"]])
        centroids.append(p["centroid"])
        audio_tokens.append(p["audio_token"])
        mask.append(1)

    while len(role_ids) < MAX_OBJECTS:
        role_ids.append(0)
        name_ids.append(0)
        color_ids.append(0)
        centroids.append(np.zeros(2, dtype=np.float32))
        audio_tokens.append(np.zeros(768, dtype=np.float32))
        mask.append(0)

    return role_ids, name_ids, color_ids, centroids, audio_tokens, mask


def build_tensors(rows):
    imgs = []
    task_ids = []
    prompt_ids = []

    obj_roles = []
    obj_names = []
    obj_colors = []
    obj_centroids = []
    obj_audio = []
    obj_masks = []

    targets = []

    for s in rows:
        r, n, c, cent, aud, m = encode_objects(s["placeholders"])

        imgs.append(load_image(s["image_path"]))
        task_ids.append(task2idx[s["task"]])
        prompt_ids.append(encode_prompt(s["prompt_tokens"]))

        obj_roles.append(r)
        obj_names.append(n)
        obj_colors.append(c)
        obj_centroids.append(cent)
        obj_audio.append(aud)
        obj_masks.append(m)

        targets.append(s["target"])

    return (
        torch.tensor(np.array(imgs), dtype=torch.float32),
        torch.tensor(task_ids, dtype=torch.long),
        torch.tensor(np.array(prompt_ids), dtype=torch.long),
        torch.tensor(np.array(obj_roles), dtype=torch.long),
        torch.tensor(np.array(obj_names), dtype=torch.long),
        torch.tensor(np.array(obj_colors), dtype=torch.long),
        torch.tensor(np.array(obj_centroids), dtype=torch.float32),
        torch.tensor(np.array(obj_audio), dtype=torch.float32),
        torch.tensor(np.array(obj_masks), dtype=torch.bool),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )


X_img_tr, X_task_tr, X_prompt_tr, X_orole_tr, X_oname_tr, X_ocolor_tr, X_ocent_tr, X_oaudio_tr, X_omask_tr, Y_train = build_tensors(train_samples)
X_img_te, X_task_te, X_prompt_te, X_orole_te, X_oname_te, X_ocolor_te, X_ocent_te, X_oaudio_te, X_omask_te, Y_test = build_tensors(test_samples)


class TransformerAudioVIMA(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 128
        self.d_model = d_model

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 16, d_model),
            nn.ReLU(),
        )

        self.task_emb = nn.Embedding(len(tasks), d_model)
        self.word_emb = nn.Embedding(len(prompt_vocab) + 1, d_model, padding_idx=0)

        self.role_emb = nn.Embedding(len(roles), 32)
        self.name_emb = nn.Embedding(len(obj_names), 32)
        self.color_emb = nn.Embedding(len(obj_colors), 32)

        self.object_proj = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.obj_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.policy = nn.Sequential(
            nn.Linear(d_model * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, img, task, prompt, obj_role, obj_name, obj_color, obj_centroid, obj_audio, obj_mask):
        img_feat = self.cnn(img)

        task_feat = self.task_emb(task)

        word_e = self.word_emb(prompt)
        prompt_mask = prompt != 0
        prompt_len = prompt_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
        prompt_feat = (word_e * prompt_mask.unsqueeze(-1)).sum(dim=1) / prompt_len

        role_e = self.role_emb(obj_role)
        name_e = self.name_emb(obj_name)
        color_e = self.color_emb(obj_color)

        obj_x = torch.cat([role_e, name_e, color_e, obj_centroid], dim=-1)
        obj_tokens = self.object_proj(obj_x)

        # Transformer expects True for padding positions
        padding_mask = ~obj_mask
        obj_encoded = self.obj_transformer(obj_tokens, src_key_padding_mask=padding_mask)

        valid = obj_mask.unsqueeze(-1).float()
        obj_feat = (obj_encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        fused = torch.cat([img_feat, task_feat, prompt_feat, obj_feat], dim=1)
        return self.policy(fused)


model = TransformerAudioVIMA()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

best_test_loss = float("inf")
best_epoch = -1

for epoch in range(150):
    model.train()

    pred = model(
        X_img_tr, X_task_tr, X_prompt_tr,
        X_orole_tr, X_oname_tr, X_ocolor_tr,
        X_ocent_tr, X_oaudio_tr, X_omask_tr
    )

    loss = loss_fn(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(
            X_img_te, X_task_te, X_prompt_te,
            X_orole_te, X_oname_te, X_ocolor_te,
            X_ocent_te, X_oaudio_te, X_omask_te
        )

        test_loss = loss_fn(test_pred, Y_test)
        pose0_err = torch.norm(test_pred[:, 0:2] - Y_test[:, 0:2], dim=1).mean()
        pose1_err = torch.norm(test_pred[:, 2:4] - Y_test[:, 2:4], dim=1).mean()

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "metadata": metadata},
                "transformer_baseline_100eps_best.pt",
            )

    if epoch % 5 == 0 or epoch == 149:
        print(
            f"epoch {epoch:03d} "
            f"train_loss={loss.item():.6f} "
            f"test_loss={test_loss.item():.6f} "
            f"pose0_err={pose0_err.item():.4f} "
            f"pose1_err={pose1_err.item():.4f}"
        )

torch.save(
    {"model_state": model.state_dict(), "metadata": metadata},
    "transformer_baseline_100eps_final.pt",
)


print("saved transformer_audiovima_final.pt")
print("best_epoch:", best_epoch)
print("best_test_loss:", best_test_loss)
