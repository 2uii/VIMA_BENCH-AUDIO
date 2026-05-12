import pickle
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

DATA_PATH = "full_multitask_audiovima_100eps_dataset.pkl"
CKPT_PATH = "transformer_audiovima_best.pt"
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
word2idx = {x: i + 1 for i, x in enumerate(prompt_vocab)}

random.shuffle(samples)
split = int(0.8 * len(samples))
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
    role_ids, name_ids, color_ids = [], [], []
    centroids, audio_tokens, mask = [], [], []

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
    imgs, task_ids, prompt_ids = [], [], []
    obj_roles, obj_names_, obj_colors_ = [], [], []
    obj_centroids, obj_audio, obj_masks = [], [], []
    targets = []

    for s in rows:
        r, n, c, cent, aud, m = encode_objects(s["placeholders"])

        imgs.append(load_image(s["image_path"]))
        task_ids.append(task2idx[s["task"]])
        prompt_ids.append(encode_prompt(s["prompt_tokens"]))

        obj_roles.append(r)
        obj_names_.append(n)
        obj_colors_.append(c)
        obj_centroids.append(cent)
        obj_audio.append(aud)
        obj_masks.append(m)

        targets.append(s["target"])

    return (
        torch.tensor(np.array(imgs), dtype=torch.float32),
        torch.tensor(task_ids, dtype=torch.long),
        torch.tensor(np.array(prompt_ids), dtype=torch.long),
        torch.tensor(np.array(obj_roles), dtype=torch.long),
        torch.tensor(np.array(obj_names_), dtype=torch.long),
        torch.tensor(np.array(obj_colors_), dtype=torch.long),
        torch.tensor(np.array(obj_centroids), dtype=torch.float32),
        torch.tensor(np.array(obj_audio), dtype=torch.float32),
        torch.tensor(np.array(obj_masks), dtype=torch.bool),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )


class TransformerAudioVIMA(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 128

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
            nn.Linear(32 + 32 + 32 + 2 + 768, d_model),
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

        obj_x = torch.cat([role_e, name_e, color_e, obj_centroid, obj_audio], dim=-1)
        obj_tokens = self.object_proj(obj_x)

        padding_mask = ~obj_mask
        obj_encoded = self.obj_transformer(obj_tokens, src_key_padding_mask=padding_mask)

        valid = obj_mask.unsqueeze(-1).float()
        obj_feat = (obj_encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        fused = torch.cat([img_feat, task_feat, prompt_feat, obj_feat], dim=1)
        return self.policy(fused)


X_img, X_task, X_prompt, X_orole, X_oname, X_ocolor, X_ocent, X_oaudio, X_omask, Y = build_tensors(test_samples)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = TransformerAudioVIMA()
model.load_state_dict(ckpt["model_state"])
model.eval()

loss_fn = nn.MSELoss()

with torch.no_grad():
    pred_clean = model(X_img, X_task, X_prompt, X_orole, X_oname, X_ocolor, X_ocent, X_oaudio, X_omask)
    clean_loss = loss_fn(pred_clean, Y)

    shuffled_audio = X_oaudio[torch.randperm(X_oaudio.shape[0])]
    pred_shuffle = model(X_img, X_task, X_prompt, X_orole, X_oname, X_ocolor, X_ocent, shuffled_audio, X_omask)
    shuffle_loss = loss_fn(pred_shuffle, Y)

    zero_audio = torch.zeros_like(X_oaudio)
    pred_zero = model(X_img, X_task, X_prompt, X_orole, X_oname, X_ocolor, X_ocent, zero_audio, X_omask)
    zero_loss = loss_fn(pred_zero, Y)

print("clean_audio_loss:", clean_loss.item())
print("shuffled_audio_loss:", shuffle_loss.item())
print("zero_audio_loss:", zero_loss.item())
print("delta_shuffle:", shuffle_loss.item() - clean_loss.item())
print("delta_zero:", zero_loss.item() - clean_loss.item())
