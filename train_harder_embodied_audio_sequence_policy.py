import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATA_PATH = "balanced_harder_audio_dataset.pkl"
OUT_PATH = "harder_embodied_audio_sequence_policy.pt"

MAX_OBJECTS = 8
MAX_SEQ = 3

EPOCHS = 200
LR = 3e-4

data = pickle.load(open(DATA_PATH, "rb"))

roles = sorted(list(set(
    c["role"]
    for s in data
    for c in s["candidates"]
)))

role2idx = {r: i for i, r in enumerate(roles)}


def encode_sample(sample):
    rgb = sample["rgb_top"].astype("float32") / 255.0

    img = torch.tensor(rgb).unsqueeze(0)

    img = nn.functional.interpolate(
        img,
        size=(64, 128),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    object_roles = []
    object_audio = []
    object_centroids = []
    object_mask = []

    for c in sample["candidates"][:MAX_OBJECTS]:
        object_roles.append(role2idx[c["role"]])

        object_audio.append(
            np.asarray(c["audio_token"], dtype=np.float32)
        )

        object_centroids.append(
            np.asarray(c["centroid"], dtype=np.float32)
        )

        object_mask.append(1)

    while len(object_roles) < MAX_OBJECTS:
        object_roles.append(0)
        object_audio.append(np.zeros(768, dtype=np.float32))
        object_centroids.append(np.zeros(2, dtype=np.float32))
        object_mask.append(0)

    seq_audio = []
    seq_pos = []

    for i, ev in enumerate(sample["heard_sequence"][:MAX_SEQ]):
        seq_audio.append(
            np.asarray(ev["audio_token"], dtype=np.float32)
        )
        seq_pos.append(i)

    while len(seq_audio) < MAX_SEQ:
        seq_audio.append(np.zeros(768, dtype=np.float32))
        seq_pos.append(0)

    return (
        img,

        torch.tensor(object_roles, dtype=torch.long),

        torch.tensor(
            np.array(object_audio),
            dtype=torch.float32,
        ),

        torch.tensor(
            np.array(object_centroids),
            dtype=torch.float32,
        ),

        torch.tensor(
            object_mask,
            dtype=torch.bool,
        ),

        torch.tensor(
            np.array(seq_audio),
            dtype=torch.float32,
        ),

        torch.tensor(
            seq_pos,
            dtype=torch.long,
        ),

        torch.tensor(
            sample["target_position"],
            dtype=torch.long,
        ),

        torch.tensor(
            sample["target_index"],
            dtype=torch.long,
        ),

        torch.tensor(
            sample["target_centroid"],
            dtype=torch.float32,
        ),
    )


encoded = [encode_sample(s) for s in data]
random.shuffle(encoded)

split = int(0.8 * len(encoded))

train_data = encoded[:split]
test_data = encoded[split:]


class EmbodiedAudioPolicy(nn.Module):
    def __init__(self, n_roles):
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

        self.role_emb = nn.Embedding(n_roles, 32)
        self.pos_emb = nn.Embedding(MAX_SEQ, 32)

        self.seq_proj = nn.Sequential(
            nn.Linear(768 + 32, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.obj_proj = nn.Sequential(
            nn.Linear(32 + 768 + 2 + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.selector = nn.Sequential(
            nn.Linear(d_model + d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.centroid_head = nn.Sequential(
            nn.Linear(d_model + d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        img,
        obj_roles,
        obj_audio,
        obj_centroids,
        obj_mask,
        seq_audio,
        seq_pos,
        query_pos,
    ):
        img_feat = self.cnn(img)

        pos_e = self.pos_emb(seq_pos)

        seq_x = torch.cat([seq_audio, pos_e], dim=-1)

        seq_feat_all = self.seq_proj(seq_x)

        batch_idx = torch.arange(seq_feat_all.shape[0])

        query_feat = seq_feat_all[
            batch_idx,
            query_pos,
        ]

        role_e = self.role_emb(obj_roles)

        query_expand = query_feat.unsqueeze(1).expand(
            -1,
            obj_audio.shape[1],
            -1,
        )

        obj_x = torch.cat([
            role_e,
            obj_audio,
            obj_centroids,
            query_expand,
        ], dim=-1)

        obj_feat = self.obj_proj(obj_x)

        img_expand = img_feat.unsqueeze(1).expand(
            -1,
            obj_audio.shape[1],
            -1,
        )

        fused = torch.cat([
            img_expand,
            obj_feat,
        ], dim=-1)

        logits = self.selector(fused).squeeze(-1)

        logits = logits.masked_fill(
            ~obj_mask,
            -1e9,
        )

        centroid_pred = self.centroid_head(
            fused.mean(dim=1)
        )

        return logits, centroid_pred


def make_batch(rows):
    cols = list(zip(*rows))
    return [torch.stack(c) for c in cols]


if __name__ == "__main__":
    model = EmbodiedAudioPolicy(
        n_roles=len(roles)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
    )

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    best_acc = 0.0

    for epoch in range(EPOCHS):
        random.shuffle(train_data)

        model.train()

        total_loss = 0.0

        for i in range(0, len(train_data), 16):
            batch = train_data[i:i+16]

            (
                img,
                obj_roles,
                obj_audio,
                obj_centroids,
                obj_mask,
                seq_audio,
                seq_pos,
                query_pos,
                target_idx,
                target_centroid,
            ) = make_batch(batch)

            logits, centroid_pred = model(
                img,
                obj_roles,
                obj_audio,
                obj_centroids,
                obj_mask,
                seq_audio,
                seq_pos,
                query_pos,
            )

            cls_loss = cls_loss_fn(
                logits,
                target_idx,
            )

            reg_loss = reg_loss_fn(
                centroid_pred,
                target_centroid,
            )

            loss = cls_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        centroid_err = 0.0

        with torch.no_grad():
            for i in range(0, len(test_data), 16):
                batch = test_data[i:i+16]

                (
                    img,
                    obj_roles,
                    obj_audio,
                    obj_centroids,
                    obj_mask,
                    seq_audio,
                    seq_pos,
                    query_pos,
                    target_idx,
                    target_centroid,
                ) = make_batch(batch)

                logits, centroid_pred = model(
                    img,
                    obj_roles,
                    obj_audio,
                    obj_centroids,
                    obj_mask,
                    seq_audio,
                    seq_pos,
                    query_pos,
                )

                pred = logits.argmax(dim=1)

                correct += (
                    pred == target_idx
                ).sum().item()

                total += target_idx.numel()

                centroid_err += (
                    (
                        centroid_pred
                        - target_centroid
                    ) ** 2
                ).mean().item()

        acc = correct / max(total, 1)

        centroid_err /= max(
            len(test_data) / 16,
            1,
        )

        if acc > best_acc:
            best_acc = acc

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "roles": roles,
                    "best_acc": best_acc,
                },
                OUT_PATH,
            )

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"epoch {epoch:03d} "
                f"loss={total_loss:.4f} "
                f"test_acc={acc:.4f} "
                f"centroid_mse={centroid_err:.6f}"
            )

    print("saved:", OUT_PATH)
    print("best_acc:", best_acc)
