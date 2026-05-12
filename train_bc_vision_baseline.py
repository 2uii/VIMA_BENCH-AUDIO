import os, pickle, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

DATA_ROOT = "data_audio_train/sweep_without_exceeding"
DATA_PATH = "bc_multifeature_dataset.pkl"

random.seed(42)
torch.manual_seed(42)

data = pickle.load(open(DATA_PATH, "rb"))
samples = data["samples"]
metadata = data["metadata"]

roles = metadata["roles"]
obj_names = metadata["obj_names"]
obj_colors = metadata["obj_colors"]

role2idx = {x: i for i, x in enumerate(roles)}
name2idx = {x: i for i, x in enumerate(obj_names)}
color2idx = {x: i for i, x in enumerate(obj_colors)}

random.shuffle(samples)
split = int(0.8 * len(samples))
train_samples = samples[:split]
test_samples = samples[split:]


def load_image(ep):
    path = os.path.join(DATA_ROOT, ep, "rgb_top", "0.jpg")
    img = Image.open(path).convert("RGB").resize((128, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def build_tensors(rows):
    imgs, role_ids, name_ids, color_ids, centroids, targets = [], [], [], [], [], []

    for s in rows:
        chosen = None
        for p in s["placeholders"]:
            if p["role"] == "swept_obj":
                chosen = p
                break
        if chosen is None:
            chosen = s["placeholders"][0]

        imgs.append(load_image(s["episode"]))
        role_ids.append(role2idx[chosen["role"]])
        name_ids.append(name2idx[chosen["obj_name"]])
        color_ids.append(color2idx[chosen["obj_color"]])
        centroids.append(chosen["centroid"])
        targets.append(s["target"])

    return (
        torch.tensor(np.array(imgs), dtype=torch.float32),
        torch.tensor(role_ids, dtype=torch.long),
        torch.tensor(name_ids, dtype=torch.long),
        torch.tensor(color_ids, dtype=torch.long),
        torch.tensor(np.array(centroids), dtype=torch.float32),
        torch.tensor(np.array(targets), dtype=torch.float32),
    )


X_img_tr, X_role_tr, X_name_tr, X_color_tr, X_cent_tr, Y_train = build_tensors(train_samples)
X_img_te, X_role_te, X_name_te, X_color_te, X_cent_te, Y_test = build_tensors(test_samples)


class VisionBaselineBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 16, 128),
            nn.ReLU(),
        )

        self.role_emb = nn.Embedding(len(roles), 8)
        self.name_emb = nn.Embedding(len(obj_names), 8)
        self.color_emb = nn.Embedding(len(obj_colors), 16)

        input_dim = 128 + 8 + 8 + 16 + 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, img, role, name, color, centroid):
        img_feat = self.cnn(img)
        role_e = self.role_emb(role)
        name_e = self.name_emb(name)
        color_e = self.color_emb(color)

        x = torch.cat([img_feat, role_e, name_e, color_e, centroid], dim=1)
        return self.net(x)


model = VisionBaselineBCPolicy()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

best_test_loss = float("inf")
best_epoch = -1

for epoch in range(100):
    model.train()

    pred = model(X_img_tr, X_role_tr, X_name_tr, X_color_tr, X_cent_tr)
    loss = loss_fn(pred, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_img_te, X_role_te, X_name_te, X_color_te, X_cent_te)
        test_loss = loss_fn(test_pred, Y_test)

        pose0_err = torch.norm(test_pred[:, 0:2] - Y_test[:, 0:2], dim=1).mean()
        pose1_err = torch.norm(test_pred[:, 2:4] - Y_test[:, 2:4], dim=1).mean()

        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "metadata": metadata},
                "bc_vision_baseline_policy_best.pt",
            )

    if epoch % 5 == 0 or epoch == 99:
        print(
            f"epoch {epoch:03d} "
            f"train_loss={loss.item():.6f} "
            f"test_loss={test_loss.item():.6f} "
            f"pose0_err={pose0_err.item():.4f} "
            f"pose1_err={pose1_err.item():.4f}"
        )

torch.save(
    {"model_state": model.state_dict(), "metadata": metadata},
    "bc_vision_baseline_policy.pt",
)

print("saved bc_vision_baseline_policy.pt")
print("best_epoch:", best_epoch)
print("best_test_loss:", best_test_loss)
