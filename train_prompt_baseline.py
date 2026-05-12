import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

data = pickle.load(open("prompt_dataset.pkl", "rb"))

random.seed(42)
torch.manual_seed(42)

# shuffle and split
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# vocab from all data
obj_names = sorted({d["obj_name"] for d in data})
colors = sorted({d["obj_color"] for d in data})
queries = sorted({d["query"] for d in data})

name2idx = {x: i for i, x in enumerate(obj_names)}
color2idx = {x: i for i, x in enumerate(colors)}
query2idx = {x: i for i, x in enumerate(queries)}

def build_tensors(rows):
    X = []
    Y = []
    for d in rows:
        X.append([
            name2idx[d["obj_name"]],
            color2idx[d["obj_color"]],
            query2idx[d["query"]],
        ])
        Y.append(d["label"])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    return X, Y

X_train, Y_train = build_tensors(train_data)
X_test, Y_test = build_tensors(test_data)

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_name = nn.Embedding(len(obj_names), 16)
        self.embed_color = nn.Embedding(len(colors), 16)
        self.embed_query = nn.Embedding(len(queries), 16)

        self.fc = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        name = self.embed_name(x[:, 0])
        color = self.embed_color(x[:, 1])
        query = self.embed_query(x[:, 2])
        out = torch.cat([name, color, query], dim=1)
        return self.fc(out)

model = Baseline()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

for epoch in range(20):
    model.train()
    pred_train = model(X_train)
    train_loss = loss_fn(pred_train, Y_train)

    opt.zero_grad()
    train_loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = ((pred_train > 0.5) == Y_train).float().mean().item()

        pred_test = model(X_test)
        test_loss = loss_fn(pred_test, Y_test)
        test_acc = ((pred_test > 0.5) == Y_test).float().mean().item()

    print(
        f"epoch {epoch} "
        f"train_loss={train_loss.item():.4f} "
        f"train_acc={train_acc:.3f} "
        f"test_loss={test_loss.item():.4f} "
        f"test_acc={test_acc:.3f}"
    )

torch.save(model.state_dict(), "baseline_split.pt")
print("Saved baseline_split.pt")

