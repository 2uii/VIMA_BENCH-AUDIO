import pickle
import random
import torch

from train_harder_embodied_audio_sequence_policy import (
    EmbodiedAudioPolicy,
    encode_sample,
    make_batch,
)

DATA_PATH = "balanced_harder_audio_dataset.pkl"
CKPT_PATH = "harder_embodied_audio_sequence_policy.pt"

data = pickle.load(open(DATA_PATH, "rb"))
ckpt = torch.load(CKPT_PATH, map_location="cpu")

model = EmbodiedAudioPolicy(n_roles=len(ckpt["roles"]))
model.load_state_dict(ckpt["model_state"])
model.eval()

encoded = [encode_sample(s) for s in data]

random.seed(42)
random.shuffle(encoded)

split = int(0.8 * len(encoded))
test_data = encoded[split:]


def evaluate(mode="clean"):
    correct = 0
    total = 0
    centroid_mse = 0.0
    batches = 0

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

            if mode == "zero_audio":
                seq_audio = torch.zeros_like(seq_audio)

            elif mode == "shuffle_order":
                seq_audio = seq_audio.flip(dims=[1])

            elif mode == "wrong_query":
                query_pos = (query_pos + 1) % 3

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

            correct += (pred == target_idx).sum().item()
            total += target_idx.numel()

            centroid_mse += ((centroid_pred - target_centroid) ** 2).mean().item()
            batches += 1

    return correct / max(total, 1), centroid_mse / max(batches, 1)


clean_acc, clean_mse = evaluate("clean")
shuffle_acc, shuffle_mse = evaluate("shuffle_order")
zero_acc, zero_mse = evaluate("zero_audio")
wrong_acc, wrong_mse = evaluate("wrong_query")

print("clean_acc:", clean_acc)
print("shuffle_order_acc:", shuffle_acc)
print("zero_audio_acc:", zero_acc)
print("wrong_query_acc:", wrong_acc)

print("drop_shuffle_order:", clean_acc - shuffle_acc)
print("drop_zero_audio:", clean_acc - zero_acc)
print("drop_wrong_query:", clean_acc - wrong_acc)

print("clean_centroid_mse:", clean_mse)
print("shuffle_centroid_mse:", shuffle_mse)
print("zero_centroid_mse:", zero_mse)
print("wrong_query_centroid_mse:", wrong_mse)
