import pickle
import random
import numpy as np

from vima_bench import make
from vima_bench.tasks import ALL_TASKS
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

SAVE_PATH = "embodied_audio_sequence_dataset.pkl"
NUM_EPISODES = 1700

TASKS = sorted(list(ALL_TASKS.keys()))

dataset = []

print("=== GENERATING EMBODIED AUDIO SEQUENCE DATASET ===")


def centroid_from_asset(asset):
    segm = asset.get("segm", None)

    if isinstance(segm, dict):
        mask = segm.get("mask", None)
    else:
        mask = segm

    if mask is None:
        return np.array([0.5, 0.5], dtype=np.float32)

    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = mask.squeeze()

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.5, 0.5], dtype=np.float32)

    cx = float(xs.mean() / mask.shape[1])
    cy = float(ys.mean() / mask.shape[0])

    return np.array([cx, cy], dtype=np.float32)


for task in TASKS:
    print("\nTASK:", task)

    env = make(task, modalities=["rgb"])

    wrapper = AudioIdentityWrapper(
        env,
        object_sound_map={
            1: "impact",
            2: "tingting",
            3: "alarm",
        },
        debug=False,
    )

    for ep in range(NUM_EPISODES // len(TASKS)):
        try:
            obs = wrapper.reset()
        except Exception as e:
            print("RESET FAILED:", task, "|", e)
            continue
        prompt_assets = wrapper.env.prompt_assets

        candidates = []

        for role, asset in prompt_assets.items():
            audio_token = np.asarray(
                asset.get("audio_token", np.zeros(768)),
                dtype=np.float32,
            )

            if audio_token.shape[0] != 768:
                continue

            centroid = centroid_from_asset(asset)

            candidates.append({
                "role": role,
                "audio_token": audio_token,
                "centroid": centroid,
                "audio_id": asset.get("audio_id", "unknown"),
                "wav_path": asset.get("wav_path", "unknown"),
            })

        if len(candidates) < 2:
            continue

        random.shuffle(candidates)

        heard_sequence = candidates[:2]

        target_position = random.choice([0, 1])
        target = heard_sequence[target_position]

        prompt = (
            "Pick the object that sounded first."
            if target_position == 0
            else "Pick the object that sounded second."
        )

        target_index = None
        for i, c in enumerate(candidates):
            if c["role"] == target["role"]:
                target_index = i
                break

        if target_index is None:
            continue

        sample = {
            "task": task,
            "prompt": prompt,
            "rgb_top": obs["rgb"]["top"],

            "candidates": candidates,

            "heard_sequence": [
                {
                    "step": i,
                    "role": ev["role"],
                    "audio_token": ev["audio_token"],
                    "audio_id": ev["audio_id"],
                    "wav_path": ev["wav_path"],
                }
                for i, ev in enumerate(heard_sequence)
            ],

            "target_role": target["role"],
            "target_index": target_index,
            "target_position": target_position,
            "target_centroid": target["centroid"],
        }

        dataset.append(sample)

        print(
            "sample",
            len(dataset),
            "| task:",
            task,
            "| candidates:",
            len(candidates),
            "| target:",
            target["role"],
            "| centroid:",
            target["centroid"],
        )

    env.close()

with open(SAVE_PATH, "wb") as f:
    pickle.dump(dataset, f)

print("\nSaved:", SAVE_PATH)
print("Samples:", len(dataset))
