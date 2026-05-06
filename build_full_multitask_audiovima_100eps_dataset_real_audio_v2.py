import os
import pickle
import re
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from vima_bench.audio.audio_encoder import Wav2Vec2TokenEncoder

audio_encoder = Wav2Vec2TokenEncoder(token_dim=768)

def encode_audio_from_wav(wav_path):
    try:
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        audio_feat = audio_encoder._encode(waveform).squeeze(0)
        return audio_feat.detach().cpu().numpy().astype("float32")
    except Exception:
        return np.zeros(768, dtype=np.float32)

DATA_ROOT = "data_audio_train_full_100eps"
OUT_PATH = "full_multitask_audiovima_100eps_real_audio_v2_dataset.pkl"


def tokenize_prompt(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9_{} ]+", " ", text)
    return [t for t in text.split() if t]


def centroid_from_mask(segm):
    if segm is None:
        return np.array([0.0, 0.0], dtype=np.float32)

    segm = np.asarray(segm)
    mask = segm > 0

    if mask.sum() == 0:
        return np.array([0.0, 0.0], dtype=np.float32)

    ys, xs = np.where(mask)
    return np.array([xs.mean() / segm.shape[1], ys.mean() / segm.shape[0]], dtype=np.float32)


def safe_obj_info(segm):
    if not isinstance(segm, dict):
        return {}

    obj_info = segm.get("obj_info", {}) or {}

    if isinstance(obj_info, dict):
        return obj_info

    if isinstance(obj_info, list):
        for item in obj_info:
            if isinstance(item, dict):
                return item

    return {}


def safe_mask(segm):
    if isinstance(segm, dict):
        return segm.get("top", None) if segm.get("top", None) is not None else segm.get("front", None)
    return segm


def choose_action_placeholder(placeholders):
    preferred = [
        "swept_obj",
        "dragged_obj",
        "base_obj",
        "target_obj",
        "obj",
        "constraint",
    ]

    for role in preferred:
        for p in placeholders:
            if p["role"] == role:
                return p

    return placeholders[0]


samples = []
tasks = set()
roles = set()
obj_names = set()
obj_colors = set()
prompt_vocab = set()
skipped = 0

for task_name in sorted(os.listdir(DATA_ROOT)):
    task_dir = os.path.join(DATA_ROOT, task_name)
    if not os.path.isdir(task_dir):
        continue

    for ep in sorted(os.listdir(task_dir)):
        ep_dir = os.path.join(task_dir, ep)
        if not os.path.isdir(ep_dir):
            continue

        traj_path = os.path.join(ep_dir, "trajectory.pkl")
        action_path = os.path.join(ep_dir, "action.pkl")
        img_top = os.path.join(ep_dir, "rgb_top", "0.jpg")
        img_front = os.path.join(ep_dir, "rgb_front", "0.jpg")

        img_path = img_top if os.path.exists(img_top) else img_front

        if not (os.path.exists(traj_path) and os.path.exists(action_path) and os.path.exists(img_path)):
            skipped += 1
            continue

        try:
            traj = pickle.load(open(traj_path, "rb"))
            action = pickle.load(open(action_path, "rb"))
        except Exception:
            skipped += 1
            continue

        if "pose0_position" not in action or "pose1_position" not in action:
            skipped += 1
            continue

        pose0_all = np.asarray(action["pose0_position"], dtype=np.float32).reshape(-1, 2)
        pose1_all = np.asarray(action["pose1_position"], dtype=np.float32).reshape(-1, 2)

        if len(pose0_all) == 0 or len(pose1_all) == 0:
            skipped += 1
            continue

        target = np.concatenate([pose0_all[0], pose1_all[0]], axis=0).astype(np.float32)

        prompt = str(traj.get("prompt", ""))
        prompt_tokens = tokenize_prompt(prompt)
        prompt_vocab.update(prompt_tokens)

        placeholders = []

        for role, asset in traj.get("prompt_assets", {}).items():
            audio_token = asset.get("audio_token", None)
            if audio_token is None:
                continue

            segm = asset.get("segm", {})
            obj_info = safe_obj_info(segm)
            mask = safe_mask(segm)

            obj_name = str(obj_info.get("obj_name", "unknown")).replace(" ", "_")
            obj_color = str(obj_info.get("obj_color", "unknown")).replace(" ", "_")

            roles.add(role)
            obj_names.add(obj_name)
            obj_colors.add(obj_color)

            placeholders.append(
                {
                    "role": role,
                    "obj_name": obj_name,
                    "obj_color": obj_color,
                    "centroid": centroid_from_mask(mask),
                    "audio_token": encode_audio_from_wav(asset.get("wav_path", "")),
                }
            )

        if not placeholders:
            skipped += 1
            continue

        chosen = choose_action_placeholder(placeholders)

        tasks.add(task_name)

        samples.append(
            {
                "task": task_name,
                "episode": ep,
                "image_path": img_path,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "placeholders": placeholders,
                "chosen": chosen,
                "target": target,
                "success": bool(traj.get("success", False)),
                "failure": bool(traj.get("failure", False)),
            }
        )

metadata = {
    "tasks": sorted(tasks),
    "roles": sorted(roles),
    "obj_names": sorted(obj_names),
    "obj_colors": sorted(obj_colors),
    "prompt_vocab": sorted(prompt_vocab),
}

pickle.dump(
    {
        "samples": samples,
        "metadata": metadata,
        "skipped": skipped,
    },
    open(OUT_PATH, "wb"),
)

print("saved:", OUT_PATH)
print("samples:", len(samples))
print("skipped:", skipped)
print("tasks:", metadata["tasks"])
print("roles:", metadata["roles"])
print("n_obj_names:", len(metadata["obj_names"]))
print("n_obj_colors:", len(metadata["obj_colors"]))
print("n_prompt_vocab:", len(metadata["prompt_vocab"]))
