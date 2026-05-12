import pickle
import torch.nn as nn

from infer_realaudio_audiovima import (
    TransformerAudioVIMA,
    CKPT_PATH,
    tokenize_prompt,
    encode_prompt,
)
import numpy as np
import torch
from collections import OrderedDict

from vima_bench import make
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

print("=== LIVE AUDIO-VIMA ENV TEST ===")

# --------------------------------------------------
# create live environment
# --------------------------------------------------

env = make(
    "instruction_following/rotate",
    modalities=["rgb"],
)

wrapper = AudioIdentityWrapper(
    env,
    object_sound_map={
        1: "impact",
        2: "tingting",
        3: "alarm",
    },
    debug=True,
)

obs = wrapper.reset()

print("Environment reset OK")

# --------------------------------------------------
# load trained transformer
# --------------------------------------------------

# --------------------------------------------------
# load trained transformer
# --------------------------------------------------

model = TransformerAudioVIMA()

ckpt = torch.load(CKPT_PATH, map_location="cpu")

if "model_state_dict" in ckpt:

    model.load_state_dict(ckpt["model_state_dict"])

elif "model_state" in ckpt:

    model.load_state_dict(ckpt["model_state"])

else:

    model.load_state_dict(ckpt)

model.eval()

print("Model loaded")
# --------------------------------------------------
# fake model prediction for now
# later this becomes transformer output
# --------------------------------------------------

# --------------------------------------------------
# build SIMPLE live model inputs
# --------------------------------------------------

import torch.nn.functional as F

rgb = obs["rgb"]["top"].astype("float32") / 255.0
X_img = torch.tensor(rgb, dtype=torch.float32).unsqueeze(0)
X_img = F.interpolate(X_img, size=(64, 128), mode="bilinear", align_corners=False)

X_task = torch.zeros(1, dtype=torch.long)

prompt_tokens = tokenize_prompt(wrapper.prompt)
prompt_ids = encode_prompt(prompt_tokens)
X_prompt = torch.tensor([prompt_ids], dtype=torch.long)

print("Live prompt:", wrapper.prompt)

# --------------------------------------------------
# build REAL live object/audio tensors
# --------------------------------------------------

prompt_assets = wrapper.env.prompt_assets

roles = list(prompt_assets.keys())

max_objs = 3

role_ids = []
name_ids = []
color_ids = []
audio_feats = []
centroids = []
mask = []

for role in roles[:max_objs]:

    asset = prompt_assets[role]

    audio_token = asset.get("audio_token", np.zeros(768, dtype=np.float32))

    role_ids.append(1)
    name_ids.append(1)
    color_ids.append(1)

    audio_feats.append(np.asarray(audio_token, dtype=np.float32))

    segm = asset.get("segm", {})
    seg_mask = None

    if isinstance(segm, dict):
        seg_mask = segm.get("mask", None)

    if seg_mask is not None:
        ys, xs = np.where(seg_mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            cx = float(xs.mean() / seg_mask.shape[1])
            cy = float(ys.mean() / seg_mask.shape[0])
            centroid = np.array([cx, cy], dtype=np.float32)
        else:
            centroid = np.array([0.5, 0.5], dtype=np.float32)
    else:
        centroid = np.array([0.5, 0.5], dtype=np.float32)

    centroids.append(centroid)

    mask.append(True)

while len(role_ids) < max_objs:

    role_ids.append(0)
    name_ids.append(0)
    color_ids.append(0)

    audio_feats.append(np.zeros(768, dtype=np.float32))

    centroids.append(np.zeros(2, dtype=np.float32))

    mask.append(False)

X_role = torch.tensor([role_ids], dtype=torch.long)

X_name = torch.tensor([name_ids], dtype=torch.long)

X_color = torch.tensor([color_ids], dtype=torch.long)

X_audio = torch.tensor([audio_feats], dtype=torch.float32)

X_ocent = torch.tensor([centroids], dtype=torch.float32)

X_omask = torch.tensor([mask], dtype=torch.bool)

print("Live objects:", roles)
print("X_audio shape:", X_audio.shape)

with torch.no_grad():

    pred = model(
        X_img,
        X_task,
        X_prompt,
        X_role,
        X_name,
        X_color,
        X_audio,
        X_ocent,
        X_omask,
    )

pred_np = pred.squeeze(0).cpu().numpy()

print("Model predicted:", pred_np)


# --------------------------------------------------
# convert prediction -> env action
# --------------------------------------------------

action = env.action_space.sample()

action["pose0_position"] = np.array(
    [pred_np[0], pred_np[1]],
    dtype=np.float32,
)

action["pose1_position"] = np.array(
    [pred_np[2], pred_np[3]],
    dtype=np.float32,
)

print("\nExecuting action...")

# --------------------------------------------------
# execute multiple steps
# --------------------------------------------------

for step in range(5):

    obs, reward, done, info = wrapper.step(action)

    print(f"\nSTEP {step}")

    audio_events = info.get("audio_events", [])
    audio_memory = info.get("audio_memory", {})

    print("audio_events:", len(audio_events))
    print("audio_memory keys:", list(audio_memory.keys()))

    if audio_events:
        ev = audio_events[-1]

        print("heard sound:", ev["sound"])
        print("event_type:", ev["event_type"])
        print("token_dim:", len(ev["audio_token"]))

    if done:
        break

env.close()

print("\n=== TEST COMPLETE ===")

