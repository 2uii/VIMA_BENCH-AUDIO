import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import random
import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch
from collections import OrderedDict

def pred_to_env_action(pred_np, env):
    action = env.action_space.sample()

    pose0_x, pose0_y, pose1_x, pose1_y = pred_np

    pose0_x = float(np.clip(pose0_x, 0.25, 0.75))
    pose0_y = float(np.clip(pose0_y, -0.5, 0.5))
    pose1_x = float(np.clip(pose1_x, 0.25, 0.75))
    pose1_y = float(np.clip(pose1_y, -0.5, 0.5))

    action["pose0_position"] = np.array([pose0_x, pose0_y], dtype=np.float32)
    action["pose1_position"] = np.array([pose1_x, pose1_y], dtype=np.float32)

    # keep sampled rotations for now
    return action

def draw_action_overlay(image_path, pred_np):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size

    pose0_x, pose0_y, pose1_x, pose1_y = pred_np
    # keep predictions inside valid image range
    pose0_x = max(0.0, min(1.0, pose0_x))
    pose0_y = max(0.0, min(1.0, pose0_y))
    pose1_x = max(0.0, min(1.0, pose1_x))
    pose1_y = max(0.0, min(1.0, pose1_y))

    # Convert normalized coordinates to image pixels.
    x0 = int(pose0_x * w)
    y0 = int((1.0 - pose0_y) * h)

    x1 = int(pose1_x * w)
    y1 = int((1.0 - pose1_y) * h)

    r = 6

    # pick point
    draw.ellipse((x0-r, y0-r, x0+r, y0+r), fill="red")
    draw.text((x0 + 8, y0), "pick", fill="red")

    # place point
    draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill="blue")
    draw.text((x1 + 8, y1), "place", fill="blue")

    # arrow/line
    draw.line((x0, y0, x1, y1), fill="yellow", width=3)

    return img

from infer_realaudio_audiovima import (
    TransformerAudioVIMA,
    DATA_PATH,
    CKPT_PATH,
    tokenize_prompt,
    encode_prompt,
    encode_objects,
    load_image,
    task2idx,
)

st.set_page_config(page_title="Audio-VIMA Real-Audio Interface", layout="wide")

st.title("Audio-VIMA Real-Audio Prompting Interface")

@st.cache_resource
def load_model():
    model = TransformerAudioVIMA()
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@st.cache_data
def load_dataset():
    data = pickle.load(open(DATA_PATH, "rb"))
    return data["samples"]

model = load_model()
samples = load_dataset()

st.sidebar.header("Controls")

available_tasks = sorted(list(set(s["task"] for s in samples)))

selected_task = st.sidebar.selectbox(
    "Task",
    available_tasks,
)

task_samples = [s for s in samples if s["task"] == selected_task]

sample_idx = st.sidebar.number_input(
    "Sample index within selected task",
    min_value=0,
    max_value=len(task_samples) - 1,
    value=0,
    step=1,
)

sample = task_samples[sample_idx]

default_prompt = sample.get("prompt", "")
user_prompt = st.text_area(
    "Enter user prompt",
    value=default_prompt,
    height=120,
)

if st.button("Run Audio-VIMA Prediction"):
    r, n, c, cent, aud, m = encode_objects(sample["placeholders"])

    X_img = torch.tensor(load_image(sample["image_path"])).unsqueeze(0).float()
    X_task = torch.tensor([task2idx[sample["task"]]], dtype=torch.long)
    X_prompt = torch.tensor([encode_prompt(tokenize_prompt(user_prompt))], dtype=torch.long)
    X_orole = torch.tensor([r], dtype=torch.long)
    X_oname = torch.tensor([n], dtype=torch.long)
    X_ocolor = torch.tensor([c], dtype=torch.long)
    X_ocent = torch.tensor(np.array([cent]), dtype=torch.float32)
    X_oaudio = torch.tensor(np.array([aud]), dtype=torch.float32)
    X_omask = torch.tensor([m], dtype=torch.bool)

    with torch.no_grad():
        pred = model(
            X_img,
            X_task,
            X_prompt,
            X_orole,
            X_oname,
            X_ocolor,
            X_ocent,
            X_oaudio,
            X_omask,
        )

    pred_np = pred.squeeze(0).numpy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scene")
        overlay_img = draw_action_overlay(sample["image_path"], pred_np)
        st.image(overlay_img, caption=f"Task: {sample['task']} | Red = pick, Blue = place")

    with col2:
        st.subheader("Prediction")
        st.write("Prompt:")
        st.code(user_prompt)
        st.subheader("Trimodal Prompt View")

        st.markdown("**Language modality**")
        st.write(user_prompt)

        st.markdown("**Vision modality**")
        st.write(sample["image_path"])

        st.markdown("**Audio modality / object tokens**")
        audio_rows = []
        for p in sample["placeholders"]:
            audio_token = p.get("audio_token", [])
            audio_rows.append({
                "role": p.get("role", "unknown"),
                "object": p.get("obj_name", "unknown"),
                "color": p.get("obj_color", "unknown"),
                "audio_id": p.get("audio_id", "unknown"),
                "wav_path": p.get("wav_path", "unknown"),
                "audio_dim": len(audio_token),
                "centroid": str(p.get("centroid", "")),
            })

        st.table(audio_rows)       
        st.subheader("Prompt Interpretation Summary")

        st.info(
            "Audio-VIMA interprets this as a trimodal prompt: "
            "the language instruction defines the task, the image provides visual grounding, "
            "and each object carries a real-audio identity token from its assigned WAV file. "
            "The model fuses these modalities to predict the robot pick-and-place action."
        )

        st.write("Predicted action:")
        st.json({
            "pose0_x": float(pred_np[0]),
            "pose0_y": float(pred_np[1]),
            "pose1_x": float(pred_np[2]),
            "pose1_y": float(pred_np[3]),
        })

        if "target" in sample:
            st.write("Ground truth action:")
            st.json({
	                "pose0_x": float(sample["target"][0]),
                "pose0_y": float(sample["target"][1]),
                "pose1_x": float(sample["target"][2]),
                "pose1_y": float(sample["target"][3]),
            })

st.caption("Note: Raw predicted coordinates are shown numerically. Visual markers are clipped to image bounds.")
st.subheader("Current Sample Info")
st.write("Task:", sample["task"])
st.write("Image path:", sample["image_path"])
st.write("Number of placeholders:", len(sample["placeholders"]))
st.subheader("Live Audio Memory")

try:
    from vima_bench import make
    from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

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
        debug=False,
    )

    obs = wrapper.reset()

    for _ in range(3):
        action = pred_to_env_action(pred_np, env)
        obs, reward, done, info = wrapper.step(action)

    audio_memory = info.get("audio_memory", {})

    if audio_memory:
        memory_rows = []

        for obj_id, mem in audio_memory.items():
            memory_rows.append({
                "object_id": obj_id,
                "last_sound": mem["last_sound"],
                "last_step": mem["last_step"],
                "event_count": mem["event_count"],
                "event_type": mem["last_event_type"],
                "token_dim": len(mem["last_audio_token"]),
            })

        st.table(memory_rows)

    else:
        st.info("No audio events heard yet.")

    env.close()

except Exception as e:
    st.error(f"Audio memory demo failed: {e}")

st.subheader("Model Performance Summary")

comparison_rows = [
    {
        "Model": "No-audio baseline",
        "Best Test Loss": 0.030058,
    },
    {
        "Model": "Old Audio-VIMA",
        "Best Test Loss": 0.029948,
    },
    {
        "Model": "Real-audio Audio-VIMA",
        "Best Test Loss": 0.026178,
    },
]

st.table(comparison_rows)

st.subheader("Audio Ablation Results")

ablation_rows = [
    {
        "Condition": "Clean audio",
        "Loss": 0.026178,
    },
    {
        "Condition": "Shuffled audio",
        "Loss": 0.026709,
    },
    {
        "Condition": "Zero audio",
        "Loss": 0.030784,
    },
]

st.table(ablation_rows)

st.success(
    "Real-audio Audio-VIMA achieved the best performance and showed measurable degradation when audio embeddings were shuffled or removed."
)

