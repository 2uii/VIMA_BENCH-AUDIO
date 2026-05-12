import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import random
import numpy as np
from PIL import Image
import streamlit as st
import torch

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

sample_idx = st.sidebar.number_input(
    "Sample index",
    min_value=0,
    max_value=len(samples) - 1,
    value=0,
    step=1,
)

sample = samples[sample_idx]

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
        st.image(sample["image_path"], caption=f"Task: {sample['task']}")

    with col2:
        st.subheader("Prediction")
        st.write("Prompt:")
        st.code(user_prompt)

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

st.subheader("Current Sample Info")
st.write("Task:", sample["task"])
st.write("Image path:", sample["image_path"])
st.write("Number of placeholders:", len(sample["placeholders"]))

