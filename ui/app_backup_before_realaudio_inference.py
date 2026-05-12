import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import os
import streamlit as st
from vima_bench import VIMAEnvBase
import numpy as np
from PIL import Image
import time
import os

def run_preview(task):
    env = VIMAEnvBase(
        task=task,
        modalities=["rgb"],
        display_debug_window=False,
        hide_arm_rgb=False,
    )

    obs = env.reset()

    for _ in range(5):
        obs, _, _, _ = env.step()
        time.sleep(0.05)

    if "top" in obs["rgb"]:
        frame = obs["rgb"]["top"]
    else:
        view = list(obs["rgb"].keys())[0]
        frame = obs["rgb"][view]

    frame = np.transpose(frame, (1, 2, 0)).astype(np.uint8)

    os.makedirs("backend", exist_ok=True)
    Image.fromarray(frame).save("backend/latest_frame.png")

    env.close()

st.set_page_config(page_title="Audio-VIMA Workbench", layout="wide")

PROMPT_PATH = "prompt.json"
FRAME_PATH = "backend/latest_frame.png"

st.title("Audio-VIMA Workbench")

tab1, tab2, tab3 = st.tabs(["Prompt", "Observe", "Files"])

with tab1:
    st.header("Prompt Panel")

    task = st.selectbox(
        "Task",
        [
            "sweep_without_exceeding",
            "sweep_without_touching",
            "rotate",
            "scene_understanding",
            "visual_manipulation",
            "novel_adj",
            "novel_adj_and_noun",
            "novel_noun",
            "twist",
            "follow_motion",
            "follow_order",
            "rearrange",
            "manipulate_old_neighbor",
            "pick_in_order_then_restore",
            "rearrange_then_restore",
            "same_shape",
            "same_texture",
        ],
    )

    obj = st.selectbox(
        "Object",
        ["small_block", "line", "three-sided_rectangle"],
    )

    color = st.selectbox(
        "Color",
        ["red", "blue", "wooden"],
    )

    audio = st.selectbox(
        "Audio Identity",
        ["obj1.wav", "obj2.wav", "hunk_hunk.wav"],
    )

    prompt = {
        "task": task,
        "object": obj,
        "color": color,
        "audio_identity": audio,
    }

    st.subheader("Generated Prompt")
    st.json(prompt)

    if st.button("Save Prompt"):
        with open(PROMPT_PATH, "w") as f:
            json.dump(prompt, f, indent=2)
        st.success(f"Saved prompt to {PROMPT_PATH}")
    if st.button("Run Preview"):
        st.info("Running simulation...")
        run_preview(task)
        st.success("Preview generated!")

with tab2:
    st.header("Observe")

    if os.path.exists(FRAME_PATH):
        st.image(FRAME_PATH, caption="Latest Robot Frame")
    else:
        st.info("No frame available yet.")

with tab3:
    st.header("Files")

    if os.path.exists(PROMPT_PATH):
        st.subheader("Current prompt.json")
        with open(PROMPT_PATH, "r") as f:
            st.code(f.read(), language="json")
    else:
        st.info("prompt.json does not exist yet.")

    st.write("Project root:", os.getcwd())

