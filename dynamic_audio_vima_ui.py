from true_trimodal_vima_backend import TrueTrimodalVIMABackend
from temporal_audio_memory import TemporalAudioMemory
import random
import numpy as np
import torch
import streamlit as st

from vima_bench import make
from vima_bench.tasks import ALL_TASKS
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

from train_harder_embodied_audio_sequence_policy import (
    EmbodiedAudioPolicy,
    encode_sample,
)

CKPT_PATH = "harder_embodied_audio_sequence_policy.pt"

TASKS = [
    "constraint_satisfaction/sweep_without_exceeding",
    "novel_concept_grounding/twist",
    "one_shot_imitation/follow_motion",
    "one_shot_imitation/follow_order",
    "require_memory/pick_in_order_then_restore",
]

st.set_page_config(page_title="Dynamic Audio-VIMA", layout="wide")
st.title("Audio-VIMA Dynamic Trimodal Reasoning System")

st.sidebar.title("Audio-VIMA Control Panel")

selected_task = st.sidebar.selectbox(
    "Select Task",
    TASKS,
)

execution_mode = st.sidebar.radio(
    "Execution Mode",
    [
        "Temporal Audio Memory",
        "Embodied Prompting",
        "Constraint Reasoning",
        "Memory Reasoning",
    ],
)

if st.sidebar.button("Reset Live Environment"):
    st.session_state.reset_requested = True

if st.sidebar.button("Generate Hearing Sequence"):
    st.session_state.generate_requested = True

st.sidebar.markdown("---")

st.sidebar.markdown("### Current System")
st.sidebar.write(f"Mode: {execution_mode}")

if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.wrapper = None
    st.session_state.obs = None
    st.session_state.candidates = []
    st.session_state.heard_sequence = []
    st.session_state.last_result = None
    st.session_state.true_vima_backend = None
    st.session_state.audio_memory = TemporalAudioMemory()

if st.session_state.heard_sequence:
    st.sidebar.success(
        f"Hearing memory active ({len(st.session_state.heard_sequence)} events)"
    )
else:
    st.sidebar.warning("No hearing memory yet")

@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = EmbodiedAudioPolicy(n_roles=len(ckpt["roles"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

model, ckpt = load_model()


def centroid_from_asset(asset):
    segm = asset.get("segm", None)

    if not isinstance(segm, dict):
        return np.array([0.5, 0.5], dtype=np.float32)

    mask = segm.get("top", None)

    if mask is None:
        return np.array([0.5, 0.5], dtype=np.float32)

    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = mask.squeeze()

    obj_pixels = mask != 255
    ys, xs = np.where(obj_pixels)

    if len(xs) == 0 or len(ys) == 0:
        return np.array([0.5, 0.5], dtype=np.float32)

    return np.array(
        [float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])],
        dtype=np.float32,
    )


def make_centroid_action(env, centroid):
    action = env.action_space.sample()

    x = float(np.clip(centroid[0], 0.25, 0.75))
    y = float(np.clip(centroid[1], -0.5, 0.5))

    action["pose0_position"] = np.array([x, y], dtype=np.float32)
    action["pose1_position"] = np.array([x, float(np.clip(y - 0.15, -0.5, 0.5))], dtype=np.float32)

    return action


def reset_live_env(task):
    env = make(task, modalities=["rgb", "segm"])

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

    candidates = []

    for role, asset in wrapper.env.prompt_assets.items():
        audio_token = np.asarray(asset.get("audio_token", np.zeros(768)), dtype=np.float32)

        if audio_token.shape[0] != 768:
            continue

        candidates.append({
            "role": role,
            "audio_token": audio_token,
            "centroid": centroid_from_asset(asset),
            "audio_id": asset.get("audio_id", "unknown"),
            "wav_path": asset.get("wav_path", "unknown"),
        })

    return env, wrapper, obs, candidates


tab_live, tab_memory, tab_results = st.tabs([
    "Live Trimodal Prompting",
    "Temporal Audio Memory",
    "Results",
])

st.markdown("---")
st.subheader("Live Multimodal Workspace")

scene_col, memory_col = st.columns([1.25, 1])

with scene_col:
        st.markdown("### Live Environment")

        st.caption(f"Selected task: `{selected_task}`")

        if st.button("Reset live scene", key="reset_live_scene_main"):
            try:
                env, wrapper, obs, candidates = reset_live_env(selected_task)

                st.session_state.env = env
                st.session_state.wrapper = wrapper
                st.session_state.obs = obs
                st.session_state.raw_obs = obs
                st.session_state.candidates = candidates
                st.session_state.heard_sequence = []
                st.session_state.last_result = None

                st.success(f"Scene reset: {selected_task}")
            except Exception as e:
                st.error(f"Reset failed: {e}")

        obs = st.session_state.obs

        if obs is not None:
            rgb = obs.get("rgb", {})

            cam_col1, cam_col2 = st.columns(2)

            with cam_col1:
                if "top" in rgb:
                    top = rgb["top"]
                    if top.shape[0] == 3:
                        top = np.transpose(top, (1, 2, 0))
                    st.image(top, caption="Top camera", use_container_width=True)

            with cam_col2:
                if "front" in rgb:
                    front = rgb["front"]
                    if front.shape[0] == 3:
                        front = np.transpose(front, (1, 2, 0))
                    st.image(front, caption="Front camera", use_container_width=True)
        else:
            st.info("Reset a live scene to show robot cameras.")

        st.markdown("### Candidate Objects")

        if st.session_state.candidates:
            st.dataframe(
                [
                    {
                        "role": c["role"],
                        "audio_id": c["audio_id"],
                        "wav_path": c["wav_path"],
                        "centroid": c["centroid"].tolist(),
                    }
                    for c in st.session_state.candidates
                ],
                use_container_width=True,
            )
        else:
            st.info("No candidate objects yet.")

with memory_col:
    st.markdown("### Temporal Hearing Timeline")

    if st.button("Generate heard sequence", key="generate_heard_sequence_main"):
        candidates = st.session_state.candidates

        if len(candidates) < 3:
            st.warning("Need at least 3 candidate objects for first/second/third reasoning.")
        else:
            random.shuffle(candidates)
            st.session_state.heard_sequence = candidates[:3]

            st.session_state.audio_memory.clear()

            for ev in st.session_state.heard_sequence:
                st.session_state.audio_memory.remember(
                    role=ev["role"],
                    audio_id=ev["audio_id"],
                    wav_path=ev["wav_path"],
                    audio_token=ev["audio_token"],
                    centroid=ev["centroid"],
                )

            st.success("Generated 3-event heard sequence and stored it in temporal audio memory.")

    heard_sequence = st.session_state.heard_sequence

    if heard_sequence:
        order_labels = ["First", "Second", "Third"]

        for i, ev in enumerate(heard_sequence):
            st.markdown(
                f"""
                <div style="
                    padding: 14px;
                    margin-bottom: 10px;
                    border-radius: 14px;
                    border: 1px solid #ddd;
                    background: #f8f9fb;
                ">
                    <b>{order_labels[i]} Sound</b><br>
                    Role: <code>{ev["role"]}</code><br>
                    Audio ID: <code>{ev["audio_id"]}</code><br>
                    WAV: <code>{ev["wav_path"]}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
   
             

    

    st.markdown("### Prompt Console")

    st.caption("Ask the robot using temporal audio memory.")

    user_prompt_live = st.text_input(
        "Prompt",
        value="Pick the object that sounded second.",
        key="live_prompt_box",
    )

    if st.button("Predict and execute", key="predict_execute_live"):
        st.session_state.pending_prompt = user_prompt_live
        st.session_state.run_prediction_requested = True

    if st.button("Run True Trimodal VIMA Backend", key="run_true_vima_backend"):
        if st.session_state.env is None or st.session_state.obs is None:
            st.error("Reset or generate a live scene first.")
        else:
            try:
                if st.session_state.true_vima_backend is None:
                    st.session_state.true_vima_backend = TrueTrimodalVIMABackend()

                true_result = st.session_state.true_vima_backend.predict(
                    prompt=st.session_state.env.prompt,
                    prompt_assets=st.session_state.env.prompt_assets,
                    obs=st.session_state.raw_obs,
                    meta=st.session_state.env.meta_info,
                )
                   
                st.session_state.last_result = true_result
                st.success("True Trimodal VIMA backend executed.")
                st.json(true_result)

            except Exception as e:
                st.error(f"True Trimodal VIMA backend failed: {e}")

        if st.session_state.last_result:
            result = st.session_state.last_result

            if result.get("backend") == "true_trimodal_vima":
                st.markdown("### Latest True Trimodal VIMA Result")
                st.write("Backend:", result.get("backend"))
                st.write("Prompt:", result.get("prompt"))
                st.write("Prompt tokens shape:", result.get("prompt_tokens_shape"))
                st.write("Observation tokens shape:", result.get("obs_token_shape"))
                st.write("Predicted action vector:", result.get("pred_action_vec"))
        else:
            status = "SUCCESS" if result.get("selection_success") == 1 else "FAILED"

            st.markdown("### Latest Result")
            st.metric("Selection", status)
            st.write("Target role:", result.get("target_role"))
            st.write("Predicted role:", result.get("pred_role"))
            st.write("Reward:", result.get("reward"))
            st.write("Audio events after action:", result.get("audio_events"))

