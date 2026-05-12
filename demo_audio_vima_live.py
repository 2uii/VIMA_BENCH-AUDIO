import random
import numpy as np
import torch

from vima_bench import make
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper
from true_trimodal_vima_backend import TrueTrimodalVIMABackend


TASK = "constraint_satisfaction/sweep_without_exceeding"


def main():
    print("\n=== AUDIO-VIMA LIVE DEMO ===\n")

    env = make(TASK, modalities=["rgb", "segm"])
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

    print("Task:", TASK)
    print("Prompt:", wrapper.env.prompt)

    print("\n=== PROMPT ASSET AUDIO IDENTITIES ===")
    candidates = []

    for role, asset in wrapper.env.prompt_assets.items():
        audio_id = asset.get("audio_id", "unknown")
        wav_path = asset.get("wav_path", "unknown")
        audio_token = np.asarray(asset.get("audio_token", np.zeros(768)), dtype=np.float32)

        print(f"Role: {role}")
        print(f"  audio_id: {audio_id}")
        print(f"  wav_path : {wav_path}")
        print(f"  token shape: {audio_token.shape}")

        candidates.append({
            "role": role,
            "audio_id": audio_id,
            "wav_path": wav_path,
            "audio_token": audio_token,
        })

    print("\n=== SIMULATED HEARING SEQUENCE ===")
    random.shuffle(candidates)
    heard_sequence = candidates[:3]

    order_names = ["FIRST", "SECOND", "THIRD"]
    for order, ev in zip(order_names, heard_sequence):
        print(f"{order} SOUND -> role={ev['role']} | audio_id={ev['audio_id']} | wav={ev['wav_path']}")

    user_prompt = "Pick the object that sounded second."
    target_event = heard_sequence[1]

    print("\nUser prompt:", user_prompt)
    print("Resolved target role:", target_event["role"])

    print("\n=== RUNNING TRUE TRIMODAL VIMA BACKEND ===")
    backend = TrueTrimodalVIMABackend()

    result = backend.predict(
        prompt=wrapper.env.prompt,
        prompt_assets=wrapper.env.prompt_assets,
        obs=obs,
        meta=wrapper.env.meta_info,
    )

    print("\n=== BACKEND RESULT ===")
    print("Backend:", result["backend"])
    print("Prompt token type:", result["prompt_token_type"])
    print("Prompt tokens shape:", result["prompt_tokens_shape"])
    print("Observation tokens shape:", result["obs_token_shape"])
    print("Predicted action vector:", result["pred_action_vec"])

    print("\n=== DEMO COMPLETE ===")


if __name__ == "__main__":
    main()
