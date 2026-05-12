import json
import os
import threading

import numpy as np
from PIL import Image

from vima_bench import VIMAEnvBase
from backend.state import state


def run_preview():
    try:
        state["status"] = "running"

        # load prompt
        with open("prompt.json", "r") as f:
            prompt = json.load(f)

        task = prompt.get("task", "sweep_without_exceeding")
        state["task"] = task
        state["prompt"] = prompt

        print("[RUNNER] Starting task:", task)

        env = VIMAEnvBase(
            task=task,
            modalities=["rgb", "segm"],
            display_debug_window=False,
            hide_arm_rgb=False,
        )

        obs = env.reset()

        # get frame
        if "front" in obs["rgb"]:
            frame = obs["rgb"]["front"]
        else:
            view = list(obs["rgb"].keys())[0]
            frame = obs["rgb"][view]
            print("[RUNNER] Using fallback view:", view)

        frame = np.transpose(frame, (1, 2, 0)).astype(np.uint8)

        os.makedirs("backend", exist_ok=True)
        out_path = state["frame_path"]

        Image.fromarray(frame).save(out_path)

        print("[RUNNER] Saved frame:", out_path)

        env.close()

        state["status"] = "finished"

    except Exception as e:
        print("[RUNNER ERROR]", e)
        state["status"] = "error"


def start_background_run():
    thread = threading.Thread(target=run_preview)
    thread.start()
