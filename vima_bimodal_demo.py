import os
import time
from typing import Any, Dict, Optional

import numpy as np
import imageio.v2 as imageio

from vima_bench import make
from vima_bench.env.wrappers.audio_wrapper import AudioWrapper

TASK = "constraint_satisfaction/sweep_without_touching"
# Try others:
# TASK = "constraint_satisfaction/sweep_without_exceeding"
# TASK = "instruction_following/rotate"

EPISODES = 2
STEPS_PER_EP = 60
SAVE_DIR = "bimodal_demo_frames"
DISPLAY_DEBUG_WINDOW = True   # set False if you only want saved RGB frames
SLEEP = 0.05                  # slow down stepping so you can watch

os.makedirs(SAVE_DIR, exist_ok=True)


def find_rgb_in_obs(obs: Any) -> Optional[np.ndarray]:
    """
    VIMA observations are usually dict-like. We try to locate an RGB image array.
    We look for common keys and also fall back to scanning arrays with shape (H,W,3) or (3,H,W).
    """
    if not isinstance(obs, dict):
        return None

    # Preferred key names (vary by repo versions)
    preferred = [
        "rgb", "image", "images", "obs", "observation",
        "rgb_front", "rgb_top", "rgb_static", "rgb_cam",
        "front_rgb", "top_rgb",
    ]
    for k in preferred:
        if k in obs:
            arr = obs[k]
            rgb = normalize_to_hwc_rgb(arr)
            if rgb is not None:
                return rgb

    # Fallback: scan all dict values
    for k, v in obs.items():
        rgb = normalize_to_hwc_rgb(v)
        if rgb is not None:
            return rgb

    return None


def normalize_to_hwc_rgb(x: Any) -> Optional[np.ndarray]:
    """Return uint8 RGB image in (H,W,3) if x looks like an image, else None."""
    if not isinstance(x, np.ndarray):
        return None
    if x.ndim != 3:
        return None

    # HWC
    if x.shape[-1] == 3 and x.shape[0] >= 32 and x.shape[1] >= 32:
        img = x
    # CHW
    elif x.shape[0] == 3 and x.shape[1] >= 32 and x.shape[2] >= 32:
        img = np.transpose(x, (1, 2, 0))
    else:
        return None

    # Convert float [0,1] -> uint8, or clip uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0) if np.issubdtype(img.dtype, np.floating) else img
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def summarize_obs(obs: Any) -> str:
    if not isinstance(obs, dict):
        return f"obs_type={type(obs)}"

    parts = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            parts.append(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            parts.append(f"{k}: type={type(v).__name__}")
    return " | ".join(parts)


def main():
    env = make(task_name=TASK, display_debug_window=DISPLAY_DEBUG_WINDOW)

    #ENABLE AUDIO HERE
    env = AudioWrapper(env, verbose=True)

    log_path = os.path.join(SAVE_DIR, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"TASK={TASK}\n")
        f.write(f"DISPLAY_DEBUG_WINDOW={DISPLAY_DEBUG_WINDOW}\n\n")

    for ep in range(EPISODES):
        obs = env.reset()
        prompt = getattr(env, "prompt", "<no env.prompt attribute>")
        obs_summary = summarize_obs(obs)

        print(f"\n=== EPISODE {ep} ===")
        print("PROMPT (language):", prompt)
        print("OBS SUMMARY:", obs_summary)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"EPISODE {ep}\n")
            f.write(f"PROMPT: {prompt}\n")
            f.write(f"OBS: {obs_summary}\n")

        # Save an initial RGB frame (vision)
        rgb0 = find_rgb_in_obs(obs)
        if rgb0 is not None:
            p0 = os.path.join(SAVE_DIR, f"ep{ep:02d}_step000_rgb.png")
            imageio.imwrite(p0, rgb0)
            print("Saved initial RGB:", p0)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Saved: {p0}\n")
        else:
            print("⚠️ No RGB image found in obs dict. (This depends on your VIMA version/wrappers.)")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("No RGB found in obs.\n")

        # Step loop
        for step in range(1, STEPS_PER_EP + 1):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            rgb = find_rgb_in_obs(obs)
            if rgb is not None and (step % 5 == 0):  # save every 5 steps to reduce disk spam
                path = os.path.join(SAVE_DIR, f"ep{ep:02d}_step{step:03d}_rgb.png")
                imageio.imwrite(path, rgb)

            line = f"ep={ep} step={step} reward={reward} done={done}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            time.sleep(SLEEP)

            if done:
                print("Episode ended early (normal with random actions).")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("Episode ended early.\n\n")
                break

    env.close()
    print("\n✅ Done.")
    print(f"Frames saved to: {SAVE_DIR}/")
    print(f"Log saved to: {SAVE_DIR}/run_log.txt")


if __name__ == "__main__":
    main()
