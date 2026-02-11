# scripts/eval_baseline.py
# ------------------------------------------------------------
# Baseline evaluator for VIMA (Vision + Language only)
# Runs N episodes with fixed seeds and writes CSV to results/
# ------------------------------------------------------------
import csv
import os
import random
from datetime import datetime

import numpy as np
from vima_bench import make

TASK = "constraint_satisfaction/sweep_without_touching"

N_EPISODES = 50
MAX_STEPS = 250

# Reproducible seed list
BASE_SEED = 123
SEEDS = [BASE_SEED + i for i in range(N_EPISODES)]

OUT_DIR = "results"
OUT_PATH = os.path.join(
    OUT_DIR, f"baseline_{TASK.replace('/','_')}_{N_EPISODES}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def run_episode(env, seed: int):
    """
    Runs one episode using random actions (baseline reference).
    Returns: (success: bool, steps: int, failure: bool)
    """
    set_global_seed(seed)
    env.reset()

    success = False
    failure = False

    for t in range(1, MAX_STEPS + 1):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        success = bool(info.get("success", False))
        failure = bool(info.get("failure", False))

        if done or success or failure:
            break

    return success, t, failure

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    env = make(task_name=TASK, display_debug_window=False)

    rows = []
    success_count = 0

    for i, seed in enumerate(SEEDS, start=1):
        success, steps, failure = run_episode(env, seed)
        success_count += int(success)

        rows.append({
            "episode": i,
            "seed": seed,
            "success": int(success),
            "failure": int(failure),
            "steps": steps,
        })

        print(f"[Baseline] ep={i}/{N_EPISODES} seed={seed} success={success} steps={steps}")

    env.close()

    success_rate = 100.0 * success_count / N_EPISODES
    print("\n=== BASELINE SUMMARY ===")
    print(f"Task: {TASK}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Successes: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"CSV: {OUT_PATH}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
