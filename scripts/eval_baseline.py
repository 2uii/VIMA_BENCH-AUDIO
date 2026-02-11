# scripts/eval_baseline.py
# ------------------------------------------------------------
# Baseline evaluator for VIMA (Vision + Language only)
# Runs N episodes with fixed seeds and writes CSV to results/
# ------------------------------------------------------------
import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
from vima_bench import make

TASK = "constraint_satisfaction/sweep_without_exceeding"

N_EPISODES = 100
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
    set_global_seed(seed)
    env.seed(seed)

    obs = env.reset()
    task = env.task
    oracle_fn = task.oracle(env)

    success = False
    failure = False

    for t in range(1, min(MAX_STEPS, task.oracle_max_steps) + 1):
        oracle_action = oracle_fn.act(obs)
        if oracle_action is None:
            # Oracle sometimes fails (e.g., target not visible). Treat as failure and stop episode.
            failure = True
            break

        oracle_action = {
            k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
            for k, v in oracle_action.items()
        }
        obs, reward, done, info = env.step(action=oracle_action, skip_oracle=False)

        success = bool(info.get("success", False))
        failure = bool(info.get("failure", False))

        if done or success or failure:
            break

    return success, t, failure
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default=TASK)
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--seed", type=int, default=BASE_SEED)
    return p.parse_args()


def main():
    args = parse_args()
    global TASK, N_EPISODES, BASE_SEED, SEEDS, OUT_PATH
    TASK = args.task
    N_EPISODES = args.episodes
    BASE_SEED = args.seed
    SEEDS = [BASE_SEED + i for i in range(N_EPISODES)]
    OUT_PATH = os.path.join(
        OUT_DIR, f"baseline_{TASK.replace('/','_')}_{N_EPISODES}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print(f"RUNNING BASELINE VIMA")
    print(f"TASK: {TASK}")
    print(f"EPISODES: {N_EPISODES}")
    print("=" * 60 + "\n")


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
