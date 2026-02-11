# scripts/eval_audio_vima.py
# ------------------------------------------------------------
# Audio-VIMA evaluator (Vision + Language + Audio Identity)
# Runs N episodes with fixed seeds and writes CSV to results/
# Also computes "compliance" based on silent-touch termination.
# ------------------------------------------------------------
import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import pybullet as p
from vima_bench import make
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

TASK = "constraint_satisfaction/sweep_without_exceeding"

N_EPISODES = 100
MAX_STEPS = 250

BASE_SEED = 123
SEEDS = [BASE_SEED + i for i in range(N_EPISODES)]

OUT_DIR = "results"
OUT_PATH = os.path.join(
    OUT_DIR, f"audiovima_{TASK.replace('/','_')}_{N_EPISODES}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

SOUND_LABELS = ["tingting", "thud", "alarm"]  # cycle through these
SILENT_FRACTION = 0.33  # fraction of objects that will be silent (None)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_body_name(bid: int) -> str:
    try:
        return (p.getBodyInfo(bid)[1] or b"").decode("utf-8", "ignore")
    except Exception:
        return ""

def detect_robot_body() -> int:
    """Heuristic: robot usually has the most joints."""
    nb = p.getNumBodies()
    best, best_j = 0, -1
    for bid in range(nb):
        try:
            nj = p.getNumJoints(bid)
            if nj > best_j:
                best, best_j = bid, nj
        except Exception:
            pass
    return best

def build_object_sound_map(ignore_bodies: set, tool_bodies: set):
    """
    Build a mapping of dynamic object bodies -> sound label or None (silent).
    We treat: mass>0 bodies as candidate objects, excluding known ignore/tool bodies.
    """
    candidates = []
    for bid in range(p.getNumBodies()):
        if bid in ignore_bodies or bid in tool_bodies:
            continue
        try:
            mass = p.getDynamicsInfo(bid, -1)[0]
        except Exception:
            continue
        if mass and mass > 0:
            candidates.append(bid)

    candidates.sort()
    obj_map = {}

    # assign some objects silent to enable compliance measurement
    n = len(candidates)
    silent_n = int(round(n * SILENT_FRACTION))
    silent_set = set(candidates[:silent_n])  # deterministic: first slice silent

    si = 0
    for bid in candidates:
        if bid in silent_set:
            obj_map[bid] = None
        else:
            obj_map[bid] = SOUND_LABELS[si % len(SOUND_LABELS)]
            si += 1

    return obj_map, candidates

def run_episode(seed: int):
    set_global_seed(seed)

    base_env = make(task_name=TASK, display_debug_window=False)
    base_env.seed(seed)
    obs = base_env.reset()

    task = base_env.task
    oracle_fn = task.oracle(base_env)

    # Identify bodies to ignore and tool bodies
    ignore_bodies = set()
    tool_bodies = set()

    for bid in range(p.getNumBodies()):
        name = get_body_name(bid).lower()
        if any(k in name for k in ["plane", "ground", "table", "workspace"]):
            ignore_bodies.add(bid)

    robot_id = detect_robot_body()
    ignore_bodies.add(robot_id)
    tool_bodies.add(robot_id)

    object_sound_map, dynamic_objects = build_object_sound_map(
        ignore_bodies, tool_bodies
    )

    env = AudioIdentityWrapper(
        base_env,
        object_sound_map=object_sound_map,
        debug=False,
        ignore_bodies=ignore_bodies,
        tool_bodies=tool_bodies,
        terminate_on_silent_touch=True,
        silent_penalty=-1.0,
    )

    obs = env.reset()

    success = False
    failure = False
    terminated_reason = ""
    audio_event_count = 0

    for t in range(1, min(MAX_STEPS, task.oracle_max_steps) + 1):
        oracle_action = oracle_fn.act(obs)
        if oracle_action is None:
            # Oracle sometimes fails (e.g., target not visible). Don’t crash the run.
            failure = True
            terminated_reason = "oracle_returned_none"
            break


        oracle_action = {
            k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
            for k, v in oracle_action.items()
        }

        obs, reward, done, info = env.step(oracle_action)

        success = bool(info.get("success", False))
        failure = bool(info.get("failure", False))
        terminated_reason = str(info.get("terminated_reason", ""))

        audio_events = info.get("audio_events", [])
        audio_event_count = len(audio_events) if isinstance(audio_events, list) else 0

        if done or success or failure:
            break

    env.close()

    return {
        "seed": seed,
        "success": int(success),
        "failure": int(failure),
        "steps": t,
        "terminated_reason": terminated_reason,
        "compliant": int(terminated_reason != "touched_silent_object_with_tool"),
        "audio_events": audio_event_count,
        "num_dynamic_objects": len(dynamic_objects),
        "num_silent_objects": sum(1 for v in object_sound_map.values() if v is None),
    }

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
        OUT_DIR, f"audiovima_{TASK.replace('/','_')}_{N_EPISODES}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    success_count = 0
    compliant_count = 0

    for i, seed in enumerate(SEEDS, start=1):
        row = run_episode(seed)
        rows.append({"episode": i, **row})

        success_count += row["success"]
        compliant_count += row["compliant"]

        print(
            f"[Audio-VIMA] ep={i}/{N_EPISODES} seed={seed} "
            f"success={bool(row['success'])} compliant={bool(row['compliant'])} "
            f"reason={row['terminated_reason'] or '-'} steps={row['steps']}"
        )

    success_rate = 100.0 * success_count / N_EPISODES
    compliance_rate = 100.0 * compliant_count / N_EPISODES
    avg_audio_events = float(np.mean([r["audio_events"] for r in rows])) if rows else 0.0

    print("\n=== AUDIO-VIMA SUMMARY ===")
    print(f"Task: {TASK}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Successes: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Compliance Rate: {compliance_rate:.2f}%")
    print(f"Avg Audio Events/Episode: {avg_audio_events:.2f}")
    print(f"CSV: {OUT_PATH}")


    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
