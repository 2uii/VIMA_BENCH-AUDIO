import time
import numpy as np
import pybullet as p

from vima_bench import make
from vima_bench.env.wrappers.audio_identity import AudioIdentityWrapper

# -------------------------
# DEMO CONFIG
# -------------------------
TASK = "constraint_satisfaction/sweep_without_touching"
STEP_DELAY = 0.05          # slow, stable motion
TOTAL_STEPS = 2000

print("\n=== TRIMODAL VIMA DEMO ===")

# -------------------------
# Create environment WITH GUI
# -------------------------
env = make(
    task_name=TASK,
    display_debug_window=True
)

obs = env.reset()

# -------------------------
# Inspect bodies + auto-detect tool/ignore/objects
# -------------------------
print("\nBODY LIST:")
tool_ids = []
ignore_bodies = set()
object_ids = []

for i in range(p.getNumBodies()):
    mass = p.getDynamicsInfo(i, -1)[0]
    name = p.getBodyInfo(i)[1].decode("utf-8", errors="ignore")
    lname = name.lower()

    print(f" {i:2d} | mass={mass:.4f} | {name}")

    if mass == 0 or "line.urdf" in lname or "workspace" in lname or "ur5" in lname or "plane" in lname:
        ignore_bodies.add(i)

    if "spatula" in lname or "suction" in lname or "gripper" in lname:
        tool_ids.append(i)

    if mass > 0 and not ("spatula" in lname or "suction" in lname or "gripper" in lname):
        object_ids.append(i)

print("\nAUTO-DETECTED:")
print(" tool_ids:", tool_ids)
print(" ignore_bodies:", sorted(ignore_bodies))
print(" object_ids:", object_ids)

# -------------------------
# AUDIO DEFINES IDENTITY
# -------------------------
labels = ["tingting", "thud", "alarm"]
object_sound_map = {}
for idx, oid in enumerate(object_ids):
    object_sound_map[oid] = labels[idx % len(labels)]

# -------------------------
# ATTACH AUDIO WRAPPER  (CRITICAL)
# -------------------------
env = AudioIdentityWrapper(
    env,
    object_sound_map=object_sound_map,
    cooldown=0.25,
    debug=False,
    ignore_bodies=ignore_bodies,
    tool_bodies=set(tool_ids),
    terminate_on_silent_touch=False,
)

# -------------------------
# TRIMODAL PROMPT (do NOT assign; prompt may be read-only)
# -------------------------
trimodal_prompt = (
    "Sweep ONLY objects with sound token <AUD_TING>. "
    "Avoid objects with sound token <AUD_THUD>."
)

print("\n=== PROMPT ===")
print("TEXT:", trimodal_prompt)
print("AUDIO MAP:", object_sound_map)
print("VISION: PyBullet GUI with UR5 arm")

# -------------------------
# Inspect bodies (for transparency)
# -------------------------
# -------------------------
# Inspect bodies + auto-detect tool/ignore/objects
# -------------------------
print("\nBODY LIST:")
tool_ids = []
ignore_bodies = set()
object_ids = []

for i in range(p.getNumBodies()):
    mass = p.getDynamicsInfo(i, -1)[0]
    name = p.getBodyInfo(i)[1].decode("utf-8", errors="ignore")
    lname = name.lower()

    print(f" {i:2d} | mass={mass:.4f} | {name}")

    # ignore common static clutter + boundaries
    if mass == 0 or "line.urdf" in lname or "workspace" in lname or "ur5" in lname or "plane" in lname:
        ignore_bodies.add(i)

    # tool detection
    if "spatula" in lname or "suction" in lname or "gripper" in lname:
        tool_ids.append(i)

    # candidate objects: dynamic bodies that are not tools
    if mass > 0 and not ("spatula" in lname or "suction" in lname or "gripper" in lname):
        object_ids.append(i)

print("\nAUTO-DETECTED:")
print(" tool_ids:", tool_ids)
print(" ignore_bodies:", sorted(ignore_bodies))
print(" object_ids:", object_ids)

# -------------------------
# AUDIO DEFINES IDENTITY (auto assign)
# -------------------------
# Rule: first object(s) are targets, second is silent, rest targets (simple demo)
# -------------------------
# AUDIO DEFINES IDENTITY (2 objects)
# -------------------------
object_sound_map = {}
if len(object_ids) >= 1:
    object_sound_map[object_ids[0]] = "tingting"  # Object A
if len(object_ids) >= 2:
    object_sound_map[object_ids[1]] = "thud"      # Object B (different)
# if more objects exist in other tasks, you can extend:
for oid in object_ids[2:]:
    object_sound_map[oid] = "alarm"


# BASELINE (NO AUDIO WRAPPER)
# Keep env as-is


obs = env.reset()

# -------------------------
# TRIMODAL PROMPT
# -------------------------
target_token = "<AUD_TING>"
avoid_token  = "<AUD_THUD>"

# -------------------------
# PROMPT OVERRIDE (safe for wrapper or base env)
# -------------------------
prompt_text = (
    "Sweep ONLY objects with sound token <AUD_TING>. "
    "Avoid objects with sound token <AUD_THUD>."
)

if hasattr(env, "env"):          # wrapper case
    env.env.prompt = prompt_text
else:                            # baseline case
    env.prompt = prompt_text


print("\n=== PROMPT ===")
print("TEXT:", env.prompt)
print("AUDIO MAP:", object_sound_map)
print("VISION: PyBullet GUI with UR5 arm")

# -------------------------
# DEMO LOOP (slow + stable)
# -------------------------
print("\nStarting demo...")

action_space = env.action_space

for step in range(TOTAL_STEPS):
    if step < 50:
# valid action for VIMA: Dict with 2 XY poses + quaternions
        action = action_space.sample()

# OPTIONAL: make rotation stable (identity quaternion)
        action["pose0_rotation"] = np.array([0, 0, 0, 1], dtype=np.float32)
        action["pose1_rotation"] = np.array([0, 0, 0, 1], dtype=np.float32)
    else:
        action = action_space.sample()

    obs, r, d, info = env.step(action)
    # (keep your audio_vec block after this)

# -------------------------
# AUDIO -> OBS (vector)
# -------------------------
emb_dim = info.get("audio_emb_dim", 128)
audio_vec = np.zeros((emb_dim,), dtype=np.float32)

events = info.get("audio_events", [])
emb_map = info.get("audio_obj_emb", {})

# If there is at least one audio event, use the most recent event's object embedding
if len(events) > 0:
    last = events[-1]                  # most recent event
    obj_id = last["object_id"]
    if obj_id in emb_map:
        audio_vec = np.array(emb_map[obj_id], dtype=np.float32)

# attach audio vector into observation (new modality)
if isinstance(obs, dict):
    obs["audio_vec"] = audio_vec

# one-time print when audio starts happening (AFTER audio_vec is filled)
if (not hasattr(env, "_printed_first_audio")) and len(events) > 0:
    env._printed_first_audio = True
    print("FIRST AUDIO EVENT:", events[0])
    print("audio_vec sum after first event:", float(audio_vec.sum()))

    # quick sanity print once

    if step == 0:
        emb = info.get("audio_obj_emb", {})
        print("audio_emb_dim:", info.get("audio_emb_dim"))
        print("audio_obj_emb keys:", list(emb.keys()))
        if emb:
            k0 = list(emb.keys())[0]
            print("sample emb len:", len(emb[k0]), "first 5:", emb[k0][:5])
    if step == 0:
        print("obs has audio_vec:", isinstance(obs, dict) and "audio_vec" in obs)


    if step % 50 == 0:
        print("info keys:", list(info.keys()), "| audio_events:", len(info.get("audio_events", [])))

    time.sleep(STEP_DELAY)

print("\n=== DEMO FINISHED ===")
print("Audio events:", len(info.get("audio_events", [])))
env.close()

env.close()
