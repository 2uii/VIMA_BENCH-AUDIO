from vima_bench import make
import pybullet as p
import time

TASK = "constraint_satisfaction/sweep_without_touching"
# You can swap to:
# TASK = "constraint_satisfaction/sweep_without_exceeding"

STEPS = 250          # how many steps to run
SLEEP = 0.05         # speed (smaller = faster)
CAM_DIST = 0.45      # camera distance from gripper marker
CAM_PITCH = -25      # camera pitch angle

def pick_robot_body():
    """Pick the body most likely to be the robot: max joints, fallback max visuals."""
    nb = p.getNumBodies()
    joint_counts = [(bid, p.getNumJoints(bid)) for bid in range(nb)]
    joint_counts.sort(key=lambda x: x[1], reverse=True)
    best_bid, best_j = joint_counts[0]
    return best_bid

def pick_ee_link(robot_id: int):
    """Pick an end-effector link index using name heuristics; fallback to last joint."""
    nJ = p.getNumJoints(robot_id)
    if nJ <= 0:
        return None

    keywords = ["gripper", "finger", "ee", "tool", "tcp", "tip", "hand"]
    candidates = []
    for j in range(nJ):
        info = p.getJointInfo(robot_id, j)
        jname = (info[1] or b"").decode("utf-8", "ignore").lower()
        lname = (info[12] or b"").decode("utf-8", "ignore").lower()
        score = 0
        for k in keywords:
            if k in jname: score += 2
            if k in lname: score += 2
        # Prefer later links slightly
        score += j / max(1, nJ-1)
        candidates.append((score, j, jname, lname))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0]
    # If no keyword hit at all, use last joint
    if best[0] < 1.0:
        return nJ - 1
    return best[1]

def make_marker(radius=0.025, rgba=(0, 1, 0, 1)):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=list(rgba))
    marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0])
    return marker

def main():
    env = make(task_name=TASK, display_debug_window=True)
    env.reset()
    print("Task:", TASK)
    print("Prompt:", env.prompt)

    # Turn off preview panes so it doesn't look like a "picture"
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    # Hide side GUI for more viewport space (optional)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    robot_id = pick_robot_body()
    nJ = p.getNumJoints(robot_id)
    print(f"Picked robot body_id={robot_id}, joints={nJ}")

    ee = pick_ee_link(robot_id)
    print("Picked end-effector link index:", ee)

    marker = make_marker(radius=0.03, rgba=(0, 1, 0, 1))

    # Main loop: step env, move marker to EE, auto-orbit camera
    yaw = 30.0
    for i in range(STEPS):
        # Step with random action to force motion attempts
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)

        # Update marker position to EE
        if ee is not None and ee >= 0 and ee < p.getNumJoints(robot_id):
            ee_state = p.getLinkState(robot_id, ee)
            if ee_state is not None:
                ee_pos = ee_state[0]
                p.resetBasePositionAndOrientation(marker, ee_pos, [0, 0, 0, 1])

                # Auto camera orbit around EE marker so you can ALWAYS see it
                yaw += 0.8
                p.resetDebugVisualizerCamera(
                    cameraDistance=CAM_DIST,
                    cameraYaw=yaw,
                    cameraPitch=CAM_PITCH,
                    cameraTargetPosition=ee_pos
                )

        if done:
            print(f"Episode ended at step {i} (normal for random actions). Resetting...")
            env.reset()

        time.sleep(SLEEP)

    print("Done. Closing.")
    env.close()

if __name__ == "__main__":
    main()
