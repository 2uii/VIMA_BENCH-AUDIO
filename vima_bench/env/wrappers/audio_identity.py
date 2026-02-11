import pybullet as p
import time
import subprocess
from pathlib import Path
import numpy as np

class AudioIdentityWrapper:
    """
    AudioIdentityWrapper (meaningful)
    --------------------------------
    Audio defines object identity.

    Improvements:
    - Ignore floor/table/ur5 contacts (prevents spamming)
    - Only trigger when tagged object contacts:
        • the tool (spatula/gripper base), OR
        • another object (dynamic contact)
    - Optionally terminate if a SILENT object is touched by the tool.
    """


    def __init__(
        self,
        env,
        object_sound_map,
        cooldown=0.25,
        debug=False,
        ignore_bodies=None,
        tool_bodies=None,
        terminate_on_silent_touch=False,
        silent_penalty=-1.0,
    ):
        self.env = env
        self.object_sound_map = object_sound_map
        self.cooldown = cooldown
        self.debug = debug

        self.ignore_bodies = set(ignore_bodies or [])
        self.tool_bodies = set(tool_bodies or [])

        self.terminate_on_silent_touch = terminate_on_silent_touch
        self.silent_penalty = float(silent_penalty)

        self.last_play_time = {}
        self.audio_events = []
        self.step_count = 0

        # audio embedding state (persistent per episode)
        self.emb_dim = 128
        self.rng = np.random.default_rng(42)
        self.audio_obj_emb = {}

        base_dir = Path(__file__).resolve().parents[3]
        sound_dir = base_dir / "sounds"

        self.sound_bank = {
            "tingting": str(sound_dir / "obj1.wav"),
            "thud": str(sound_dir / "obj2.wav"),
            "alarm": str(sound_dir / "obj3.wav"),
            "impact": str(sound_dir / "hunk_hunk.wav"),
        }

    def reset(self):
        self.audio_events.clear()
        self.last_play_time.clear()
        self.step_count = 0

        # create persistent embeddings for this episode
        self.audio_obj_emb = {}
        for obj_id, label in self.object_sound_map.items():
            if label is None:
                continue
            self.audio_obj_emb[obj_id] = self.rng.normal(
                size=(self.emb_dim,)
            ).astype("float32")

        return self.env.reset()

    def _play(self, wav_path):
        try:
            subprocess.Popen(
                ["play", "-q", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1

        now = time.time()
        cps = p.getContactPoints()

        # optional debug
        if self.debug and self.step_count % 30 == 0 and cps:
            pairs = set((cp[1], cp[2]) for cp in cps)
            print(f"[DEBUG] step={self.step_count} contact pairs(sample)={list(pairs)[:10]}")

        # track if we touched silent object with the tool
        silent_violation = False

        for cp in cps:
            body_a = cp[1]
            body_b = cp[2]

            # ignore plane/workspace/ur5/lines contacts entirely
            if body_a in self.ignore_bodies or body_b in self.ignore_bodies:
                continue

            # determine if one side is tagged object
            for hit_id, other_id in ((body_a, body_b), (body_b, body_a)):
                if hit_id not in self.object_sound_map:
                    continue

                # We only care if interaction is with tool OR another object (not ignored)
                tool_touch = (other_id in self.tool_bodies)
                object_touch = (other_id not in self.ignore_bodies)

                # if you want only tool contacts, comment out object_touch line and require tool_touch
                if not (tool_touch or object_touch):
                    continue

                sound_label = self.object_sound_map[hit_id]

                # If silent object touched by tool -> violation
                if sound_label is None and tool_touch:
                    silent_violation = True
                    continue

                if sound_label is None:
                    continue  # silent objects make no sound

                last = self.last_play_time.get(hit_id, 0.0)
                if now - last < self.cooldown:
                    continue

                wav = self.sound_bank.get(sound_label)
                if wav:
                    self._play(wav)

                self.audio_events.append(
                    {
                        "step": self.step_count,
                        "object_id": hit_id,
                        "sound": sound_label,
                        "other_body": other_id,
                        "tool_touch": tool_touch,
                    }
                )
                self.last_play_time[hit_id] = now

        # enforce rule: touching silent objects ends episode
        if self.terminate_on_silent_touch and silent_violation and not done:
            done = True
            reward = self.silent_penalty
            info = dict(info)
            info["terminated_reason"] = "touched_silent_object_with_tool"

        # ------------------------------------------------
        # expose audio information to agent (EVERY step)
        # ------------------------------------------------
        info = dict(info) if info is not None else {}
        info["audio_events"] = list(self.audio_events)
        info["audio_identity"] = dict(self.object_sound_map)
        info["audio_obj_emb"] = {
            k: v.tolist() for k, v in self.audio_obj_emb.items()
        }
        info["audio_emb_dim"] = self.emb_dim

        return obs, reward, done, info


    def close(self):
        self.env.close()

    @property
    def prompt(self):
        return self.env.prompt

    @property
    def action_space(self):
        return self.env.action_space
