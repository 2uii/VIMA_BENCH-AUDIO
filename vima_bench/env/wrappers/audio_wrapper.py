import time
import subprocess
from pathlib import Path
import pybullet as p
import gym


class AudioWrapper(gym.Wrapper):
    """
    Plays WAV sounds when contacts happen with 3 stationary objects.
    - We auto-detect 3 static bodies (mass==0) after reset.
    - When any contact involves one of those bodies, play its assigned WAV.
    """

    def __init__(self, env, wav_paths=None, cooldown_sec=0.20, verbose=True):
        super().__init__(env)
        self.cooldown_sec = cooldown_sec
        self.verbose = verbose

        # WAV files (must exist)
        if wav_paths is None:
            wav_paths = ["sounds/obj1.wav", "sounds/obj2.wav", "sounds/obj3.wav"]
        self.wav_paths = [str(Path(w)) for w in wav_paths]

        self.static_ids = []            # 3 chosen stationary body ids
        self.sound_by_id = {}           # body_id -> wav_path
        self._last_play = {}            # body_id -> time

    def _play_wav(self, wav_path):
        try:
            # Convert WSL path to Windows path so PowerShell can read it
            win_path = subprocess.check_output(["wslpath", "-w", str(wav_path)]).decode().strip()
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    f"(New-Object Media.SoundPlayer '{win_path}').Play();"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
           )
        except Exception:
            pass

    def _choose_three_stationary_bodies(self):
        # Find static bodies (base mass == 0). Exclude the ground plane if possible.
        static = []
        for bid in range(p.getNumBodies()):
            mass = p.getDynamicsInfo(bid, -1)[0]
            if mass == 0:
                name = p.getBodyInfo(bid)[1].decode("utf-8", errors="ignore")
                static.append((bid, name))

        # Heuristic exclusions: plane/ground
        filtered = []
        for bid, name in static:
            lname = name.lower()
            if "plane" in lname or "ground" in lname:
                continue
            filtered.append((bid, name))

        # If too few, fall back to all static
        candidates = filtered if len(filtered) >= 3 else static

        # Keep first 3 unique
        chosen = candidates[:3]
        self.static_ids = [bid for bid, _ in chosen]

        # Map each id -> sound
        self.sound_by_id = {}
        for i, bid in enumerate(self.static_ids):
            self.sound_by_id[bid] = self.wav_paths[i % len(self.wav_paths)]
            self._last_play[bid] = 0.0

        if self.verbose:
            print("\n[AUDIO] Selected 3 stationary bodies:")
            for bid, name in chosen:
                print(f"  - id={bid} name={name} -> {self.sound_by_id[bid]}")

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # Every reset, re-detect static objects (IDs can change between resets)
        self._choose_three_stationary_bodies()
        return out

    def step(self, action):
        out = self.env.step(action)

        # Support old/new gym API
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = out

        # Play sound if contact involves any of the 3 stationary ids
        try:
            now = time.time()
            for cp in cps:
                bodyA = cp[1]
                bodyB = cp[2]

                # If either body is one of our stationary objects
                hit_id = None
                if bodyA in self.sound_by_id:
                    hit_id = bodyA
                elif bodyB in self.sound_by_id:
                    hit_id = bodyB

                if hit_id is not None:
                    if now - self._last_play[hit_id] >= self.cooldown_sec:
                        self._last_play[hit_id] = now
                        if self.verbose:
                           print(f"[AUDIO] Contact -> id={hit_id} wav={self.sound_by_id[hit_id]}") 
                        self._play_wav(self.sound_by_id[hit_id])
        except Exception:
            pass

        if len(out) == 4:
            return obs, reward, terminated, info
        return obs, reward, terminated, truncated, info
