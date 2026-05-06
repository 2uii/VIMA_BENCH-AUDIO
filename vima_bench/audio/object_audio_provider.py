import os
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torchaudio


@dataclass
class AudioSpec:
    """Persistent audio identity spec attached to an object."""
    audio_id: str                 # e.g., "ring_phone"
    wav_path: str                 # path to wav file
    policy: str = "loop"          # "loop" | "periodic" | "event"
    gain: float = 1.0             # volume multiplier
    period_steps: int = 30        # used if policy == "periodic"


class ObjectAudioProvider:
    """
    Produces *object-aligned* audio clips.
    Key idea for training: do NOT start with mic mixing; return clean clips per object_id.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        clip_seconds: float = 1.0,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.clip_len = int(sample_rate * clip_seconds)
        self.device = device

        # Cache decoded waveforms to avoid reloading each step
        self._wav_cache: Dict[str, torch.Tensor] = {}
        self._sr_cache: Dict[str, int] = {}

    def _load_wav(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        wav_path = os.path.expanduser(wav_path)

        if wav_path in self._wav_cache:
            return self._wav_cache[wav_path], self._sr_cache[wav_path]

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        data, sr = sf.read(wav_path, always_2d=True)  # (T, C)
        wav = torch.from_numpy(data).float().T  # (C, T)
        wav = wav.mean(dim=0, keepdim=True)  # mono (1, T)

        self._wav_cache[wav_path] = wav
        self._sr_cache[wav_path] = sr
        return wav, sr

    def _resample_if_needed(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return wav
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
        return resampler(wav.float())

    def _clip_from_wave(self, wav_mono: torch.Tensor, start: int) -> torch.Tensor:
        """
        wav_mono: (1, T) at target sample_rate
        return: (Tclip,) float32
        """
        T = wav_mono.shape[-1]
        if T <= 0:
            return torch.zeros(self.clip_len, dtype=torch.float32)

        # wrap-around indexing for looping
        idx = (torch.arange(self.clip_len) + start) % T
        clip = wav_mono[0, idx].to(torch.float32)
        return clip

    def is_active(self, spec: AudioSpec, step: int, event_flag: Optional[bool] = None) -> bool:
        if spec.policy == "loop":
            return True
        if spec.policy == "periodic":
            return (step % max(1, spec.period_steps)) == 0
        if spec.policy == "event":
            return bool(event_flag)
        return False

    def get_object_clips(
        self,
        object_audio_map: Dict[int, AudioSpec],
        step: int,
        event_flags: Optional[Dict[int, bool]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Returns dict: object_id -> clip waveform (Tclip,)
        """
        event_flags = event_flags or {}
        out: Dict[int, torch.Tensor] = {}

        for obj_id, spec in object_audio_map.items():
            if not self.is_active(spec, step, event_flags.get(obj_id, False)):
                out[obj_id] = torch.zeros(self.clip_len, dtype=torch.float32, device=self.device)
                continue

            wav, sr = self._load_wav(spec.wav_path)
            wav = self._resample_if_needed(wav, sr)  # (1, T) at sample_rate
            wav = wav.to(self.device)

            # choose deterministic start based on step (stable over time)
            start = (step * self.clip_len) % max(1, wav.shape[-1])
            clip = self._clip_from_wave(wav, start=start) * float(spec.gain)
            out[obj_id] = clip

        return out

    def mix_clips(self, clips: Dict[int, torch.Tensor], noise_std: float = 0.0) -> torch.Tensor:
        """
        Optional: create a mixed "virtual microphone" stream.
        """
        if len(clips) == 0:
            mix = torch.zeros(self.clip_len, dtype=torch.float32, device=self.device)
        else:
            mix = torch.stack(list(clips.values()), dim=0).sum(dim=0)

        if noise_std > 0:
            mix = mix + noise_std * torch.randn_like(mix)
        return mix

