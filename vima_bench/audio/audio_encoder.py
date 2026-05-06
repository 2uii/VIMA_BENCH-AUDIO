from typing import Dict

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2TokenEncoder(nn.Module):
    """
    Frozen wav2vec2 encoder -> pooled embedding -> projected token for VIMA.

    Important: we feed raw waveform directly as (B, T) float32 at 16kHz.
    This avoids shape issues introduced by processor behavior in newer Transformers.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        target_sample_rate: int = 16000,
        token_dim: int = 768,
        proj_hidden: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.target_sample_rate = target_sample_rate

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze wav2vec2
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        enc_dim = self.encoder.config.hidden_size

        # Trainable projection into token dim
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, token_dim),
        )

        self.to(self.device)

    @torch.no_grad()
    def _encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, T) float32 at 16kHz
        returns: (B, enc_dim) pooled embedding
        """
        wav = wav.to(self.device).to(torch.float32)

        # Wav2Vec2 expects (B, T)
        out = self.encoder(input_values=wav)
        feats = out.last_hidden_state  # (B, S, enc_dim)
        pooled = feats.mean(dim=1)     # (B, enc_dim)
        return pooled

    def forward(self, object_clips: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        object_clips: obj_id -> (Tclip,) waveform
        returns: obj_id -> (token_dim,) audio token
        """
        obj_ids = list(object_clips.keys())
        if len(obj_ids) == 0:
            return {}

        batch = torch.stack([object_clips[i] for i in obj_ids], dim=0)  # (B, T)
        pooled = self._encode(batch)
        tokens = self.proj(pooled)  # (B, token_dim)

        return {obj_id: tokens[k] for k, obj_id in enumerate(obj_ids)}
