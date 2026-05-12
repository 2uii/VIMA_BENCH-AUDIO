import numpy as np
import torch
import torch.nn as nn

from scripts.example import prepare_prompt, prepare_obs
from vima.policy.vima_policy import VIMAPolicy
from vima.utils import add_batch_dim


DEVICE = "cpu"
CKPT_PATH = "true_trimodal_vima_policy.pt"


class TrueTrimodalVIMABackend:
    def __init__(self):
        self.policy = VIMAPolicy(
            embed_dim=768,
            xf_n_layers=1,
            sattn_n_heads=8,
            xattn_n_heads=8,
        ).to(DEVICE)

        self.policy.t5_prompt_encoder = None

        self.action_head = nn.Linear(768, 4).to(DEVICE)

        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

        if "model_state" in ckpt:
            self.policy.load_state_dict(ckpt["model_state"], strict=False)

        if "action_head_state" in ckpt:
            self.action_head.load_state_dict(ckpt["action_head_state"], strict=False)

        self.policy.eval()
        self.action_head.eval()

    def predict(self, prompt, prompt_assets, obs, meta, rgb_top_path=None, rgb_front_path=None):
        prompt_token_type, word_batch, image_batch, audio_batch = prepare_prompt(
            prompt=prompt,
            prompt_assets=prompt_assets,
            views=["front", "top"],
        )

        word_batch = word_batch.to(DEVICE)
        image_batch = image_batch.to_torch_tensor(device=DEVICE)
        audio_batch = audio_batch.to(DEVICE)

        prompt_tokens, prompt_masks = self.policy.forward_prompt_assembly(
            (prompt_token_type, word_batch, image_batch, audio_batch)
        )

        obs = dict(obs)

        if "rgb" not in obs:
            if rgb_top_path is None or rgb_front_path is None:
                raise ValueError("obs has no rgb, so rgb_top_path and rgb_front_path are required.")

            import imageio.v2 as imageio

            obs["rgb"] = {
                "top": np.asarray(imageio.imread(rgb_top_path)).transpose(2, 0, 1),
                "front": np.asarray(imageio.imread(rgb_front_path)).transpose(2, 0, 1),
            }

        if "segm" in obs:
            obs["segm"] = {
                "front": np.asarray(obs["segm"]["front"])[0]
                if np.asarray(obs["segm"]["front"]).ndim == 3
                else np.asarray(obs["segm"]["front"]),

                "top": np.asarray(obs["segm"]["top"])[0]
                if np.asarray(obs["segm"]["top"]).ndim == 3
                else np.asarray(obs["segm"]["top"]),
            }

        ee_arr = np.asarray(obs["ee"])
        if ee_arr.ndim == 0:
            obs["ee"] = np.asarray([int(ee_arr)], dtype=np.int64)
        else:
            obs["ee"] = np.asarray([ee_arr.flatten()[0]], dtype=np.int64)

        obs_batched = add_batch_dim(obs)

        obs_prepared = prepare_obs(
            obs=obs_batched,
            rgb_dict=None,
            meta=meta,
        ).to_torch_tensor(device=DEVICE)

        if obs_prepared["ee"].ndim == 3:
            obs_prepared["ee"] = obs_prepared["ee"].squeeze(-1)

        obs_token, obs_mask = self.policy.forward_obs_token(obs_prepared)

        with torch.no_grad():
            pred_action_tokens = self.policy(
                obs_token=obs_token,
                obs_mask=obs_mask,
                action_token=None,
                prompt_token=prompt_tokens,
                prompt_token_mask=prompt_masks,
            )

            pred_action_tokens = pred_action_tokens[-1].unsqueeze(0)

            pred_action_tokens = torch.nan_to_num(
                pred_action_tokens,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )

            pred_action_vec = self.action_head(pred_action_tokens).squeeze(0)

        return {
            "backend": "true_trimodal_vima",
            "prompt": prompt,
            "prompt_token_type": prompt_token_type,
            "prompt_tokens_shape": tuple(prompt_tokens.shape),
            "obs_token_shape": tuple(obs_token.shape),
            "pred_action_vec": pred_action_vec.cpu().numpy().tolist(),
        }
