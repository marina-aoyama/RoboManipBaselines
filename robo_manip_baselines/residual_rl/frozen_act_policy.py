import os
import pickle
import sys

import numpy as np
import torch
from torchvision.transforms import v2

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "act")
)
from detr.models.detr_vae import DETRVAE  # noqa: E402
from policy import ACTPolicy  # noqa: E402

from robo_manip_baselines.common import denormalize_data, normalize_data  # noqa: E402


class FrozenActPolicy:
    """Loads a frozen, pretrained ACT checkpoint and replicates the inference
    behavior of `robo_manip_baselines.policy.act.RolloutAct` (skip-decimated
    chunk prediction with temporal ensembling) for headless use outside the
    interactive `RolloutBase` machinery.

    `get_base_action` is meant to be called once per simulation step so its
    `skip`/temporal-ensembling cadence matches the deployed ACT policy exactly.
    """

    def __init__(self, checkpoint_path, device="cuda"):
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        with open(os.path.join(checkpoint_dir, "model_meta_info.pkl"), "rb") as f:
            self.model_meta_info = pickle.load(f)

        self.state_keys = self.model_meta_info["state"]["keys"]
        if self.state_keys != ["measured_joint_pos"]:
            raise ValueError(
                "FrozenActPolicy assumes the ACT checkpoint was trained with "
                f"state_keys=['measured_joint_pos'], got {self.state_keys}"
            )

        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.skip = self.model_meta_info["data"]["skip"]
        self.chunk_size = self.model_meta_info["data"]["chunk_size"]
        self.action_dim = len(self.model_meta_info["action"]["example"])

        self.device = torch.device(device)

        DETRVAE.set_state_dim(len(self.model_meta_info["state"]["example"]))
        DETRVAE.set_action_dim(self.action_dim)
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

        self.reset()

    def reset(self):
        """Reset the temporal ensembling history. Call at the start of each episode."""
        self.step_idx = 0
        self.policy_action_buf = []
        self.policy_action_buf_history = []
        self.policy_action = None

    def get_base_action(self, obs, rgb_images):
        """Get the base ACT action (denormalized, 7-dim: 6 arm + 1 gripper).

        Args:
            obs: env observation dict (must contain "joint_pos").
            rgb_images: dict mapping camera_name -> HWC uint8 image (e.g. `info["rgb_images"]`).
        """
        if self.step_idx % self.skip == 0:
            with torch.inference_mode():
                state = self._get_state(obs)
                images = self._get_images(rgb_images)
                action = self.policy(state, images)[0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )
            self.policy_action_buf_history.append(self.policy_action_buf)
            if len(self.policy_action_buf_history) > self.chunk_size:
                self.policy_action_buf_history.pop(0)

            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(self.policy_action_buf_history)))
            exp_weights = exp_weights / exp_weights.sum()
            ensembled_action = np.zeros(self.action_dim)
            for action_idx, _policy_action_buf in enumerate(
                reversed(self.policy_action_buf_history)
            ):
                ensembled_action += (
                    exp_weights[::-1][action_idx] * _policy_action_buf[action_idx]
                )
            self.policy_action = denormalize_data(
                ensembled_action, self.model_meta_info["action"]
            )

        self.step_idx += 1
        return self.policy_action

    def _get_state(self, obs):
        state = normalize_data(obs["joint_pos"], self.model_meta_info["state"])
        return torch.tensor(state[np.newaxis], dtype=torch.float32).to(self.device)

    def _get_images(self, rgb_images):
        images = np.stack(
            [rgb_images[camera_name] for camera_name in self.camera_names], axis=0
        )
        images = np.moveaxis(images, -1, -3)
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[torch.newaxis].to(self.device)
        return images
