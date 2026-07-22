import os
import pickle
import sys

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "third_party", "ManiFlow_Policy", "ManiFlow"
    )
)

from robo_manip_baselines.common import (  # noqa: E402
    DataKey,
    convert_data_from_policy,
    convert_data_to_policy,
    denormalize_data,
    normalize_data,
)


class FrozenManiFlowPolicy:
    """Loads a frozen, pretrained ManiFlow checkpoint and replicates the
    inference behavior of `robo_manip_baselines.policy.mani_flow_policy.
    RolloutManiFlowPolicy` (skip-decimated chunk prediction) for headless use
    outside the interactive `RolloutBase` machinery.

    Unlike `FrozenActPolicy` (ACT-only, hardcoded to raw `command_joint_pos`
    actions), this makes no assumption about the checkpoint's action/state
    representation -- it reads `state_keys`/`action_keys` from the
    checkpoint's own meta info and uses `MotionManager` (the same one the
    wrapping env's IK/FK goes through) to assemble/apply them generically.
    That means it works unmodified whether the checkpoint predicts raw joint
    targets or (as with the current Insert checkpoint) `command_eef_pose_rel`
    + `command_gripper_joint_pos`.

    Only supports `policy_type == "image"` for now (matches the checkpoints
    in use); a "pointcloud" checkpoint will raise clearly rather than
    silently doing the wrong thing.

    `get_base_action` is meant to be called once per simulation step so its
    `skip` cadence matches the deployed ManiFlow policy exactly -- the caller
    (see `ResidualEnv.step`) is responsible for calling `apply_action` with
    `is_skip=True` (hold, don't reapply the delta) on the steps in between a
    fresh call to `get_base_action`, exactly as `RolloutBase.run()` does via
    its own `rollout_time_idx % skip` cadence. Re-applying a fresh
    (non-held) relative action on every sub-step instead of holding it would
    silently compound eef-pose deltas `skip`x too far.
    """

    def __init__(self, checkpoint_path, motion_manager, device="cuda"):
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        with open(os.path.join(checkpoint_dir, "model_meta_info.pkl"), "rb") as f:
            self.model_meta_info = pickle.load(f)

        self.policy_type = self.model_meta_info["policy"]["policy_type"]
        if self.policy_type != "image":
            raise NotImplementedError(
                f"FrozenManiFlowPolicy only supports policy_type='image' checkpoints, "
                f"got {self.policy_type!r}. Pointcloud checkpoints need a separate "
                f"observation path (point_cloud construction) that hasn't been "
                f"wired up here yet."
            )

        self.motion_manager = motion_manager
        env = motion_manager.env

        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.state_dims = [DataKey.get_dim_for_policy(key, env) for key in self.state_keys]
        self.action_dims = [
            DataKey.get_dim_for_policy(key, env) for key in self.action_keys
        ]
        self.action_dim = sum(self.action_dims)

        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.skip = self.model_meta_info["data"]["skip"]
        self.n_obs_steps = self.model_meta_info["data"]["n_obs_steps"]
        self.image_size = self.model_meta_info["data"]["image_size"]

        self.device = torch.device(device)

        from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy

        self.policy = ManiFlowTransformerImagePolicy(
            **self.model_meta_info["policy"]["args"],
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

        self.reset()

    def reset(self):
        """Reset observation/action-chunk history. Call at the start of each episode."""
        self.state_buf = None
        self.images_buf = None
        self.policy_action_buf = None

    def get_base_action(self, obs, rgb_images):
        """Returns the base policy's next action as a flat, denormalized,
        policy-space vector (length `self.action_dim`, concatenated in
        `self.action_keys` order) -- NOT yet converted to env/actuator space
        or combined with any residual. Call `apply_action` (or replicate its
        `convert_data_from_policy` + `motion_manager.set_command_data` loop)
        to actually drive the robot with it.
        """
        self._update_state_buf(obs)
        self._update_images_buf(rgb_images)

        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data = {"state": self._get_state()}
            for camera_name, image in zip(self.camera_names, self._get_images()):
                input_data[DataKey.get_rgb_image_key(camera_name)] = image
            with torch.inference_mode():
                action = self.policy.predict_action(input_data)["action"][0]
            self.policy_action_buf = list(
                action.cpu().detach().numpy().astype(np.float64)
            )

        return denormalize_data(
            self.policy_action_buf.pop(0), self.model_meta_info["action"]
        )

    def apply_action(self, action_vec, is_skip=False):
        """Splits a flat policy-space action vector (as returned by
        `get_base_action`, or that plus a residual) along `self.action_keys`
        and applies each piece via `MotionManager.set_command_data`."""
        idx = 0
        for key, dim in zip(self.action_keys, self.action_dims):
            command = convert_data_from_policy(action_vec[idx : idx + dim], key)
            self.motion_manager.set_command_data(key, command, is_skip)
            idx += dim

    def _update_state_buf(self, obs):
        state = np.concatenate(
            [
                convert_data_to_policy(self.motion_manager.get_data(key, obs), key)
                for key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [state for _ in range(self.n_obs_steps)]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def _get_state(self):
        return torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

    def _update_images_buf(self, rgb_images):
        images = []
        for camera_name in self.camera_names:
            image = cv2.resize(rgb_images[camera_name], self.image_size)
            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            # Adjust to a range from -1 to 1 to match ManiFlow's own training pipeline
            image = image * 2.0 - 1.0
            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.n_obs_steps)] for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

    def _get_images(self):
        return [
            torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
            for single_images_buf in self.images_buf
        ]
