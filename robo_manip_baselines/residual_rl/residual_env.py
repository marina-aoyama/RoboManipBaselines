import numpy as np
from gymnasium import spaces

from robo_manip_baselines.common import DataKey

from .config import ResidualRlConfig
from .frozen_maniflow_policy import FrozenManiFlowPolicy
from .frozen_policy_env_base import FrozenPolicyEnvBase

# Keys whose policy-space vector is [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
# (a 6-dim "rel" pose is always 3 translation + 3 rotation by construction --
# see DataKey.get_dim / MathUtils.get_se3_from_rel_pose -- so this is a
# property of the *key*, not of any particular env or checkpoint).
_POSE_REL_KEYS = (DataKey.COMMAND_EEF_POSE_REL,)
# Keys whose policy-space vector is [pos_x, pos_y, pos_z, rot 6D...] (9-dim
# absolute pose representation -- see get_pose9_from_pose7 / DataKey.
# get_dim_for_policy).
_POSE_ABS_KEYS = (DataKey.COMMAND_EEF_POSE,)
_GRIPPER_KEYS = (DataKey.COMMAND_GRIPPER_JOINT_POS, DataKey.COMMAND_GRIPPER_JOINT_POS_REL)
_JOINT_KEYS = (DataKey.COMMAND_JOINT_POS, DataKey.COMMAND_JOINT_POS_REL)


class ResidualEnv(FrozenPolicyEnvBase):
    """Generic residual-RL / full-RL wrapper: works with any registered
    `robo_manip_baselines` env (via `env_id`, e.g.
    "robo_manip_baselines/MujocoUR5eInsertEnv-v0") and, in "residual" mode,
    any frozen ManiFlow checkpoint -- no per-task subclassing needed.

    This intentionally does NOT touch the wrapped env's own Python class:
    every task env already exposes what's needed generically --
    `_get_reward()` (returned through the standard `step()` tuple),
    `modify_world(world_idx)` for reset diversity, and (via
    `MotionManager`/`ArmManager`) FK/IK for the robot's eef pose. See
    `MujocoUR5eInsertEnv._get_reward` for a concrete example of the binary
    success signal this wrapper consumes as-is. Shared reset/success/
    termination bookkeeping lives in `FrozenPolicyEnvBase` -- see there for
    that, and see `DsrlEnv` for the noise-space-steering sibling of this
    action-space-residual approach.

    Action composition depends on `config.policy_mode`:
    - "residual" (default): the frozen ManiFlow policy's own action
      representation (whatever `action_keys` its checkpoint was trained
      with -- e.g. `command_eef_pose_rel` + `command_gripper_joint_pos` for
      the current Insert checkpoint) is read from its meta info, and the
      RL action is a small additive residual in that same policy-space
      representation, split and applied via `MotionManager.set_command_data`
      exactly like `RolloutBase` applies the checkpoint's own actions.
    - "full": no base-policy prior -- the `[-1,1]` action is mapped directly
      onto either the raw actuator range (`full_rl.action_space="joint"`) or
      a small eef-space delta solved via IK (`"cartesian"`), for training a
      from-scratch baseline with the same env/reward/reset machinery.

    Observation is generic and object-pose-free by design (the "basics
    first" starting point -- see PR discussion): robot proprioception (the
    same state the base policy conditions on, in "residual" mode) plus, in
    "residual" mode, the base policy's own proposed action (so the residual
    policy can see what it's correcting). Dense task-specific reward shaping
    (e.g. distance to an object) is deliberately not included -- add it back
    per-task later if the sparse signal alone proves too slow to learn from,
    rather than baking object-specific state into this generic wrapper.
    """

    def __init__(
        self,
        env_id,
        config=None,
        maniflow_checkpoint=None,
        world_idx_list=None,
        residual_action_scale=None,
        max_episode_duration=30.0,
        render_mode=None,
        device="cuda",
    ):
        config = config if config is not None else ResidualRlConfig()
        super().__init__(env_id, config, world_idx_list, max_episode_duration, render_mode)

        if self.config.policy_mode == "residual":
            if maniflow_checkpoint is None:
                raise ValueError("policy_mode='residual' requires maniflow_checkpoint")
            self.frozen_policy = FrozenManiFlowPolicy(
                maniflow_checkpoint, self.motion_manager, device=device
            )
            self.skip = self.frozen_policy.skip
            action_dim = self.frozen_policy.action_dim
            self._residual_scale = self._build_residual_scale(
                self.frozen_policy.action_keys,
                self.frozen_policy.action_dims,
                residual_action_scale or {},
            )
            state_dim = sum(self.frozen_policy.state_dims)
            obs_dim = state_dim + action_dim
        else:  # policy_mode == "full"
            self.frozen_policy = None
            self.skip = 1
            if self.config.full_rl.action_space == "cartesian":
                action_dim = 7  # xyz + rpy delta + gripper
            else:
                action_dim = self.base_env.action_space.shape[0]
            obs_dim = self.base_env.observation_space["joint_pos"].shape[0]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )

    def _build_residual_scale(self, action_keys, action_dims, scale_config):
        """Expands named per-key-type scales (e.g. {"pos": 0.01, "rot": 0.05,
        "gripper": 10.0, "joint": 0.05}) into a flat array matching the
        concatenated action_keys order -- keeps the caller from having to
        know the exact dimension layout of whatever action representation
        the checkpoint happens to use."""
        defaults = {"pos": 0.01, "rot": 0.05, "gripper": 10.0, "joint": 0.05}
        scale_config = {**defaults, **scale_config}

        scale = []
        for key, dim in zip(action_keys, action_dims):
            if key in _POSE_REL_KEYS:
                assert dim == 6, f"unexpected dim {dim} for {key}"
                scale += [scale_config["pos"]] * 3 + [scale_config["rot"]] * 3
            elif key in _POSE_ABS_KEYS:
                assert dim == 9, f"unexpected dim {dim} for {key}"
                scale += [scale_config["pos"]] * 3 + [scale_config["rot"]] * 6
            elif key in _GRIPPER_KEYS:
                scale += [scale_config["gripper"]] * dim
            elif key in _JOINT_KEYS:
                scale += [scale_config["joint"]] * dim
            else:
                raise ValueError(f"No default residual scale known for action key {key}")
        return np.array(scale)

    def _make_obs(self, base_obs, base_action=None):
        if self.frozen_policy is None:
            return base_obs["joint_pos"].astype(np.float64)
        state = self.frozen_policy.get_raw_state(base_obs)
        if base_action is None:
            base_action = np.zeros(self.frozen_policy.action_dim)
        return np.concatenate([state, base_action]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        if self.frozen_policy is not None:
            self.frozen_policy.reset()
        self._last_base_action = None
        return self._make_obs(base_obs), info

    def step(self, raw_action):
        raw_action = np.clip(raw_action, -1.0, 1.0)

        base_obs, info = self._base_obs, self._info
        reward = 0.0
        for sub_idx in range(self.skip):
            is_skip = sub_idx != 0

            if self.config.policy_mode == "full" and self.config.full_rl.action_space == "cartesian":
                if not is_skip:
                    pos_delta = raw_action[:3] * self.config.full_rl.cartesian_pos_scale
                    rot_delta = raw_action[3:6] * self.config.full_rl.cartesian_rot_scale
                    gripper_low = self.base_env.action_space.low[-1]
                    gripper_high = self.base_env.action_space.high[-1]
                    gripper_ctrl = gripper_low + (raw_action[6] + 1.0) / 2.0 * (
                        gripper_high - gripper_low
                    )
                self.motion_manager.set_command_data(
                    DataKey.COMMAND_EEF_POSE_REL, np.concatenate([pos_delta, rot_delta]), is_skip
                )
                self.motion_manager.set_command_data(
                    DataKey.COMMAND_GRIPPER_JOINT_POS, np.array([gripper_ctrl]), is_skip
                )
                env_action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            elif self.config.policy_mode == "full":
                low = self.base_env.action_space.low
                high = self.base_env.action_space.high
                env_action = low + (raw_action + 1.0) / 2.0 * (high - low)
            else:
                if not is_skip:
                    base_action = self.frozen_policy.get_base_action(
                        base_obs, info["rgb_images"]
                    )
                    self._last_base_action = base_action
                    combined_action = base_action + raw_action * self._residual_scale
                self.frozen_policy.apply_action(combined_action, is_skip)
                env_action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)

            env_action = np.clip(
                env_action, self.base_env.action_space.low, self.base_env.action_space.high
            )
            base_obs, step_reward, _, _, info = self.base_env.step(env_action)
            reward = max(reward, step_reward)

        self._base_obs = base_obs
        self._info = info

        terminated, truncated, info = self._finalize_step(reward, base_obs, info)

        return (
            self._make_obs(base_obs, self._last_base_action),
            float(reward),
            terminated,
            truncated,
            info,
        )
