import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Registers "robo_manip_baselines/<EnvClassName>-v0" with gymnasium.
import robo_manip_baselines.envs  # noqa: F401
from robo_manip_baselines.common import DataKey, MotionManager, convert_data_to_policy

from .config import ResidualRlConfig
from .frozen_maniflow_policy import FrozenManiFlowPolicy

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


class ResidualEnv(gym.Env):
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
    success signal this wrapper consumes as-is.

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

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

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
        super().__init__()

        self.config = config if config is not None else ResidualRlConfig()
        self.world_idx_list = list(world_idx_list) if world_idx_list is not None else [0]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode

        self.base_env = gym.make(env_id, render_mode=render_mode)
        self.motion_manager = MotionManager(self.base_env)
        self.arm_manager = self.motion_manager.body_manager_list[0]

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

    def _get_base_policy_state(self, base_obs):
        return np.concatenate(
            [
                convert_data_to_policy(self.motion_manager.get_data(key, base_obs), key)
                for key in self.frozen_policy.state_keys
            ]
        ).astype(np.float64)

    def _make_obs(self, base_obs, base_action=None):
        if self.frozen_policy is None:
            return base_obs["joint_pos"].astype(np.float64)
        state = self._get_base_policy_state(base_obs)
        if base_action is None:
            base_action = np.zeros(self.frozen_policy.action_dim)
        return np.concatenate([state, base_action]).astype(np.float64)

    def _get_gripper_pos(self):
        self.arm_manager.forward_kinematics()
        return self.arm_manager.current_se3.translation.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "world_idx" in options:
            world_idx = options["world_idx"]
        else:
            world_idx = int(self.np_random.choice(self.world_idx_list))
        self.base_env.unwrapped.modify_world(world_idx=world_idx)
        base_obs, info = self.base_env.reset()

        self.motion_manager.reset()
        if self.frozen_policy is not None:
            self.frozen_policy.reset()

        self._base_obs = base_obs
        self._info = info
        self._episode_start_time = self.base_env.unwrapped.get_time()
        self._best_reward_so_far = 0.0
        self._time_of_best_reward = 0.0
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

        success = reward >= self.config.success_reward_threshold
        elapsed_duration = self.base_env.unwrapped.get_time() - self._episode_start_time

        if reward > self._best_reward_so_far:
            self._best_reward_so_far = reward
            self._time_of_best_reward = elapsed_duration

        terminated = bool(success)
        truncated = (not terminated) and (elapsed_duration >= self.max_episode_duration)

        if self.config.early_truncation.enabled and not terminated and not truncated:
            if (
                elapsed_duration - self._time_of_best_reward
                >= self.config.early_truncation.patience_seconds
            ):
                truncated = True

        out_of_bounds = False
        if self.config.workspace_bounds.enabled and not terminated and not truncated:
            gripper_pos = self._get_gripper_pos()
            wb = self.config.workspace_bounds
            out_of_bounds = not (
                wb.x_min <= gripper_pos[0] <= wb.x_max
                and wb.y_min <= gripper_pos[1] <= wb.y_max
                and wb.z_min <= gripper_pos[2] <= wb.z_max
            )
            if out_of_bounds:
                truncated = True

        info = {**info, "success": success, "out_of_bounds": out_of_bounds}

        return (
            self._make_obs(base_obs, self._last_base_action),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def sanity_check_reward(self, num_episodes=3, max_steps=None, verbose=True):
        """Runs a few episodes with a zero residual (pure base policy, or in
        "full" mode a no-op zero action) and reports whether any non-zero
        reward was ever observed. Meant to be called once before spending
        compute on training -- catches a dead/misconfigured reward signal
        (e.g. wrong `success_reward_threshold`, or an env whose
        `_get_reward()` never fires for reasons unrelated to policy quality)
        in seconds instead of after a full training run against noise.

        Returns (any_nonzero_reward_observed, max_reward_observed).
        """
        max_reward_observed = 0.0
        for episode_idx in range(num_episodes):
            world_idx = self.world_idx_list[episode_idx % len(self.world_idx_list)]
            obs, info = self.reset(options={"world_idx": world_idx})
            terminated = truncated = False
            step_idx = 0
            episode_max_reward = 0.0
            zero_action = np.zeros(self.action_space.shape)
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.step(zero_action)
                episode_max_reward = max(episode_max_reward, reward)
                step_idx += 1
                if max_steps is not None and step_idx >= max_steps:
                    break
            max_reward_observed = max(max_reward_observed, episode_max_reward)
            if verbose:
                print(
                    f"[sanity_check_reward] episode {episode_idx} "
                    f"(world_idx={world_idx}): max_reward={episode_max_reward:.3f}, "
                    f"{'success' if info.get('success') else 'no success'}"
                )

        any_nonzero = max_reward_observed > 0.0
        if verbose:
            if any_nonzero:
                print(
                    f"[sanity_check_reward] OK: observed non-zero reward "
                    f"(max={max_reward_observed:.3f}) across {num_episodes} episodes."
                )
            else:
                print(
                    f"[sanity_check_reward] WARNING: reward was 0.0 for the entire "
                    f"{num_episodes}-episode check. Either the base policy never "
                    f"succeeds within max_episode_duration on world_idx_list="
                    f"{self.world_idx_list}, or success_reward_threshold "
                    f"({self.config.success_reward_threshold}) doesn't match this "
                    f"env's _get_reward() scale -- check both before training."
                )
        return any_nonzero, max_reward_observed

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
