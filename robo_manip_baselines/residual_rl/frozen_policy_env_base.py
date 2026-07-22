import gymnasium as gym
import numpy as np

# Registers "robo_manip_baselines/<EnvClassName>-v0" with gymnasium.
import robo_manip_baselines.envs  # noqa: F401
from robo_manip_baselines.common import MotionManager


class FrozenPolicyEnvBase(gym.Env):
    """Shared machinery for wrapping any registered `robo_manip_baselines`
    env with an RL layer on top of a frozen base policy: env construction,
    world-idx reset cycling via the wrapped env's own `modify_world()`,
    success/termination/truncation bookkeeping driven by the wrapped env's
    own `_get_reward()`, and the pre-training reward sanity check. Used by
    both `ResidualEnv` (action-space residual) and `DsrlEnv` (noise-space
    steering) -- the two differ only in what the RL action *is* and how
    it's turned into an env-executable command, not in any of this.

    Subclasses must set `self.action_space`/`self.observation_space` in
    their own `__init__` (after calling `super().__init__(...)`) and
    implement `_make_obs(base_obs)` and `step(raw_action)` (calling
    `self._finalize_step(reward, base_obs, info)` at the end of `step` to
    get the shared success/termination/truncation/info handling).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, env_id, config, world_idx_list, max_episode_duration, render_mode):
        super().__init__()

        self.config = config
        self.world_idx_list = list(world_idx_list) if world_idx_list is not None else [0]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode

        self.base_env = gym.make(env_id, render_mode=render_mode)
        self.motion_manager = MotionManager(self.base_env)
        self.arm_manager = self.motion_manager.body_manager_list[0]

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

        self._base_obs = base_obs
        self._info = info
        self._episode_start_time = self.base_env.unwrapped.get_time()
        self._best_reward_so_far = 0.0
        self._time_of_best_reward = 0.0

        return base_obs, info

    def _finalize_step(self, reward, base_obs, info):
        """Shared success/termination/truncation/workspace-bounds logic,
        driven entirely by the wrapped env's own reward -- see
        `ResidualEnv`'s module docstring for why this is deliberately
        env-agnostic (no object-pose-dependent shaping)."""
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
        return terminated, truncated, info

    def sanity_check_reward(self, num_episodes=3, max_steps=None, verbose=True):
        """Runs a few episodes with an all-zeros RL action (for `DsrlEnv`
        this means noise=0, an arbitrary-but-valid deterministic point in
        noise space, not "unsteered") and reports whether any non-zero
        reward was ever observed. Meant to be called once before spending
        compute on training -- catches a dead/misconfigured reward signal
        (e.g. wrong `success_reward_threshold`, or an env whose
        `_get_reward()` never fires for reasons unrelated to policy
        quality) in seconds instead of after a full training run against
        noise.

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
                    f"{num_episodes}-episode check. Either the test action never "
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
