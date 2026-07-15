import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Registers "robo_manip_baselines/MujocoUR5eToolboxEnv-v0" with gymnasium.
import robo_manip_baselines.envs  # noqa: F401

from .frozen_act_policy import FrozenActPolicy


class ResidualToolboxEnv(gym.Env):
    """Wraps `MujocoUR5eToolboxEnv` (target_task="pick_and_place") together with
    a frozen ACT base policy. The action is a small residual added on top of
    ACT's own 7-dim (6 arm + 1 gripper) joint-position command; one RL step
    corresponds to one ACT inference "skip"-block, matching the cadence ACT is
    deployed with.

    Observation (10-dim): robot `joint_pos` (7, the same proprio ACT itself
    conditions on) + `toolbox` xyz position (3).

    Reward is the sparse `pick_and_place` success indicator (1.0/0.0), which
    also ends the episode on success. `pick_success` is tracked separately and
    reported in `info` purely for diagnostics (it does not affect reward) --
    both are computed directly here (duplicating `MujocoUR5eToolboxEnv.
    _get_reward()`'s thresholds) since the env's own `_get_reward()` only
    reports whichever single `target_task` is configured, not both at once.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        act_checkpoint,
        world_idx_list=None,
        residual_action_scale_arm=0.05,
        residual_action_scale_gripper=10.0,
        max_episode_duration=30.0,
        render_mode=None,
        device="cuda",
    ):
        super().__init__()

        self.base_env = gym.make(
            "robo_manip_baselines/MujocoUR5eToolboxEnv-v0", render_mode=render_mode
        )
        self.base_env.unwrapped.target_task = "pick_and_place"

        self.world_idx_list = (
            list(world_idx_list) if world_idx_list is not None else list(range(6))
        )
        self.max_episode_duration = max_episode_duration

        self.frozen_act = FrozenActPolicy(act_checkpoint, device=device)

        self._residual_scale = np.array(
            [residual_action_scale_arm] * 6 + [residual_action_scale_gripper]
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float64
        )

        self.render_mode = render_mode

    def _make_obs(self, base_obs):
        toolbox_pos = self.base_env.unwrapped.get_body_pose("toolbox")[:3]
        return np.concatenate([base_obs["joint_pos"], toolbox_pos]).astype(np.float64)

    def _get_success_flags(self):
        """Replicates MujocoUR5eToolboxEnv._get_reward()'s pick/pick_and_place
        thresholds, computed independently of `target_task` so both are
        available at once."""
        toolbox_pos = self.base_env.unwrapped.data.body("toolbox").xpos
        mat_pos = self.base_env.unwrapped.data.body("mat").xpos

        pick_z_thre = mat_pos[2] + 0.01
        pick_success = toolbox_pos[2] > pick_z_thre

        pap_xy_thre = 0.03
        pap_z_thre = mat_pos[2] + 0.005
        pap_success = (
            np.max(np.abs(toolbox_pos[:2] - mat_pos[:2])) < pap_xy_thre
        ) and (toolbox_pos[2] < pap_z_thre)

        return bool(pick_success), bool(pap_success)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "world_idx" in options:
            world_idx = options["world_idx"]
        else:
            world_idx = int(self.np_random.choice(self.world_idx_list))
        self.base_env.unwrapped.modify_world(world_idx=world_idx)

        base_obs, info = self.base_env.reset(seed=seed)
        self.frozen_act.reset()
        self._episode_start_time = self.base_env.unwrapped.get_time()
        self._base_obs = base_obs
        self._info = info
        self._episode_pick_success = False

        return self._make_obs(base_obs), info

    def step(self, residual_action):
        residual_action = np.clip(residual_action, -1.0, 1.0) * self._residual_scale

        pick_success = False
        pap_success = False
        base_obs, info = self._base_obs, self._info
        for _ in range(self.frozen_act.skip):
            base_action = self.frozen_act.get_base_action(
                base_obs, info["rgb_images"]
            )
            combined_action = np.clip(
                base_action + residual_action,
                self.base_env.action_space.low,
                self.base_env.action_space.high,
            )
            base_obs, _, _, _, info = self.base_env.step(combined_action)
            step_pick_success, step_pap_success = self._get_success_flags()
            pick_success = pick_success or step_pick_success
            pap_success = pap_success or step_pap_success

        self._base_obs = base_obs
        self._info = info
        if pick_success:
            self._episode_pick_success = True

        reward = 1.0 if pap_success else 0.0

        elapsed_duration = self.base_env.unwrapped.get_time() - self._episode_start_time
        terminated = pap_success
        truncated = (not pap_success) and (elapsed_duration >= self.max_episode_duration)
        info = {
            **info,
            "success": pap_success,
            "pick_success": self._episode_pick_success,
        }

        return self._make_obs(base_obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
