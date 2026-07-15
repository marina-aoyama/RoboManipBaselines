import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Registers "robo_manip_baselines/MujocoUR5eToolboxEnv-v0" with gymnasium.
import robo_manip_baselines.envs  # noqa: F401
from robo_manip_baselines.common import MotionManager

from .config import ResidualRlConfig
from .diverse_reset import CachedResetSampler, choose_reset_category
from .frozen_act_policy import FrozenActPolicy

DEFAULT_RESET_DATASET_PATH = os.path.join(os.path.dirname(__file__), "reset_dataset.pkl")


class ResidualToolboxEnv(gym.Env):
    """Wraps `MujocoUR5eToolboxEnv` (target_task="pick_and_place") with a
    frozen ACT base policy. Action composition depends on `config.policy_mode`:
    - "residual" (default): the 7-dim [-1,1] action is a small residual added
      on top of ACT's own joint-position command; one RL step corresponds to
      one ACT inference "skip"-block, matching the cadence ACT is deployed
      with.
    - "full": no ACT prior at all -- the [-1,1] action is mapped directly onto
      the full actuator range, for training a from-scratch policy (e.g. as a
      baseline comparison against residual RL) using the same env/reward/
      reset machinery.

    Observation (10-dim): robot `joint_pos` (7, the same proprio ACT itself
    conditions on) + `toolbox` xyz position (3), regardless of policy_mode.

    Reward/reset behavior is entirely driven by `config` (a `ResidualRlConfig`,
    see `config.py` and `configs/*.yaml`): whether dense reach/dist reward
    terms are added on top of the sparse `pick_and_place` success indicator,
    what mix of {default, near_object, stable_grasp, near_goal} reset states
    episodes start from, and whether dead episodes (gripper stuck far from the
    object, never having picked it) are truncated early. `pick_success` is
    always tracked and reported in `info` regardless of config, purely for
    diagnostics -- both `pick_success`/`pap_success` are computed directly
    here (duplicating `MujocoUR5eToolboxEnv._get_reward()`'s thresholds) since
    the env's own `_get_reward()` only reports whichever single `target_task`
    is configured, not both at once.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        act_checkpoint,
        config=None,
        world_idx_list=None,
        residual_action_scale_arm=0.05,
        residual_action_scale_gripper=10.0,
        max_episode_duration=30.0,
        reset_dataset_path=None,
        render_mode=None,
        device="cuda",
    ):
        super().__init__()

        self.config = config if config is not None else ResidualRlConfig()

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

        # A shared MotionManager/ArmManager (Pinocchio-based FK/IK) is only
        # constructed when actually needed -- keeps configs/baseline.yaml
        # exactly as lightweight as the original setup. Resets themselves
        # don't need it: they're a lookup into a pre-validated dataset
        # (generate_reset_dataset.py), not live IK.
        needs_arm_manager = (
            self.config.reach.enabled
            or self.config.early_truncation.enabled
            or self.config.workspace_bounds.enabled
            or (
                self.config.policy_mode == "full"
                and self.config.full_rl.action_space == "cartesian"
            )
        )

        self.arm_manager = None
        if needs_arm_manager:
            self.motion_manager = MotionManager(self.base_env)
            self.arm_manager = self.motion_manager.body_manager_list[0]

        non_default_weights = {
            cat: w
            for cat, w in self.config.reset_weights.items()
            if cat != "default" and w > 0
        }
        self.reset_sampler = None
        if non_default_weights:
            self.reset_sampler = CachedResetSampler(
                self.base_env, reset_dataset_path or DEFAULT_RESET_DATASET_PATH
            )

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

    def _get_gripper_pos(self, base_obs):
        """Live gripper (eef) position via FK, resynced from the actual
        current joint state each call (rather than trusting `arm_manager`'s
        own internal `arm_joint_pos`, which is meant purely as IK/FK scratch
        state, not a mirror of the live sim)."""
        self.arm_manager.arm_joint_pos = base_obs["joint_pos"][:6].copy()
        self.arm_manager.forward_kinematics()
        return self.arm_manager.current_se3.translation.copy()

    def _compute_dists(self, base_obs):
        """(gripper_pos, gripper-to-object dist, object-to-goal dist) for the
        current state. Used both to initialize the previous-step baseline at
        reset() and to compute this step's progress reward -- see
        RewardTermConfig's docstring for why these are progress (delta)
        rewards, not absolute-proximity ones."""
        gripper_pos = self._get_gripper_pos(base_obs)
        toolbox_pos = self.base_env.unwrapped.data.body("toolbox").xpos
        mat_pos = self.base_env.unwrapped.data.body("mat").xpos
        gripper_to_object_dist = float(np.linalg.norm(gripper_pos - toolbox_pos))
        object_to_goal_dist = float(np.linalg.norm(toolbox_pos - mat_pos))
        return gripper_pos, gripper_to_object_dist, object_to_goal_dist

    def _cartesian_action_to_arm_qpos(self, raw_action, base_obs):
        """Convert a [-1,1]^6 (xyz + rpy) delta into joint targets via IK, one
        damped-least-squares Newton step (`ArmManager.set_command_eef_pose_rel`)
        -- adequate since the deltas are deliberately small, same as every
        other delta-pose control path in this codebase (teleop, rollout).

        Only resyncs `arm_joint_pos` (the IK Newton step's local seed) to the
        true physical joint state each call -- `target_se3` is intentionally
        LEFT ALONE here, since it's meant to persist and accumulate deltas
        across the whole episode (set once in `reset()`), matching how
        `set_command_eef_pose_rel` is used everywhere else in this codebase.
        Resetting target_se3 to "wherever the arm currently is" on every call
        was a real bug caught during testing: any tiny steady-state servo/
        gravity droop between calls got silently accepted as the new target,
        ratcheting the arm downward over many steps even under an all-zero
        action, instead of holding a stable commanded position.
        """
        self.arm_manager.arm_joint_pos = base_obs["joint_pos"][:6].copy()
        self.arm_manager.forward_kinematics()

        pos_delta = raw_action[:3] * self.config.full_rl.cartesian_pos_scale
        rot_delta = raw_action[3:6] * self.config.full_rl.cartesian_rot_scale
        self.arm_manager.set_command_eef_pose_rel(
            np.concatenate([pos_delta, rot_delta])
        )
        return self.arm_manager.arm_joint_pos.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options is not None and "world_idx" in options:
            # Explicit world_idx (used for deterministic eval) always uses
            # the environment's own default reset, bypassing diverse resets.
            self.base_env.unwrapped.modify_world(world_idx=options["world_idx"])
            base_obs, info = self.base_env.reset()
        else:
            base_obs, info = self._sample_reset()

        self.frozen_act.reset()
        self._episode_start_time = self.base_env.unwrapped.get_time()
        self._base_obs = base_obs
        self._info = info
        self._episode_pick_success = False
        self._time_last_close = 0.0

        if self.arm_manager is not None:
            # Baseline for the first step's progress reward -- e.g. a
            # near-goal Stable-Grasp reset should score its first step's
            # dist-reward relative to *this* already-close starting point,
            # not implicitly reward "progress" that was actually just where
            # the episode began.
            _, self._prev_gripper_to_object_dist, self._prev_object_to_goal_dist = (
                self._compute_dists(base_obs)
            )
            # Cartesian-mode delta target: persists and accumulates across
            # the whole episode from here (see _cartesian_action_to_arm_qpos),
            # initialized to the arm's true post-reset pose via the FK that
            # _compute_dists just ran.
            self.arm_manager.target_se3 = self.arm_manager.current_se3.copy()

        return self._make_obs(base_obs), info

    def _sample_reset(self):
        if self.reset_sampler is not None:
            category = choose_reset_category(self.config.reset_weights, self.np_random)
            if category != "default":
                result = self.reset_sampler.sample(category, self.np_random)
                if result is not None:
                    return result
                # Every attempt for the sampled category was invalid --
                # fall through to the default reset below.

        world_idx = int(self.np_random.choice(self.world_idx_list))
        self.base_env.unwrapped.modify_world(world_idx=world_idx)
        return self.base_env.reset()

    def step(self, raw_action):
        raw_action = np.clip(raw_action, -1.0, 1.0)

        pick_success = False
        pap_success = False
        base_obs, info = self._base_obs, self._info
        for _ in range(self.frozen_act.skip):
            if self.config.policy_mode == "full" and self.config.full_rl.action_space == "cartesian":
                # No ACT prior, and no raw joint-space actuator targets either
                # -- the [-1,1] action is a small eef-space delta (xyz + rpy),
                # solved to joint targets via IK, so a randomly-exploring
                # policy moves smoothly instead of commanding huge absolute
                # joint swings every step.
                arm_qpos = self._cartesian_action_to_arm_qpos(raw_action, base_obs)
                gripper_low = self.base_env.action_space.low[6]
                gripper_high = self.base_env.action_space.high[6]
                gripper_ctrl = (
                    gripper_low + (raw_action[6] + 1.0) / 2.0 * (gripper_high - gripper_low)
                )
                combined_action = np.clip(
                    np.concatenate([arm_qpos, [gripper_ctrl]]),
                    self.base_env.action_space.low,
                    self.base_env.action_space.high,
                )
            elif self.config.policy_mode == "full":
                # No ACT prior at all: the policy's [-1,1] output is mapped
                # directly onto the full actuator range, not a small delta
                # -- this is a genuinely different action space from residual
                # mode, not just "residual with the ACT term zeroed out"
                # (that would still be stuck with the tiny residual bound).
                low = self.base_env.action_space.low
                high = self.base_env.action_space.high
                combined_action = low + (raw_action + 1.0) / 2.0 * (high - low)
            else:
                base_action = self.frozen_act.get_base_action(
                    base_obs, info["rgb_images"]
                )
                combined_action = np.clip(
                    base_action + raw_action * self._residual_scale,
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

        gripper_to_object_dist = None
        if self.arm_manager is not None and (
            self.config.reach.enabled
            or self.config.dist.enabled
            or self.config.early_truncation.enabled
            or self.config.workspace_bounds.enabled
        ):
            gripper_pos, gripper_to_object_dist, object_to_goal_dist = self._compute_dists(
                base_obs
            )

            # Progress rewards: weight * (prev_dist - curr_dist), not
            # weight * f(curr_dist) -- see RewardTermConfig's docstring for
            # why (an absolute-proximity reward is farmable by loitering near
            # a good-but-not-successful state; a progress reward pays exactly
            # zero for holding still, no matter how "close" that state is).
            if self.config.reach.enabled:
                reward += self.config.reach.weight * (
                    self._prev_gripper_to_object_dist - gripper_to_object_dist
                )
            if self.config.dist.enabled:
                reward += self.config.dist.weight * (
                    self._prev_object_to_goal_dist - object_to_goal_dist
                )

            self._prev_gripper_to_object_dist = gripper_to_object_dist
            self._prev_object_to_goal_dist = object_to_goal_dist

        elapsed_duration = self.base_env.unwrapped.get_time() - self._episode_start_time
        terminated = pap_success
        truncated = (not pap_success) and (elapsed_duration >= self.max_episode_duration)

        if (
            self.config.early_truncation.enabled
            and not terminated
            and not truncated
        ):
            if gripper_to_object_dist <= self.config.early_truncation.reach_threshold:
                self._time_last_close = elapsed_duration
            if (
                (not self._episode_pick_success)
                and (elapsed_duration - self._time_last_close)
                >= self.config.early_truncation.patience_seconds
            ):
                truncated = True

        out_of_bounds = False
        if self.config.workspace_bounds.enabled and not terminated and not truncated:
            wb = self.config.workspace_bounds
            out_of_bounds = not (
                wb.x_min <= gripper_pos[0] <= wb.x_max
                and wb.y_min <= gripper_pos[1] <= wb.y_max
                and wb.z_min <= gripper_pos[2] <= wb.z_max
            )
            if out_of_bounds:
                truncated = True

        info = {
            **info,
            "success": pap_success,
            "pick_success": self._episode_pick_success,
            "out_of_bounds": out_of_bounds,
        }

        return self._make_obs(base_obs), reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
