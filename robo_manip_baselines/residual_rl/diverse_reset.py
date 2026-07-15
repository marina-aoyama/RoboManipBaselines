import pickle

import mujoco
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MotionManager
from robo_manip_baselines.common.utils.MathUtils import get_se3_from_pose

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_JOINT_NAMES = [
    "right_driver_joint",
    "right_coupler_joint",
    "right_spring_link_joint",
    "right_follower_joint",
    "left_driver_joint",
    "left_coupler_joint",
    "left_spring_link_joint",
    "left_follower_joint",
]

# From envs/assets/mujoco/envs/ur5e/env_ur5e_toolbox.xml: table surface height,
# no explicit reachable-workspace constant exists so ranges below are sampled
# near the toolbox/mat region and validated via IK convergence + settle checks.
TABLE_SURFACE_Z = 0.815


def _average_poses(poses):
    """Average N poses (tx,ty,tz,qw,qx,qy,qz) -- naive mean+renormalize for
    the quaternion, a fine approximation when the rotations are all close
    together (true here, per the tight clustering of the mined grasp points).
    Guards against antipodal sign flips (q and -q are the same rotation) by
    flipping any quaternion that points the "wrong way" relative to the first
    before averaging."""
    translations = poses[:, :3]
    quats = poses[:, 3:7].copy()
    ref = quats[0]
    flip = (quats @ ref) < 0
    quats[flip] *= -1
    mean_quat = quats.mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    return np.concatenate([translations.mean(axis=0), mean_quat])


def choose_reset_category(reset_weights, np_random):
    categories = list(reset_weights.keys())
    weights = np.array([reset_weights[c] for c in categories], dtype=float)
    weights = weights / weights.sum()
    return np_random.choice(categories, p=weights)


class DiverseResetSampler:
    """Samples Near-Object / Stable-Grasp / Near-Goal reset states for
    `ResidualToolboxEnv`. Each candidate is written directly into the sim's
    qpos -- including the gripper's own 8 coupled joints, snapped to a
    pre-calibrated "closed around the object" shape (see `_calibrate_
    closed_gripper_qpos`) rather than commanded to close dynamically after
    the object is already placed, which otherwise races against gravity for
    airborne placements -- then verified stable with a handful of real,
    un-pinned physics steps and validity-checked before being accepted,
    retrying with a new sample on failure.
    """

    def __init__(
        self,
        base_env,
        grasp_points_path,
        arm_manager=None,
        max_attempts=5,
        ik_max_iters=100,
        ik_tol=1e-4,
        settle_steps=15,
        calibration_max_steps=40,
        gripper_settle_qvel_tol=0.05,
        object_xy_range=0.08,
        near_goal_xy_range=0.05,
    ):
        self.base_env = base_env
        with open(grasp_points_path, "rb") as f:
            grasp_points = pickle.load(f)  # (N, 7) poses
        # The mined grasp points are tightly clustered (same grasp technique
        # across demos, just human teleoperation noise) rather than genuinely
        # different grasp strategies -- so instead of sampling one of the N
        # per reset (which sometimes picks a marginal/less-secure individual
        # demo's grasp), use a single averaged canonical grasp pose. This
        # keeps the diversity that matters (object position, via IK) while
        # dropping the diversity that was mostly just causing unreliable
        # grasps for Stable-Grasp resets specifically.
        self.canonical_grasp_pose = _average_poses(grasp_points)

        # Reuse a shared ArmManager if the caller already has one (e.g. for
        # the dense reach reward's FK), otherwise build our own for IK only.
        if arm_manager is None:
            self.motion_manager = MotionManager(base_env)
            self.arm_manager = self.motion_manager.body_manager_list[0]
        else:
            self.arm_manager = arm_manager

        self.max_attempts = max_attempts
        self.ik_max_iters = ik_max_iters
        self.ik_tol = ik_tol
        self.settle_steps = settle_steps
        self.calibration_max_steps = calibration_max_steps
        self.gripper_settle_qvel_tol = gripper_settle_qvel_tol
        self.object_xy_range = object_xy_range
        self.near_goal_xy_range = near_goal_xy_range

        self._arm_qpos_addrs = [self._joint_qpos_addr(n) for n in ARM_JOINT_NAMES]
        self._gripper_qpos_addrs = [self._joint_qpos_addr(n) for n in GRIPPER_JOINT_NAMES]
        self._toolbox_qpos_addr = self._joint_qpos_addr("toolbox_freejoint")
        self._gripper_driver_qvel_addr = self._joint_dof_addr("right_driver_joint")
        # Orientation is never randomized (only x,y) -- keep it exactly as the
        # original demo/spawn condition (init_qpos is still pristine here,
        # since modify_world() itself never touches orientation either).
        self._original_toolbox_quat = base_env.unwrapped.init_qpos[
            self._toolbox_qpos_addr + 3 : self._toolbox_qpos_addr + 7
        ].copy()

        # Diagnostics for external inspection (e.g. show_diverse_resets.py) --
        # not used by the RL-facing API. `last_sample_info` is only set on
        # success; `last_attempt_info` is set after every attempt, pass or fail.
        self.last_sample_info = None
        self.last_attempt_info = None

        # Precompute once: what does the gripper's own 8-dof joint
        # configuration look like when actually closed around the object at
        # the canonical grasp pose? Calibrated with the object resting on the
        # table (real support, no time pressure), then reused directly for
        # every future closed-gripper reset instead of re-simulating a
        # dynamic close (which otherwise races against gravity when the
        # object has no support, e.g. Stable-Grasp resets).
        self.closed_gripper_qpos = self._calibrate_closed_gripper_qpos()

    def _joint_qpos_addr(self, joint_name):
        model = self.base_env.unwrapped.model
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(model.jnt_qposadr[joint_id])

    def _joint_dof_addr(self, joint_name):
        model = self.base_env.unwrapped.model
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(model.jnt_dofadr[joint_id])

    def sample(self, category, np_random):
        """Attempt `category` reset up to `max_attempts` times.

        `np_random` is taken fresh on each call (not stored) so that
        re-seeding the environment after construction (`reset(seed=...)`,
        which replaces gymnasium's `env.np_random` object) can't leave this
        sampler holding a stale generator reference.

        Returns (obs, info) on success, or None if every attempt was invalid
        (caller should fall back to the environment's default reset). Retries
        happen silently/back-to-back inside this call -- if you want to
        observe each individual attempt (e.g. for visualization), call
        `try_once` yourself in a loop instead."""
        assert category in ("near_object", "stable_grasp", "near_goal")

        for _ in range(self.max_attempts):
            result = self.try_once(category, np_random)
            if result is not None:
                return result

        return None

    def try_once(self, category, np_random):
        """A single reset attempt, no retries. Returns (obs, info) on success,
        None on failure. Always updates `last_attempt_info` -- except on an
        IK failure, where the sim state never changed (no candidate qpos was
        ever written), so `last_attempt_info` is set to a minimal dict with
        `ik_failed=True` instead of a full post-settle snapshot."""
        assert category in ("near_object", "stable_grasp", "near_goal")

        object_pose, gripper_closed = self._sample_object_pose_and_gripper(
            category, np_random
        )
        grasp_pose = self._sample_grasp_point(np_random)
        gripper_target_se3 = get_se3_from_pose(object_pose) * get_se3_from_pose(
            grasp_pose
        )

        if not self._solve_ik(gripper_target_se3):
            self.last_attempt_info = {
                "ik_failed": True,
                "object_pose": object_pose.copy(),
                "gripper_closed": gripper_closed,
            }
            return None

        arm_qpos = self.arm_manager.arm_joint_pos.copy()
        return self._apply_and_settle(arm_qpos, object_pose, gripper_closed)

    def _sample_grasp_point(self, np_random):
        pose = self.canonical_grasp_pose.copy()
        # Small random translational offset (paper: "grasp points with a
        # small random offset"); rotation left as the canonical average for
        # simplicity/reliability.
        pose[:3] += np_random.uniform(-0.01, 0.01, size=3)
        return pose

    def _sample_object_pose_and_gripper(self, category, np_random):
        original_toolbox_xy = self.base_env.unwrapped.original_toolbox_pos[:2]
        mat_xy = self.base_env.unwrapped.data.body("mat").xpos[:2].copy()

        if category == "near_object":
            xy = original_toolbox_xy + np_random.uniform(
                -self.object_xy_range, self.object_xy_range, size=2
            )
            z = TABLE_SURFACE_Z
            gripper_closed = bool(np_random.integers(2))
        elif category == "stable_grasp":
            # Span the full corridor from pickup to goal (not just around the
            # pickup point) -- otherwise "elevated + holding + near goal,
            # about to descend" has zero coverage: near_goal only covers low
            # height/about-to-release, and this category previously only
            # covered elevated-near-pickup. Assumes original_toolbox and mat
            # share x (true for this task's geometry); jitters laterally
            # around their midpoint x rather than hardcoding 0 so it degrades
            # gracefully if that assumption ever stops holding.
            x_center = (original_toolbox_xy[0] + mat_xy[0]) / 2.0
            y_min = min(original_toolbox_xy[1], mat_xy[1]) - self.object_xy_range
            y_max = max(original_toolbox_xy[1], mat_xy[1]) + self.near_goal_xy_range
            xy = np.array(
                [
                    x_center + np_random.uniform(-self.object_xy_range, self.object_xy_range),
                    np_random.uniform(y_min, y_max),
                ]
            )
            z = TABLE_SURFACE_Z + np_random.uniform(0.05, 0.25)
            gripper_closed = True
        elif category == "near_goal":
            xy = mat_xy + np_random.uniform(
                -self.near_goal_xy_range, self.near_goal_xy_range, size=2
            )
            # 5-12cm above the surface: well clear of the pap_success
            # threshold (z < mat_z + 0.5cm; a 0-5cm range let ~20% of samples
            # settle into an already-"solved" state -- see _is_valid's
            # explicit post-settle check, which remains the authoritative
            # guard against that regardless of this range) -- and also well
            # clear of the "final instant before release" height that real
            # teleop demos likely only ever passed through momentarily, never
            # dwelled at. Lower/tighter on average than stable_grasp's
            # typical 5-25cm (which, since being widened to span the full
            # pickup<->goal corridor, now also covers "elevated near goal" --
            # near_goal's distinguishing feature is its tighter xy focus).
            z = TABLE_SURFACE_Z + np_random.uniform(0.05, 0.12)
            # Always closed: "near goal" means still carrying the object
            # toward the goal, about to place it -- not already released.
            gripper_closed = True
        else:
            raise ValueError(f"Unknown reset category: {category}")

        object_pose = np.concatenate([[xy[0], xy[1], z], self._original_toolbox_quat])
        return object_pose, gripper_closed

    def _solve_ik(self, target_se3):
        self.arm_manager.reset()
        self.arm_manager.target_se3 = target_se3
        for _ in range(self.ik_max_iters):
            self.arm_manager.inverse_kinematics()
            error_se3 = self.arm_manager.current_se3.actInv(target_se3)
            if np.linalg.norm(pin.log(error_se3).vector) < self.ik_tol:
                return True
        return False

    def _calibrate_closed_gripper_qpos(self):
        object_pose = np.concatenate(
            [
                self.base_env.unwrapped.original_toolbox_pos,
                self._original_toolbox_quat,
            ]
        )
        target_se3 = get_se3_from_pose(object_pose) * get_se3_from_pose(
            self.canonical_grasp_pose
        )
        if not self._solve_ik(target_se3):
            raise RuntimeError(
                "DiverseResetSampler: IK failed to solve for the canonical "
                "grasp pose during closed-gripper calibration -- this should "
                "always converge since it's the same geometry the grasp "
                "points were mined from."
            )
        arm_qpos = self.arm_manager.arm_joint_pos.copy()

        init_qpos = self.base_env.unwrapped.init_qpos
        for addr, val in zip(self._arm_qpos_addrs, arm_qpos):
            init_qpos[addr] = val
        init_qpos[self._toolbox_qpos_addr : self._toolbox_qpos_addr + 7] = object_pose
        self.base_env.reset()

        data = self.base_env.unwrapped.data
        hold_action = np.concatenate([arm_qpos, [self.base_env.action_space.high[6]]])
        for _ in range(self.calibration_max_steps):
            self.base_env.step(hold_action)
            if abs(data.qvel[self._gripper_driver_qvel_addr]) < self.gripper_settle_qvel_tol:
                break

        return np.array([data.qpos[addr] for addr in self._gripper_qpos_addrs])

    def _apply_and_settle(self, arm_qpos, object_pose, gripper_closed):
        init_qpos = self.base_env.unwrapped.init_qpos
        for addr, val in zip(self._arm_qpos_addrs, arm_qpos):
            init_qpos[addr] = val
        init_qpos[self._toolbox_qpos_addr : self._toolbox_qpos_addr + 7] = object_pose
        if gripper_closed:
            for addr, val in zip(self._gripper_qpos_addrs, self.closed_gripper_qpos):
                init_qpos[addr] = val
        # else: gripper qpos left at its existing (open) init_qpos value.

        obs, info = self.base_env.reset()

        action_space = self.base_env.action_space
        gripper_ctrl = (
            action_space.high[6] if gripper_closed else action_space.low[6]
        )
        hold_action = np.concatenate([arm_qpos, [gripper_ctrl]])

        # The gripper/object are already placed in their final (closed-grip
        # or open-resting) shape from frame 0 -- no dynamic closing transient
        # to race against gravity through. These steps are purely a stability
        # verification window under real, un-pinned physics.
        for _ in range(self.settle_steps):
            obs, _, _, _, info = self.base_env.step(hold_action)

        valid = self._is_valid(object_pose, gripper_closed)

        # Always recorded, pass or fail, so callers (e.g. show_diverse_resets.py)
        # can inspect what a *rejected* attempt actually looked like -- not
        # just successful ones.
        self.last_attempt_info = {
            "ik_failed": False,
            "arm_qpos": arm_qpos.copy(),
            "gripper_ctrl": gripper_ctrl,
            "object_pose": object_pose.copy(),
            "gripper_closed": gripper_closed,
            "valid": valid,
        }

        if not valid:
            return None

        self.last_sample_info = self.last_attempt_info
        return obs, info

    def _is_valid(self, target_object_pose, gripper_closed):
        data = self.base_env.unwrapped.data
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            return False
        if np.max(np.abs(data.qvel)) > 20.0:
            return False

        toolbox_pos = data.body("toolbox").xpos
        if toolbox_pos[2] < TABLE_SURFACE_Z - 0.1:
            return False  # fell through/off the table
        if gripper_closed and np.linalg.norm(
            toolbox_pos - target_object_pose[:3]
        ) > 0.03:
            return False  # object slipped/fell instead of actually being held

        # Reject any reset that already satisfies the task's own pap_success
        # condition (MujocoUR5eToolboxEnv._get_reward()'s thresholds,
        # duplicated here) -- a state that's already "solved" gives a free
        # reward=1.0 on the very first step regardless of the agent's action,
        # teaching nothing. Checked post-settle, not from the commanded
        # target pose, since Near-Goal placements reliably settle/drop 1.5-3cm
        # even when the commanded height had margin above the threshold.
        mat_pos = data.body("mat").xpos
        pap_xy_thre, pap_z_thre = 0.03, mat_pos[2] + 0.005
        already_succeeded = (
            np.max(np.abs(toolbox_pos[:2] - mat_pos[:2])) < pap_xy_thre
        ) and (toolbox_pos[2] < pap_z_thre)
        if already_succeeded:
            return False

        return True


class CachedResetSampler:
    """Samples from a pre-validated, offline-generated dataset of reset
    states (see `generate_reset_dataset.py`) -- the emergent-dexterity
    paper's actual approach: validate resets once ahead of time (via
    `DiverseResetSampler`, used offline by the generation script), cache the
    results, then just look one up at training reset time instead of
    re-running IK/settling/rejection live on every single reset. Also makes
    exactly what training draws from inspectable ahead of time, rather than
    only visible reset-by-reset while training runs.
    """

    def __init__(self, base_env, dataset_path):
        self.base_env = base_env
        with open(dataset_path, "rb") as f:
            self.dataset = pickle.load(f)  # {category: (N, nq) qpos array}
        for category, qpos_arr in self.dataset.items():
            if qpos_arr.shape[0] == 0:
                raise ValueError(
                    f"CachedResetSampler: dataset at {dataset_path} has zero "
                    f"states for category '{category}' -- re-run "
                    f"generate_reset_dataset.py"
                )

    def sample(self, category, np_random):
        qpos_arr = self.dataset[category]
        idx = np_random.integers(len(qpos_arr))
        self.base_env.unwrapped.init_qpos[:] = qpos_arr[idx]
        return self.base_env.reset()
