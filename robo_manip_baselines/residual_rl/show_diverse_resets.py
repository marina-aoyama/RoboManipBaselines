#!/usr/bin/env python3
"""Visualize reset states before spending training time on them.

Two modes:
  - Live mode (default): calls `DiverseResetSampler.try_once()` directly, one
    attempt at a time, showing/holding every attempt (accepted or rejected).
    Useful for tuning the sampling logic itself before generating a dataset.
  - Dataset mode (--dataset_path): loads an already-generated dataset (see
    `generate_reset_dataset.py`) and shows actual entries from it -- this is
    exactly what training will draw from, so it's the one to use to know
    ahead of time what reset states a training run will actually see.

Renders live and keeps holding/stepping each shown state for a few seconds
(not just the initial snapshot), so instability that only shows up a bit
later is visible too.

Examples:
    python3 show_diverse_resets.py --category near_object --num_samples 5
    python3 show_diverse_resets.py --dataset_path reset_dataset.pkl --num_samples 5
"""

import argparse
import os
import pickle
import time

import gymnasium as gym
import mujoco
import numpy as np

import robo_manip_baselines.envs  # noqa: F401
from diverse_reset import DiverseResetSampler

CATEGORIES = ("near_object", "stable_grasp", "near_goal")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "if given, inspect an already-generated dataset (see "
            "generate_reset_dataset.py) instead of live-sampling -- this is "
            "exactly what training will draw from"
        ),
    )
    parser.add_argument(
        "--grasp_points_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "grasp_points.pkl"),
        help="(live mode only)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(CATEGORIES),
        help="show only this category (default: cycle through all 3)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="samples per category"
    )
    parser.add_argument(
        "--hold_seconds",
        type=float,
        default=3.0,
        help=(
            "how long to keep holding/stepping each shown reset, so "
            "instability that only appears a bit later is visible"
        ),
    )
    parser.add_argument(
        "--instability_qvel_threshold",
        type=float,
        default=5.0,
        help="print a warning whenever max |qvel| exceeds this during the hold period",
    )
    parser.add_argument(
        "--fail_hold_seconds",
        type=float,
        default=None,
        help=(
            "(live mode only) how long to hold/show a *rejected* attempt "
            "instead of silently skipping to the next one -- that's what "
            "actually failed, useful for seeing why. Defaults to --hold_seconds."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.fail_hold_seconds is None:
        args.fail_hold_seconds = args.hold_seconds
    return args


def hold_and_watch(env, data, hold_action, hold_seconds, instability_qvel_threshold):
    """Keep stepping `hold_action` for `hold_seconds`, printing a warning if
    velocity spikes (renders live throughout since render_mode='human')."""
    t_end = time.time() + hold_seconds
    warned = False
    while time.time() < t_end:
        env.step(hold_action)
        qvel_max = float(np.max(np.abs(data.qvel)))
        if qvel_max > instability_qvel_threshold:
            toolbox_pos = data.body("toolbox").xpos
            print(
                f"    !! instability during hold: toolbox_pos={toolbox_pos} "
                f"qvel_max={qvel_max:.2f}"
            )
            warned = True
    if not warned:
        print("    (stable throughout hold period)")


def run_live(env, data, args, categories):
    np_random = np.random.default_rng(args.seed)
    sampler = DiverseResetSampler(env, args.grasp_points_path)

    for category in categories:
        print(f"\n=== {category} (live) ===")
        n_success = 0
        for i in range(args.num_samples):
            env.unwrapped.modify_world(world_idx=0)
            accepted = False

            # Drive attempts one at a time (instead of delegating retries to
            # sampler.sample(), which does up to max_attempts internally with
            # no visibility) so every individual attempt gets shown and held,
            # not just flashed through invisibly during the search.
            for attempt_idx in range(1, sampler.max_attempts + 1):
                result = sampler.try_once(category, np_random)
                info = sampler.last_attempt_info

                if info.get("ik_failed"):
                    print(
                        f"  sample {i} attempt {attempt_idx}/{sampler.max_attempts}: "
                        f"IK failed to converge for object_xyz={info['object_pose'][:3]} "
                        f"(no sim state change, not held)"
                    )
                    continue

                hold_action = np.concatenate(
                    [info["arm_qpos"], [info["gripper_ctrl"]]]
                )
                status = "ACCEPTED" if result is not None else "rejected"
                print(
                    f"  sample {i} attempt {attempt_idx}/{sampler.max_attempts}: {status} "
                    f"object_xyz={info['object_pose'][:3]} "
                    f"gripper_closed={info['gripper_closed']} "
                    f"post-settle qvel_max={np.max(np.abs(data.qvel)):.3f}"
                )
                hold_seconds = (
                    args.hold_seconds if result is not None else args.fail_hold_seconds
                )
                hold_and_watch(
                    env, data, hold_action, hold_seconds, args.instability_qvel_threshold
                )

                if result is not None:
                    accepted = True
                    n_success += 1
                    break

            if not accepted:
                print(f"  sample {i}: FAILED after {sampler.max_attempts} attempts")

        print(f"  -> {n_success}/{args.num_samples} accepted")


def run_dataset(env, data, args, categories):
    with open(args.dataset_path, "rb") as f:
        dataset = pickle.load(f)

    np_random = np.random.default_rng(args.seed)
    model = env.unwrapped.model
    driver_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint"
    )
    driver_qpos_addr = int(model.jnt_qposadr[driver_joint_id])

    for category in categories:
        qpos_arr = dataset.get(category)
        if qpos_arr is None or len(qpos_arr) == 0:
            print(f"\n=== {category} (dataset) === -- no states in dataset, skipping")
            continue

        print(f"\n=== {category} (dataset, {len(qpos_arr)} states total) ===")
        n_shown = min(args.num_samples, len(qpos_arr))
        indices = np_random.choice(len(qpos_arr), size=n_shown, replace=False)
        for i, idx in enumerate(indices):
            env.unwrapped.init_qpos[:] = qpos_arr[idx]
            env.reset()

            # Hold the arm exactly where the cached qpos put it (ctrl targets
            # matching the actuated joints' current qpos) and gripper at its
            # own current position, purely to keep it visually in place --
            # this is what training would resume acting from, not a replay.
            arm_qpos = data.qpos[:6].copy()
            # Driver joint range is [0, 0.8] (0=open, 0.8=closed, see
            # diverse_reset.py); a cached "closed" entry settles near the
            # calibrated ~0.79, "open" stays near 0 -- the midpoint cleanly
            # separates them. Only used cosmetically here, to send a ctrl
            # target that holds the gripper roughly where it already is.
            looks_closed = data.qpos[driver_qpos_addr] > 0.4
            gripper_ctrl = (
                env.action_space.high[6] if looks_closed else env.action_space.low[6]
            )
            hold_action = np.concatenate([arm_qpos, [gripper_ctrl]])

            toolbox_pos = data.body("toolbox").xpos.copy()
            print(
                f"  dataset entry {idx}: object_xyz={toolbox_pos} "
                f"post-load qvel_max={np.max(np.abs(data.qvel)):.3f}"
            )
            hold_and_watch(
                env, data, hold_action, args.hold_seconds, args.instability_qvel_threshold
            )

        print(f"  -> shown {n_shown}/{len(qpos_arr)} states")


def main():
    args = parse_args()
    categories = [args.category] if args.category else list(CATEGORIES)

    env = gym.make("robo_manip_baselines/MujocoUR5eToolboxEnv-v0", render_mode="human")
    env.unwrapped.target_task = "pick_and_place"
    data = env.unwrapped.data

    if args.dataset_path:
        run_dataset(env, data, args, categories)
    else:
        run_live(env, data, args, categories)

    print("\nDone.")


if __name__ == "__main__":
    main()
