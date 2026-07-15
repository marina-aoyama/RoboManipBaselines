#!/usr/bin/env python3
"""Mine grasp points from existing teleop demonstrations.

For each demo episode, finds the timestep the gripper starts closing (grasp
onset) and reads the recorded `measured_eef_pose` there. Since the object's
pose isn't recorded in the demo data but the object doesn't move until
touched, its pose at that moment is reconstructed by replaying just the
deterministic `modify_world(world_idx)` + `reset()` (no full physics replay
needed). The gripper-relative-to-object transform at each grasp is cached to
`grasp_points.pkl` as a plain (N, 7) array of poses (tx,ty,tz,qw,qx,qy,qz), for
use by `DiverseResetSampler`.

Example:
    python3 mine_grasp_points.py
"""

import argparse
import glob
import os
import pickle

import gymnasium as gym
import numpy as np

import robo_manip_baselines.envs  # noqa: F401
from robo_manip_baselines.common.data.RmbData import RmbData
from robo_manip_baselines.common.utils.MathUtils import (
    get_pose_from_se3,
    get_se3_from_pose,
)


def detect_grasp_idx(gripper_pos, threshold_frac=0.9):
    """Index of the first crossing past `threshold_frac` of the way from open
    to closed. Using the onset of closing (~0.5) captures the gripper while
    it's often still approaching, fingers not yet actually around the object
    -- a fine approach configuration but not a secure grasp, which matters a
    lot for Stable-Grasp resets (nothing but the grip itself holds the object
    up). A later, more-closed threshold captures a configuration much closer
    to an actual secured grasp."""
    gripper_pos = np.asarray(gripper_pos).flatten()
    thresh = gripper_pos.min() + threshold_frac * (gripper_pos.max() - gripper_pos.min())
    return int(np.argmax(gripper_pos > thresh))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "dataset",
                "MujocoUR5eToolbox_Dataset30",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "grasp_points.pkl"),
    )
    args = parser.parse_args()

    demo_paths = sorted(glob.glob(os.path.join(args.dataset_dir, "*.rmb")))
    if len(demo_paths) == 0:
        raise RuntimeError(f"No .rmb demo files found under {args.dataset_dir}")
    print(f"[mine_grasp_points] Found {len(demo_paths)} demo episodes")

    env = gym.make("robo_manip_baselines/MujocoUR5eToolboxEnv-v0", render_mode=None)

    object_pose_by_world_idx = {}
    relative_poses = []

    for demo_path in demo_paths:
        with RmbData(demo_path) as d:
            world_idx = int(d.attrs["world_idx"])
            gripper_pos = d["measured_gripper_joint_pos"][:]
            eef_pose = d["measured_eef_pose"][:]

        grasp_idx = detect_grasp_idx(gripper_pos)
        gripper_pose_at_grasp = eef_pose[grasp_idx]

        if world_idx not in object_pose_by_world_idx:
            env.unwrapped.modify_world(world_idx=world_idx)
            env.reset()
            object_pose_by_world_idx[world_idx] = env.unwrapped.get_body_pose(
                "toolbox"
            ).copy()
        object_pose = object_pose_by_world_idx[world_idx]

        # Gripper pose expressed in the object's frame at the grasp moment.
        se3_object = get_se3_from_pose(object_pose)
        se3_gripper = get_se3_from_pose(gripper_pose_at_grasp)
        se3_rel = se3_object.actInv(se3_gripper)
        relative_poses.append(get_pose_from_se3(se3_rel))

        print(
            f"[mine_grasp_points] {os.path.basename(demo_path)}: world_idx={world_idx} "
            f"grasp_idx={grasp_idx}/{len(gripper_pos)} "
            f"rel_translation={se3_rel.translation}"
        )

    # Note: env.close() is intentionally not called -- MujocoEnvBase's offscreen
    # viewer cleanup raises when no on-screen viewer was created (same reason
    # RolloutBase.run() and rollout_residual.py leave their own env.close() out).

    relative_poses = np.stack(relative_poses, axis=0)  # (N, 7)
    with open(args.output, "wb") as f:
        pickle.dump(relative_poses, f)
    print(f"[mine_grasp_points] Saved {len(relative_poses)} grasp points to {args.output}")

    translations = relative_poses[:, :3]
    print(
        "[mine_grasp_points] relative translation stats: "
        f"mean={translations.mean(axis=0)}, std={translations.std(axis=0)}"
    )


if __name__ == "__main__":
    main()
