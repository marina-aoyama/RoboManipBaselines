import argparse

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from robo_manip_baselines.common import ArmConfig, DataKey, RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Backfill measured_object_pose/measured_goal_pose into existing RMB episodes "
        "by replaying their recorded command_joint_pos through the MuJoCo simulator.",
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="robo_manip_baselines/MujocoUR5eToolboxEnv-v0",
        help="gym env id used to replay episodes",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="max allowed max-abs discrepancy [rad] between replayed and recorded "
        "measured_joint_pos on the arm joints (gripper excluded, see code comment); "
        "if exceeded, the world reconstruction is considered untrustworthy and the "
        "episode is aborted instead of writing bad pose data. The default allows for "
        "the small (~0.01 rad) position-controller settling transient seen at the "
        "start of a real episode, which decays to ~0 within a few steps",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether to overwrite existing value if it exists",
    )

    return parser.parse_args()


class AddObjectPoseToRmbData:
    def __init__(self, path, env, tolerance, overwrite=False):
        self.path = path
        self.env_id = env
        self.tolerance = tolerance
        self.overwrite = overwrite

    def run(self):
        env = gym.make(self.env_id)
        env.reset()
        env.unwrapped.world_random_scale = None

        object_key = DataKey.MEASURED_OBJECT_POSE
        goal_key = DataKey.MEASURED_GOAL_POSE

        rmb_path_list = find_rmb_files(self.path)
        for rmb_path in tqdm(rmb_path_list):
            tqdm.write(f"[{self.__class__.__name__}] Open {rmb_path}")
            with RmbData(rmb_path, mode="r+") as rmb_data:
                if (object_key in rmb_data.keys()) or (goal_key in rmb_data.keys()):
                    if self.overwrite:
                        for key in (object_key, goal_key):
                            if key in rmb_data.keys():
                                del rmb_data.h5file[key]
                    else:
                        raise ValueError(
                            f"[{self.__class__.__name__}] Object/goal pose already "
                            f"exists: {rmb_path} (use --overwrite to replace)"
                        )

                world_idx = int(rmb_data.attrs["world_idx"])
                command_joint_pos_seq = rmb_data[DataKey.COMMAND_JOINT_POS][:]
                recorded_measured_joint_pos_seq = rmb_data[DataKey.MEASURED_JOINT_POS][
                    :
                ]

                object_pose_seq, goal_pose_seq, max_joint_pos_err = (
                    self.replay_episode(
                        env,
                        world_idx,
                        command_joint_pos_seq,
                        recorded_measured_joint_pos_seq,
                    )
                )

                tqdm.write(
                    f"[{self.__class__.__name__}] Max measured_joint_pos discrepancy "
                    f"during replay: {max_joint_pos_err:.6f} rad"
                )
                if max_joint_pos_err > self.tolerance:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Replay diverged from the recorded "
                        f"trajectory by {max_joint_pos_err:.6f} rad (> tolerance "
                        f"{self.tolerance}) for {rmb_path}. The reconstructed world may not "
                        "match the one used during original data collection (e.g. "
                        "world_random_scale was not None); inspect before trusting the "
                        "extracted object/goal pose, or rerun with --tolerance to override."
                    )

                rmb_data.h5file[object_key] = object_pose_seq
                rmb_data.h5file[goal_key] = goal_pose_seq

    def replay_episode(
        self, env, world_idx, command_joint_pos_seq, recorded_measured_joint_pos_seq
    ):
        # Restrict the validation check to arm joints. The gripper's measured_joint_pos
        # and command_joint_pos use different reference conventions (e.g. command 0 rad
        # corresponds to measured ~1.05 rad at fully open) even in freshly-recorded data,
        # so comparing the gripper entry would produce a large, meaningless "error".
        arm_joint_idxes = np.concatenate(
            [
                body_config.arm_joint_idxes
                for body_config in env.unwrapped.body_config_list
                if isinstance(body_config, ArmConfig)
            ]
        )

        env.unwrapped.modify_world(world_idx=world_idx)
        obs, _ = env.reset()

        object_pose_seq = []
        goal_pose_seq = []
        max_joint_pos_err = 0.0

        # Each recorded step i holds the state *before* command_joint_pos_seq[i] is applied
        # (see TeleopBase.run(), which calls record_data() before env.step()).
        for step_idx, command_joint_pos in enumerate(command_joint_pos_seq):
            object_pose_seq.append(env.unwrapped.get_object_pose())
            goal_pose_seq.append(env.unwrapped.get_goal_pose())

            measured_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)
            joint_pos_err = np.max(
                np.abs(
                    measured_joint_pos[arm_joint_idxes]
                    - recorded_measured_joint_pos_seq[step_idx][arm_joint_idxes]
                )
            )
            max_joint_pos_err = max(max_joint_pos_err, joint_pos_err)

            obs, _, _, _, _ = env.step(command_joint_pos)

        return np.array(object_pose_seq), np.array(goal_pose_seq), max_joint_pos_err


if __name__ == "__main__":
    AddObjectPoseToRmbData(**vars(parse_argument())).run()
