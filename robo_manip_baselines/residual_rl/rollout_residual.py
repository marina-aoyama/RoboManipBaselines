#!/usr/bin/env python3
"""Evaluate a trained residual SAC policy on top of the frozen ACT base policy.

Example:
    python3 rollout_residual.py \\
        --residual_checkpoint_dir ../checkpoint/ResidualRl/..._ResidualRl_.../ \\
        --world_idx_list 0 1 2 3 4 5
"""

import argparse
import os
import pickle

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robo_manip_baselines.residual_rl import ResidualToolboxEnv


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--residual_checkpoint_dir",
        type=str,
        required=True,
        help="directory containing residual_sac.zip / vecnormalize.pkl / residual_meta.pkl",
    )
    parser.add_argument(
        "--act_checkpoint",
        type=str,
        default=None,
        help="overrides the act_checkpoint recorded in residual_meta.pkl",
    )
    parser.add_argument(
        "--world_idx_list",
        type=int,
        nargs="*",
        default=None,
        help="overrides the world_idx_list recorded in residual_meta.pkl",
    )
    parser.add_argument(
        "--num_episodes_per_world",
        type=int,
        default=1,
        help="number of episodes to run per world index",
    )
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument(
        "--no_residual",
        action="store_true",
        help="disable the residual (pure ACT baseline) for A/B comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(
        os.path.join(args.residual_checkpoint_dir, "residual_meta.pkl"), "rb"
    ) as f:
        meta = pickle.load(f)

    act_checkpoint = args.act_checkpoint or meta["act_checkpoint"]
    world_idx_list = (
        args.world_idx_list if args.world_idx_list is not None else meta["world_idx_list"]
    )

    raw_env = ResidualToolboxEnv(
        act_checkpoint=act_checkpoint,
        world_idx_list=world_idx_list,
        residual_action_scale_arm=meta["residual_action_scale_arm"],
        residual_action_scale_gripper=meta["residual_action_scale_gripper"],
        max_episode_duration=meta["max_episode_duration"],
        render_mode=None if args.no_render else "human",
    )

    dummy_vec_env = DummyVecEnv([lambda: raw_env])
    vec_normalize = VecNormalize.load(
        os.path.join(args.residual_checkpoint_dir, "vecnormalize.pkl"), dummy_vec_env
    )
    vec_normalize.training = False

    model = SAC.load(os.path.join(args.residual_checkpoint_dir, "residual_sac"))

    successes = []
    for world_idx in world_idx_list:
        for episode_idx in range(args.num_episodes_per_world):
            obs, info = raw_env.reset(options={"world_idx": world_idx})
            terminated = truncated = False
            while not (terminated or truncated):
                if args.no_residual:
                    action = np.zeros(raw_env.action_space.shape)
                else:
                    norm_obs = vec_normalize.normalize_obs(obs[np.newaxis])[0]
                    action, _ = model.predict(norm_obs, deterministic=True)
                obs, reward, terminated, truncated, info = raw_env.step(action)

            success = bool(info.get("success", False))
            successes.append(success)
            print(
                f"[rollout_residual] world_idx={world_idx} episode={episode_idx} "
                f"-> {'success' if success else 'failure'}"
            )

    success_rate = float(np.mean(successes)) if successes else 0.0
    print(
        f"[rollout_residual] Success rate: {success_rate:.2%} "
        f"({sum(successes)}/{len(successes)})"
    )

    # Note: env.close() is intentionally not called here — MujocoEnvBase's
    # offscreen viewer cleanup raises when no on-screen viewer was created
    # (the same reason RolloutBase.run() leaves its own env.close() commented out).


if __name__ == "__main__":
    main()
