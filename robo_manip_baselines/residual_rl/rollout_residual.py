#!/usr/bin/env python3
"""Evaluate a trained residual SAC policy on top of the frozen ManiFlow base policy.

Example:
    python3 rollout_residual.py \\
        --residual_checkpoint_dir ../checkpoint/ResidualRl/..._ResidualRl_.../ \\
        --world_idx_list 0
"""

import argparse
import os
import pickle

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robo_manip_baselines.residual_rl import ResidualEnv, ResidualRlConfig


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
        "--env_id",
        type=str,
        default=None,
        help="overrides the env_id recorded in residual_meta.pkl",
    )
    parser.add_argument(
        "--maniflow_checkpoint",
        type=str,
        default=None,
        help="overrides the maniflow_checkpoint recorded in residual_meta.pkl",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="overrides the config path recorded in residual_meta.pkl",
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
        help="disable the residual (pure base-policy baseline) for A/B comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(
        os.path.join(args.residual_checkpoint_dir, "residual_meta.pkl"), "rb"
    ) as f:
        meta = pickle.load(f)

    env_id = args.env_id or meta["env_id"]
    maniflow_checkpoint = args.maniflow_checkpoint or meta["maniflow_checkpoint"]
    world_idx_list = (
        args.world_idx_list if args.world_idx_list is not None else meta["world_idx_list"]
    )
    config_path = args.config or meta.get("config_path")
    config = ResidualRlConfig.from_yaml(config_path) if config_path else None

    raw_env = ResidualEnv(
        env_id=env_id,
        config=config,
        maniflow_checkpoint=maniflow_checkpoint,
        world_idx_list=world_idx_list,
        residual_action_scale=meta["residual_action_scale"],
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
