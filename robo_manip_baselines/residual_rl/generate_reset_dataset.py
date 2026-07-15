#!/usr/bin/env python3
"""Offline-generate and validate diverse reset states, following the
emergent-dexterity paper's actual approach: sample+validate resets ahead of
time (via `DiverseResetSampler`'s IK/settle/rejection logic), cache only the
ones that pass, and sample uniformly from that cache during training --
rather than regenerating/validating live at every reset.

Run this once (or whenever the sampling logic / grasp points change) before
training with a config that uses diverse resets.

Example:
    python3 generate_reset_dataset.py --num_states_per_category 200
    python3 show_diverse_resets.py --dataset_path reset_dataset.pkl  # inspect it
"""

import argparse
import os
import pickle

import gymnasium as gym
import numpy as np

import robo_manip_baselines.envs  # noqa: F401
from diverse_reset import DiverseResetSampler

CATEGORIES = ("near_object", "stable_grasp", "near_goal")


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--grasp_points_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "grasp_points.pkl"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "reset_dataset.pkl"),
    )
    parser.add_argument(
        "--num_states_per_category",
        type=int,
        default=200,
        help="target number of validated states to collect per category",
    )
    parser.add_argument(
        "--max_total_attempts_per_category",
        type=int,
        default=2000,
        help="safety cap on total try_once() calls per category, in case the "
        "target count can't be reached",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make("robo_manip_baselines/MujocoUR5eToolboxEnv-v0", render_mode=None)
    env.unwrapped.target_task = "pick_and_place"

    np_random = np.random.default_rng(args.seed)
    sampler = DiverseResetSampler(env, args.grasp_points_path)

    dataset = {}
    for category in CATEGORIES:
        print(f"[generate_reset_dataset] Generating '{category}'...")
        qpos_list = []
        n_attempts = 0
        while (
            len(qpos_list) < args.num_states_per_category
            and n_attempts < args.max_total_attempts_per_category
        ):
            env.unwrapped.modify_world(world_idx=0)
            result = sampler.try_once(category, np_random)
            n_attempts += 1
            if result is not None:
                qpos_list.append(env.unwrapped.data.qpos.copy())

        accept_rate = len(qpos_list) / n_attempts if n_attempts else 0.0
        print(
            f"[generate_reset_dataset] '{category}': {len(qpos_list)}/{n_attempts} "
            f"attempts accepted ({accept_rate:.1%})"
        )
        if len(qpos_list) < args.num_states_per_category:
            print(
                f"[generate_reset_dataset] WARNING: only reached "
                f"{len(qpos_list)}/{args.num_states_per_category} for "
                f"'{category}' within the {args.max_total_attempts_per_category} "
                f"attempt budget"
            )
        dataset[category] = (
            np.stack(qpos_list, axis=0)
            if qpos_list
            else np.zeros((0, env.unwrapped.model.nq))
        )

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    print(f"[generate_reset_dataset] Saved dataset to {args.output}")
    for category, qpos_arr in dataset.items():
        print(f"  {category}: {qpos_arr.shape[0]} states")


if __name__ == "__main__":
    main()
