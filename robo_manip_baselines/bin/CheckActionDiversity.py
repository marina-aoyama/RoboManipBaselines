"""Check whether a trained policy produces diverse actions for the same
observation.

Usage:
    python bin/CheckActionDiversity.py <policy> <env> [Rollout.py args...] \
        [--num_diversity_samples N] [--diversity_output PATH.npz] \
        [--diversity_image_dir DIR]

Example:
    python3 bin/CheckActionDiversity.py ManiFlowPolicy MujocoUR5eToolbox \
    --checkpoint ./checkpoint/ManiFlowPolicy/hand_world0_eef/policy_last.ckpt \
    --n_action_steps 8 \
    --world_idx_list 0 \
    --world_idx_repeat_count 1 \
    --auto_exit \
    --max_duration 30 \
    --num_diversity_samples 10 \
    --diversity_output ./diversity/hand_eef_world0.npz \
    --auto_exit \
    --target_task pick_and_place


This accepts (almost) the same arguments as bin/Rollout.py, since <policy>
and <env> select the same choices and any remaining arguments are forwarded
to the same underlying Rollout argument parser (e.g. --checkpoint, --world_idx,
--seed, etc.). Run with --help after choosing <policy>/<env> to see the full
list of forwarded arguments.
"""

import argparse
import importlib
import importlib.util
import os
import sys

import cv2
import numpy as np
import yaml


class CheckActionDiversityMain:
    """Check whether a trained policy produces diverse actions for the same
    observation, before investing in noise-steering methods (e.g. DSRL) or
    adapters.

    This accepts (almost) the same arguments as bin/Rollout.py: it runs the
    exact same rollout loop (same env resets, same policy, same plotting/
    auto_exit behavior), but at every policy inference step it repeatedly
    samples the policy on the identical observation and reports the spread
    across samples. Only one of the samples is used to actually drive the
    rollout, so the executed behavior is unaffected.
    """

    operation_parent_module_str = "robo_manip_baselines.envs.operation"
    policy_parent_module_str = "robo_manip_baselines.policy"
    policy_choices = [
        "Mlp",
        "Sarnn",
        "Act",
        "MtAct",
        "DiffusionPolicy",
        "DiffusionPolicy3d",
        "FlowPolicy",
        "ManiFlowPolicy",
        "Gr00t",
        "Pi0",
    ]

    def __init__(self):
        self.setup_args()

    def setup_args(self):
        env_utils_spec = importlib.util.spec_from_file_location(
            "EnvUtils",
            os.path.join(os.path.dirname(__file__), "..", "common/utils/EnvUtils.py"),
        )
        env_utils_module = importlib.util.module_from_spec(env_utils_spec)
        env_utils_spec.loader.exec_module(env_utils_module)

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=(
                "This is a meta argument parser for checking action diversity, "
                "switching between different policies and environments (mirrors "
                "bin/Rollout.py). The actual arguments are handled by another "
                "internal argument parser."
            ),
            fromfile_prefix_chars="@",
            add_help=False,
        )
        parser.add_argument(
            "policy",
            type=str,
            nargs="?",
            default=None,
            choices=self.policy_choices,
            help="policy",
        )
        parser.add_argument(
            "env",
            type=str,
            help="environment",
            nargs="?",
            default=None,
            choices=env_utils_module.get_env_names(
                operation_parent_module_str=self.operation_parent_module_str
            ),
        )
        parser.add_argument("--config", type=str, help="configuration file")
        parser.add_argument(
            "--num_diversity_samples",
            type=int,
            default=10,
            help="number of times to repeatedly sample the policy for the same "
            "observation at each inference step",
        )
        parser.add_argument(
            "--diversity_output",
            type=str,
            default=None,
            help="path (*.npz) to save the raw sampled actions for offline "
            "analysis (default: do not save)",
        )
        parser.add_argument(
            "--diversity_image_dir",
            type=str,
            default=None,
            help="directory to save the observation image(s) seen at each "
            "inference step, so task phase can be correlated with diversity "
            "(default: '<diversity_output stem>_images' if --diversity_output "
            "is set, otherwise images are not saved)",
        )
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.policy is None or self.args.env is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]

    def run(self):
        if "Isaac" in self.args.env:
            from isaacgym import (
                gymapi,  # noqa: F401
                gymtorch,  # noqa: F401
                gymutil,  # noqa: F401
            )

        # This includes pytorch import, so it must be later than isaac import
        import torch

        from robo_manip_baselines.common import camel_to_snake, remove_prefix

        operation_module = importlib.import_module(
            f"{self.operation_parent_module_str}.Operation{self.args.env}"
        )
        OperationEnvClass = getattr(operation_module, f"Operation{self.args.env}")

        policy_module = importlib.import_module(
            f"{self.policy_parent_module_str}.{camel_to_snake(self.args.policy)}"
        )
        RolloutPolicyClass = getattr(policy_module, f"Rollout{self.args.policy}")

        # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
        class Rollout(OperationEnvClass, RolloutPolicyClass):
            @property
            def policy_name(self):
                return remove_prefix(RolloutPolicyClass.__name__, "Rollout")

        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, "r") as f:
                config = yaml.safe_load(f)

        rollout = Rollout(**config)

        # Resolve where to save the observation image(s) seen at each
        # inference step (if at all).
        image_dir = self.args.diversity_image_dir
        if image_dir is None and self.args.diversity_output is not None:
            image_dir = os.path.splitext(os.path.abspath(self.args.diversity_output))[
                0
            ] + "_images"
        if image_dir is not None:
            os.makedirs(image_dir, exist_ok=True)

        # Wrap the policy's predict_action so that every inference call is
        # repeated `num_diversity_samples` times on the identical observation.
        # Only the first sample is returned, so the rollout itself proceeds
        # exactly as it would without this wrapper.
        records = []
        num_samples = self.args.num_diversity_samples
        original_predict_action = rollout.policy.predict_action

        def predict_action_diverse(input_data):
            raw_outputs = [
                original_predict_action(input_data) for _ in range(num_samples)
            ]
            actions = np.stack(
                [
                    raw_output["action"][0].detach().cpu().numpy()
                    for raw_output in raw_outputs
                ],
                axis=0,
            )  # [num_samples, horizon, action_dim]

            per_step_std = actions.std(axis=0)  # [horizon, action_dim]
            print(
                f"[CheckActionDiversity] world_idx={rollout.data_manager.world_idx} "
                f"episode_idx={rollout.data_manager.episode_idx} "
                f"t={rollout.rollout_time_idx} | "
                f"action std over {num_samples} samples "
                f"(mean/max over horizon & action dims): "
                f"{per_step_std.mean():.4f} / {per_step_std.max():.4f}"
            )

            image_paths = {}
            if image_dir is not None:
                for camera_name in rollout.camera_names:
                    image = rollout.info["rgb_images"][camera_name]
                    image_path = os.path.join(
                        image_dir,
                        f"step{len(records):03d}_t{rollout.rollout_time_idx:04d}_"
                        f"w{rollout.data_manager.world_idx}_"
                        f"ep{rollout.data_manager.episode_idx}_{camera_name}.png",
                    )
                    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    image_paths[camera_name] = image_path

            records.append(
                {
                    "world_idx": rollout.data_manager.world_idx,
                    "episode_idx": rollout.data_manager.episode_idx,
                    "rollout_time_idx": rollout.rollout_time_idx,
                    "actions": actions,
                    "image_paths": image_paths,
                }
            )

            return raw_outputs[0]

        rollout.policy.predict_action = predict_action_diverse

        rollout.run()

        self.summarize(records)
        if self.args.diversity_output is not None:
            self.save(records)
        if image_dir is not None:
            self.save_filmstrip(records, image_dir)

    @staticmethod
    def summarize(records):
        if len(records) == 0:
            print("[CheckActionDiversity] No inference steps were recorded.")
            return

        all_std = np.concatenate([r["actions"].std(axis=0).ravel() for r in records])
        num_samples = records[0]["actions"].shape[0]
        print(
            f"\n[CheckActionDiversity] Summary over {len(records)} inference steps "
            f"({num_samples} samples each)\n"
            f"  - action std across samples | mean: {all_std.mean():.4f}, "
            f"median: {np.median(all_std):.4f}, max: {all_std.max():.4f}, "
            f"min: {all_std.min():.4f}\n"
            "  - Compare this against the per-dimension range/std of actions in "
            "the training demos: values that are small relative to that range "
            "suggest the policy has collapsed to a near-deterministic map for "
            "these observations (little room for noise-based steering such as "
            "DSRL); larger values indicate exploitable diversity."
        )

    def save(self, records):
        print(
            f"[CheckActionDiversity] Save raw diversity samples: {self.args.diversity_output}"
        )
        output_dir = os.path.dirname(os.path.abspath(self.args.diversity_output))
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            self.args.diversity_output,
            world_idx=np.array([r["world_idx"] for r in records]),
            episode_idx=np.array([r["episode_idx"] for r in records]),
            rollout_time_idx=np.array([r["rollout_time_idx"] for r in records]),
            actions=np.stack([r["actions"] for r in records], axis=0),
        )

    @staticmethod
    def save_filmstrip(records, image_dir, thumb_width=160):
        records_with_images = [r for r in records if r["image_paths"]]
        if len(records_with_images) == 0:
            return

        camera_names = list(records_with_images[0]["image_paths"].keys())
        frames = []
        for record in records_with_images:
            camera_thumbs = []
            for camera_name in camera_names:
                image = cv2.imread(record["image_paths"][camera_name])
                height = int(image.shape[0] * thumb_width / image.shape[1])
                thumb = cv2.resize(image, (thumb_width, height))
                cv2.putText(
                    thumb,
                    camera_name,
                    (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
                camera_thumbs.append(thumb)
            frame = cv2.vconcat(camera_thumbs)
            cv2.putText(
                frame,
                f"t={record['rollout_time_idx']}",
                (4, frame.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
            frames.append(frame)

        filmstrip_path = os.path.join(image_dir, "filmstrip.png")
        cv2.imwrite(filmstrip_path, cv2.hconcat(frames))
        print(
            f"[CheckActionDiversity] Save filmstrip of observations: {filmstrip_path}"
        )


if __name__ == "__main__":
    main = CheckActionDiversityMain()
    main.run()
