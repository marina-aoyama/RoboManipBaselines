#!/usr/bin/env python3
"""Train a residual SAC policy on top of a frozen ACT checkpoint.

Example:
    python3 train_residual_sac.py \\
        --act_checkpoint ../checkpoint/Act/MujocoUR5eToolbox_Dataset30_Act_.../policy_last.ckpt \\
        --world_idx_list 0 1 2 3 4 5 \\
        --total_timesteps 100000
"""

import argparse
import datetime
import os
import pickle
from collections import deque

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robo_manip_baselines.residual_rl import ResidualRlConfig, ResidualToolboxEnv


class SuccessRateCallback(BaseCallback):
    """Logs `pick`-only and full `pick_and_place` success rates separately
    (as rolling means over the last `window_size` completed episodes), so we
    can tell whether failures are dominated by missed grasps or by placement
    once the box is picked up.

    Also saves a "best" checkpoint (model + VecNormalize stats) whenever the
    rolling `pap_success_rate` reaches a new high, once at least
    `min_episodes_for_best` episodes have completed. SAC's off-policy training
    can drift away from a good intermediate policy (e.g. if the entropy
    coefficient collapses before it stabilizes) -- without this, only the
    *final* (possibly worse) policy would ever get saved.
    """

    def __init__(
        self, checkpoint_dir, window_size=100, min_episodes_for_best=20, verbose=0
    ):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.pick_success_window = deque(maxlen=window_size)
        self.pap_success_window = deque(maxlen=window_size)
        self.min_episodes_for_best = min_episodes_for_best
        self.best_pap_success_rate = -1.0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.pick_success_window.append(bool(info.get("pick_success", False)))
                self.pap_success_window.append(bool(info.get("success", False)))

        if len(self.pap_success_window) > 0:
            pick_rate = float(np.mean(self.pick_success_window))
            pap_rate = float(np.mean(self.pap_success_window))
            self.logger.record("rollout/pick_success_rate", pick_rate)
            self.logger.record("rollout/pap_success_rate", pap_rate)

            if (
                len(self.pap_success_window) >= self.min_episodes_for_best
                and pap_rate > self.best_pap_success_rate
            ):
                self.best_pap_success_rate = pap_rate
                self.model.save(
                    os.path.join(self.checkpoint_dir, "residual_sac_best")
                )
                self.training_env.save(
                    os.path.join(self.checkpoint_dir, "vecnormalize_best.pkl")
                )
                if self.verbose > 0:
                    print(
                        f"[SuccessRateCallback] New best pap_success_rate={pap_rate:.3f} "
                        f"at step {self.num_timesteps}, saved checkpoint"
                    )
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--act_checkpoint", type=str, required=True, help="frozen ACT checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "baseline.yaml"),
        help=(
            "path to a ResidualRlConfig YAML (see configs/baseline.yaml for the "
            "original sparse-reward/single-reset setup, configs/omnireset_dense.yaml "
            "for diverse resets + dense reach/dist reward + early truncation)"
        ),
    )
    parser.add_argument(
        "--world_idx_list",
        type=int,
        nargs="*",
        default=list(range(6)),
        help="world indexes to sample from each episode reset",
    )
    parser.add_argument(
        "--residual_action_scale_arm",
        type=float,
        default=0.05,
        help="max residual magnitude for each arm joint [rad]",
    )
    parser.add_argument(
        "--residual_action_scale_gripper",
        type=float,
        default=10.0,
        help="max residual magnitude for the gripper joint [device units, 0-255 scale]",
    )
    parser.add_argument(
        "--max_episode_duration",
        type=float,
        default=30.0,
        help="episode timeout [s] (mirrors RolloutBase's --max_duration default)",
    )
    parser.add_argument(
        "--ent_coef",
        type=str,
        default="auto",
        help=(
            "SAC entropy coefficient. 'auto' automatically tunes it (recommended -- "
            "a fixed value never lets the policy consolidate/converge, which is "
            "worse). Pass a fixed float (e.g. '0.2') only for controlled experiments."
        ),
    )
    parser.add_argument(
        "--ent_coef_init",
        type=float,
        default=2.0,
        help=(
            "Initial value of the auto-tuned entropy coefficient (only used when "
            "--ent_coef=auto; SB3's own default is 1.0). A higher starting value "
            "doesn't change the decay rate, just buys more steps before the same "
            "exploration floor is reached -- one of several independent levers "
            "against premature entropy collapse (see also --learning_starts, "
            "--train_freq, --norm_reward)."
        ),
    )
    parser.add_argument(
        "--target_entropy",
        type=str,
        default="-3.5",
        help=(
            "Target entropy for SAC's auto ent_coef tuning (only used when "
            "--ent_coef=auto). SB3's own default ('auto' here resolves to "
            "-action_dim = -7) drove ent_coef toward ~0 within ~10-15k steps on "
            "this sparse-reward task, cutting off exploration before the policy "
            "stabilized. A less negative value (e.g. -3.5) makes the auto-tuner "
            "settle at a higher exploration floor instead of decaying toward 0."
        ),
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=2000,
        help=(
            "env steps of pure random exploration before any gradient update "
            "(including on the entropy coefficient) begins. SB3's own default "
            "(100) starts optimizing almost immediately, giving the replay buffer "
            "very little diverse experience before entropy starts getting pulled "
            "down; raising this delays the whole collapse clock and seeds a more "
            "diverse buffer first."
        ),
    )
    parser.add_argument(
        "--train_freq",
        type=int,
        default=4,
        help=(
            "collect this many env steps between each round of gradient updates "
            "(SB3 default: 1, i.e. one update per env step). A larger value means "
            "fewer entropy-coefficient (and actor/critic) gradient steps per env "
            "step collected, directly slowing entropy decay measured on the "
            "env-step timeline."
        ),
    )
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=1,
        help="gradient updates performed per --train_freq env steps collected",
    )
    parser.add_argument(
        "--norm_reward",
        action="store_true",
        help=(
            "normalize reward via VecNormalize (default: off, raw reward). With "
            "dense reach/dist terms at weight=5.0, raw Q-targets are large, which "
            "pushes the actor toward exploitation (and thus lower realized policy "
            "entropy) fast; normalizing keeps Q-value magnitude from dominating "
            "the actor loss early. Safe to toggle independently of rollout -- "
            "rollout only uses the saved obs normalization stats, never reward."
        ),
    )
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="output directory (default: checkpoint/ResidualRl/<act_ckpt_name>_ResidualRl_<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_wandb", action="store_true", help="whether to log to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="robo_manip_baselines_residual_rl"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="whether to render the simulation live during training (slows training down)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.act_checkpoint = os.path.abspath(args.act_checkpoint)
    args.config = os.path.abspath(args.config)
    config = ResidualRlConfig.from_yaml(args.config)

    if args.checkpoint_dir is None:
        act_ckpt_dirname = os.path.basename(os.path.dirname(args.act_checkpoint))
        checkpoint_dirname = "{}_ResidualRl_{:%Y%m%d_%H%M%S}".format(
            act_ckpt_dirname, datetime.datetime.now()
        )
        args.checkpoint_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "checkpoint",
                "ResidualRl",
                checkpoint_dirname,
            )
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"[train_residual_sac] checkpoint_dir: {args.checkpoint_dir}")

    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=os.path.basename(args.checkpoint_dir),
            config=vars(args),
            dir=args.checkpoint_dir,
            sync_tensorboard=True,
        )

    def make_env():
        env = ResidualToolboxEnv(
            act_checkpoint=args.act_checkpoint,
            config=config,
            world_idx_list=args.world_idx_list,
            residual_action_scale_arm=args.residual_action_scale_arm,
            residual_action_scale_gripper=args.residual_action_scale_gripper,
            max_episode_duration=args.max_episode_duration,
            render_mode="human" if args.render else None,
        )
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=args.norm_reward)

    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = f"auto_{args.ent_coef_init}"  # e.g. "auto_2.0"

    try:
        target_entropy = float(args.target_entropy)
    except ValueError:
        target_entropy = args.target_entropy  # e.g. "auto"

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tensorboard_log=args.checkpoint_dir,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="sac",
        callback=SuccessRateCallback(checkpoint_dir=args.checkpoint_dir, verbose=1),
    )

    model.save(os.path.join(args.checkpoint_dir, "residual_sac"))
    vec_env.save(os.path.join(args.checkpoint_dir, "vecnormalize.pkl"))

    residual_meta = {
        "act_checkpoint": args.act_checkpoint,
        "config_path": args.config,
        "residual_action_scale_arm": args.residual_action_scale_arm,
        "residual_action_scale_gripper": args.residual_action_scale_gripper,
        "world_idx_list": args.world_idx_list,
        "max_episode_duration": args.max_episode_duration,
    }
    with open(os.path.join(args.checkpoint_dir, "residual_meta.pkl"), "wb") as f:
        pickle.dump(residual_meta, f)

    print(f"[train_residual_sac] Saved residual RL checkpoint to {args.checkpoint_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
