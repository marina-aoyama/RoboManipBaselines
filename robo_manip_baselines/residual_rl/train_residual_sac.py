#!/usr/bin/env python3
"""Train a residual SAC policy on top of a frozen ManiFlow base policy.

Example:
    python3 train_residual_sac.py \\
        --env_id robo_manip_baselines/MujocoUR5eInsertEnv-v0 \\
        --maniflow_checkpoint ../checkpoint/ManiFlowPolicy/front_hand_deltaeef_world0_insert/policy_last.ckpt \\
        --world_idx_list 0 \\
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

from robo_manip_baselines.residual_rl import ResidualEnv, ResidualRlConfig
from robo_manip_baselines.residual_rl.layer_norm_policy import LayerNormSACPolicy


class SuccessRateCallback(BaseCallback):
    """Logs the rolling success rate (mean over the last `window_size`
    completed episodes), and saves a "best" checkpoint (model + VecNormalize
    stats) whenever it reaches a new high, once at least
    `min_episodes_for_best` episodes have completed. SAC's off-policy
    training can drift away from a good intermediate policy (e.g. if the
    entropy coefficient collapses before it stabilizes) -- without this,
    only the *final* (possibly worse) policy would ever get saved.
    """

    def __init__(
        self, checkpoint_dir, window_size=100, min_episodes_for_best=20, verbose=0
    ):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.success_window = deque(maxlen=window_size)
        self.min_episodes_for_best = min_episodes_for_best
        self.best_success_rate = -1.0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.success_window.append(bool(info.get("success", False)))

        if len(self.success_window) > 0:
            success_rate = float(np.mean(self.success_window))
            self.logger.record("rollout/success_rate", success_rate)

            if (
                len(self.success_window) >= self.min_episodes_for_best
                and success_rate > self.best_success_rate
            ):
                self.best_success_rate = success_rate
                self.model.save(
                    os.path.join(self.checkpoint_dir, "residual_sac_best")
                )
                self.training_env.save(
                    os.path.join(self.checkpoint_dir, "vecnormalize_best.pkl")
                )
                if self.verbose > 0:
                    print(
                        f"[SuccessRateCallback] New best success_rate={success_rate:.3f} "
                        f"at step {self.num_timesteps}, saved checkpoint"
                    )
        return True


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="gymnasium env id, e.g. robo_manip_baselines/MujocoUR5eInsertEnv-v0",
    )
    parser.add_argument(
        "--maniflow_checkpoint", type=str, required=True, help="frozen ManiFlow checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "insert_basic.yaml"),
        help="path to a ResidualRlConfig YAML (see configs/insert_basic.yaml)",
    )
    parser.add_argument(
        "--world_idx_list",
        type=int,
        nargs="*",
        default=[0],
        help="world indexes to sample from each episode reset",
    )
    parser.add_argument(
        "--residual_scale_pos",
        type=float,
        default=0.01,
        help="max residual magnitude for eef position deltas [m]",
    )
    parser.add_argument(
        "--residual_scale_rot",
        type=float,
        default=0.05,
        help="max residual magnitude for eef rotation deltas [rad]",
    )
    parser.add_argument(
        "--residual_scale_gripper",
        type=float,
        default=10.0,
        help="max residual magnitude for the gripper command [device units]",
    )
    parser.add_argument(
        "--residual_scale_joint",
        type=float,
        default=0.05,
        help="max residual magnitude for raw joint-position actions [rad] "
        "(only relevant if the checkpoint's action_keys are joint-space)",
    )
    parser.add_argument(
        "--max_episode_duration",
        type=float,
        default=30.0,
        help="episode timeout [s] (mirrors RolloutBase's --max_duration default)",
    )
    parser.add_argument(
        "--skip_reward_check",
        action="store_true",
        help="skip the pre-training reward sanity check (see ResidualEnv.sanity_check_reward) "
        "-- not recommended, it's cheap and catches a dead/misconfigured reward signal "
        "before burning a training run on it",
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
            "-action_dim) drove ent_coef toward ~0 within ~10-15k steps on a "
            "sparse-reward task in earlier experiments, cutting off exploration "
            "before the policy stabilized. A less negative value (e.g. -3.5) makes "
            "the auto-tuner settle at a higher exploration floor instead of "
            "decaying toward 0."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="discount factor. SB3 SAC's own default is 0.99; with ~150 RL-steps per "
        "episode here (skip=3 raw steps each), 0.99^150 ~= 0.22 heavily discounts a "
        "success reward that far out, while 0.999^150 ~= 0.86 doesn't -- matches "
        "train_dsrl_sac.py's value (from nakamoto/dsrl_pi0's run_libero.sh), though "
        "arguably even more needed here given residual RL's finer step granularity.",
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
            "normalize reward via VecNormalize (default: off, raw reward). Safe to "
            "toggle independently of rollout -- rollout only uses the saved obs "
            "normalization stats, never reward."
        ),
    )
    parser.add_argument(
        "--no_layer_norm",
        action="store_true",
        help="use SB3's plain MlpPolicy instead of LayerNormSACPolicy. LayerNorm is on by "
        "default for consistency with train_dsrl_sac.py, though the motivating symptom "
        "(critic-loss divergence at high update-to-data ratio) is a weaker concern here at "
        "the current --gradient_steps=1 (low UTD) than it is for DSRL's --gradient_steps=20. "
        "See layer_norm_policy.py.",
    )
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="output directory (default: checkpoint/ResidualRl/<env>_ResidualRl_<timestamp>)",
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
    args.maniflow_checkpoint = os.path.abspath(args.maniflow_checkpoint)
    args.config = os.path.abspath(args.config)
    config = ResidualRlConfig.from_yaml(args.config)

    residual_action_scale = {
        "pos": args.residual_scale_pos,
        "rot": args.residual_scale_rot,
        "gripper": args.residual_scale_gripper,
        "joint": args.residual_scale_joint,
    }

    if args.checkpoint_dir is None:
        env_dirname = args.env_id.split("/")[-1].replace("-v0", "")
        checkpoint_dirname = "{}_ResidualRl_{:%Y%m%d_%H%M%S}".format(
            env_dirname, datetime.datetime.now()
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
        return ResidualEnv(
            env_id=args.env_id,
            config=config,
            maniflow_checkpoint=args.maniflow_checkpoint,
            world_idx_list=args.world_idx_list,
            residual_action_scale=residual_action_scale,
            max_episode_duration=args.max_episode_duration,
            render_mode="human" if args.render else None,
        )

    if not args.skip_reward_check:
        print("[train_residual_sac] Running pre-training reward sanity check...")
        check_env = make_env()
        any_nonzero, _ = check_env.sanity_check_reward(
            num_episodes=min(3, len(args.world_idx_list) * 2)
        )
        # check_env.close() is intentionally not called — MujocoEnvBase's
        # viewer cleanup raises on teardown regardless of render_mode (the
        # same reason RolloutBase.run() and rollout_residual.py both skip
        # their own env.close() too). The process exiting reclaims
        # everything anyway.
        if not any_nonzero:
            raise RuntimeError(
                "[train_residual_sac] Reward sanity check found reward=0.0 the entire "
                "time -- refusing to start training against what looks like a dead "
                "reward signal. Pass --skip_reward_check to override."
            )

    vec_env = DummyVecEnv([lambda: Monitor(make_env())])
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
        "MlpPolicy" if args.no_layer_norm else LayerNormSACPolicy,
        vec_env,
        verbose=1,
        seed=args.seed,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        gamma=args.gamma,
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
        "env_id": args.env_id,
        "maniflow_checkpoint": args.maniflow_checkpoint,
        "config_path": args.config,
        "residual_action_scale": residual_action_scale,
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
