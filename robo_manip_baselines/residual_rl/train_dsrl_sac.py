#!/usr/bin/env python3
"""Train a DSRL (noise-space steering) SAC policy on top of a frozen
ManiFlow base policy. See DsrlEnv's docstring for the method; requires the
forked ManiFlow's predict_action(..., noise=...) override.

Example:
    python3 train_dsrl_sac.py \\
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

from robo_manip_baselines.residual_rl import DsrlEnv, ResidualRlConfig
from robo_manip_baselines.residual_rl.layer_norm_policy import LayerNormSACPolicy


class SuccessRateCallback(BaseCallback):
    """Same rolling-success-rate logging + best-checkpoint saving as
    train_residual_sac.py's callback -- see there for why (SAC can drift
    away from a good intermediate policy, so only saving the final one
    isn't enough)."""

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
                self.model.save(os.path.join(self.checkpoint_dir, "dsrl_sac_best"))
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
        "--noise_mode",
        type=str,
        default="shared",
        choices=["shared", "full"],
        help="'shared': one action_dim-sized noise vector broadcast across the whole "
        "chunk (smaller action space, try this first). 'full': independent noise per "
        "chunk timestep, matching the DSRL paper -- higher-dimensional, harder to "
        "explore. See DsrlEnv's docstring.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=2.0,
        help="multiplies the SAC policy's [-1,1] output before feeding it to ManiFlow "
        "as x0 -- the flow model was trained assuming x0 ~ N(0,1), so this should "
        "cover a few standard deviations (default 2.0), not be tiny like a residual scale",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="overrides the checkpoint's saved chunk length -- this IS the DSRL "
        "paper's query_frequency (how many env actions get executed per noise "
        "decision). Shorter means more frequent steering opportunities at the cost "
        "of more flow-ODE inference calls. Default: use the checkpoint's own value.",
    )
    parser.add_argument(
        "--include_object_pose",
        action="store_true",
        help="append the manipulated object's absolute world-frame pose to the observation "
        "(requires the env to implement get_object_pose(), e.g. MujocoUR5eInsertEnv.get_object_pose). "
        "Off by default -- opt-in experiment testing whether the persistent 'same edge every "
        "time' failure is a partial-observability problem (policy can't tell the peg has "
        "shifted in the gripper) rather than an unconverged-policy problem.",
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
        help="skip the pre-training reward sanity check (see "
        "FrozenPolicyEnvBase.sanity_check_reward) -- not recommended, it's cheap and "
        "catches a dead/misconfigured reward signal before burning a training run on it",
    )
    parser.add_argument(
        "--ent_coef",
        type=str,
        default="auto",
        help="SAC entropy coefficient. Inherited from train_residual_sac.py's tuning for "
        "the *action-space residual* entropy-collapse problem -- NOT verified to apply to "
        "DSRL's noise-space action, and nakamoto/dsrl_pi0's own reference config exposes no "
        "equivalent flags (likely just using their SAC implementation's defaults). Treat "
        "this and --ent_coef_init/--target_entropy as an untested carryover, not a match.",
    )
    parser.add_argument("--ent_coef_init", type=float, default=2.0)
    parser.add_argument("--target_entropy", type=str, default="-3.5")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="discount factor. SB3 SAC's own default is 0.99; nakamoto/dsrl_pi0's "
        "run_libero.sh uses 0.999 (one RL step = one full chunk here, same as there, "
        "so a longer effective horizon needs less discounting per step).",
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=500,
        help="env steps of pure random exploration before gradient updates begin. "
        "nakamoto/dsrl_pi0's run_libero.sh uses start_online_updates=500 (vs. "
        "train_residual_sac.py's 2000, tuned for a different problem -- matching theirs here).",
    )
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=20,
        help="gradient updates performed per --train_freq env steps collected. "
        "nakamoto/dsrl_pi0's run_libero.sh uses multi_grad_step=20 (vs. "
        "train_residual_sac.py's 1) -- a much higher update-to-data ratio; matching theirs here.",
    )
    parser.add_argument("--norm_reward", action="store_true")
    parser.add_argument(
        "--no_layer_norm",
        action="store_true",
        help="use SB3's plain MlpPolicy instead of LayerNormSACPolicy. LayerNorm is on by "
        "default here specifically because --gradient_steps=20 is a high update-to-data "
        "ratio, which plain MLP critics are prone to Q-value/critic-loss divergence at "
        "(observed in early DSRL runs: critic_loss grew ~580x over 6k steps with no sign "
        "of plateauing). See layer_norm_policy.py.",
    )
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="output directory (default: checkpoint/DsrlRl/<env>_DsrlRl_<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="robo_manip_baselines_dsrl")
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

    if args.checkpoint_dir is None:
        env_dirname = args.env_id.split("/")[-1].replace("-v0", "")
        checkpoint_dirname = "{}_DsrlRl_{:%Y%m%d_%H%M%S}".format(
            env_dirname, datetime.datetime.now()
        )
        args.checkpoint_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "checkpoint", "DsrlRl", checkpoint_dirname)
        )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"[train_dsrl_sac] checkpoint_dir: {args.checkpoint_dir}")

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
        return DsrlEnv(
            env_id=args.env_id,
            config=config,
            maniflow_checkpoint=args.maniflow_checkpoint,
            world_idx_list=args.world_idx_list,
            noise_mode=args.noise_mode,
            noise_scale=args.noise_scale,
            n_action_steps=args.n_action_steps,
            include_object_pose=args.include_object_pose,
            max_episode_duration=args.max_episode_duration,
            render_mode="human" if args.render else None,
        )

    if not args.skip_reward_check:
        print("[train_dsrl_sac] Running pre-training reward sanity check...")
        check_env = make_env()
        any_nonzero, _ = check_env.sanity_check_reward(
            num_episodes=min(3, len(args.world_idx_list) * 2)
        )
        # check_env.close() intentionally not called -- see
        # train_residual_sac.py's identical comment for why.
        if not any_nonzero:
            raise RuntimeError(
                "[train_dsrl_sac] Reward sanity check found reward=0.0 the entire "
                "time -- refusing to start training against what looks like a dead "
                "reward signal. Pass --skip_reward_check to override."
            )

    vec_env = DummyVecEnv([lambda: Monitor(make_env())])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=args.norm_reward)

    try:
        ent_coef = float(args.ent_coef)
    except ValueError:
        ent_coef = f"auto_{args.ent_coef_init}"

    try:
        target_entropy = float(args.target_entropy)
    except ValueError:
        target_entropy = args.target_entropy

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

    model.save(os.path.join(args.checkpoint_dir, "dsrl_sac"))
    vec_env.save(os.path.join(args.checkpoint_dir, "vecnormalize.pkl"))

    dsrl_meta = {
        "env_id": args.env_id,
        "maniflow_checkpoint": args.maniflow_checkpoint,
        "config_path": args.config,
        "noise_mode": args.noise_mode,
        "noise_scale": args.noise_scale,
        "n_action_steps": args.n_action_steps,
        "include_object_pose": args.include_object_pose,
        "world_idx_list": args.world_idx_list,
        "max_episode_duration": args.max_episode_duration,
    }
    with open(os.path.join(args.checkpoint_dir, "dsrl_meta.pkl"), "wb") as f:
        pickle.dump(dsrl_meta, f)

    print(f"[train_dsrl_sac] Saved DSRL checkpoint to {args.checkpoint_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
