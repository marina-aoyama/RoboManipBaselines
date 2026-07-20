# Run this in a separate terminal while training is in progress, e.g.:
#   python3 misc/EvalRolloutSuccessRate.py ManiFlowPolicy MujocoUR5eToolbox \
#     --checkpoint_dir ./checkpoint/ManiFlowPolicy/MujocoUR5eToolbox_world0 \
#     --world_idx_list 0 \
#     --world_idx_repeat_count 5 \
#     --watch

import argparse
import csv
import os
import re
import subprocess
import time

import yaml
from torch.utils.tensorboard import SummaryWriter

CKPT_RE = re.compile(r"^policy_epoch(\d+)\.ckpt$")
REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate success rate of periodic training checkpoints via headless "
            "rollout, and log the result to a CSV file and TensorBoard. "
            "Meant to be run from a separate terminal while training is in progress."
        ),
    )

    parser.add_argument("policy", type=str, help="policy, e.g. ManiFlowPolicy")
    parser.add_argument("env", type=str, help="environment, e.g. MujocoUR5eToolbox")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="training checkpoint directory to watch (same as --checkpoint_dir passed to Train.py)",
    )
    parser.add_argument(
        "--world_idx_list",
        type=int,
        nargs="*",
        default=[0],
        help="list of world indexes to roll out",
    )
    parser.add_argument(
        "--world_idx_repeat_count",
        type=int,
        default=5,
        help="number of rollout episodes per world index (total episodes = "
        "len(world_idx_list) * this)",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="maximum rollout duration per episode [s]",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep polling checkpoint_dir for new checkpoints instead of "
        "evaluating whatever is currently there once and exiting",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=60.0,
        help="seconds between checks for new checkpoints (only used with --watch)",
    )
    parser.add_argument(
        "--rollout_extra_args",
        type=str,
        nargs="*",
        default=[],
        help="extra arguments forwarded verbatim to bin/Rollout.py, e.g. --seed 0",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the on-screen rollout viewer and policy plot instead of running "
        "headless (requires a real display; will not work on a headless machine "
        "even with MUJOCO_GL=osmesa, since offscreen rendering and the on-screen "
        "viewer are different code paths)",
    )

    return parser.parse_args()


class EvalRolloutSuccessRate:
    def __init__(
        self,
        policy,
        env,
        checkpoint_dir,
        world_idx_list,
        world_idx_repeat_count,
        max_duration,
        watch,
        poll_interval,
        rollout_extra_args,
        render,
    ):
        self.policy = policy
        self.env = env
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.world_idx_list = world_idx_list
        self.world_idx_repeat_count = world_idx_repeat_count
        self.max_duration = max_duration
        self.watch = watch
        self.poll_interval = poll_interval
        self.rollout_extra_args = rollout_extra_args
        self.render = render

        self.eval_dir = os.path.join(self.checkpoint_dir, "rollout_eval")
        self.csv_path = os.path.join(self.eval_dir, "success_rate.csv")
        os.makedirs(self.eval_dir, exist_ok=True)

        self.writer = SummaryWriter(self.checkpoint_dir)
        self.evaluated_epochs = self.load_evaluated_epochs()

    def load_evaluated_epochs(self):
        evaluated = set()
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r") as f:
                for row in csv.DictReader(f):
                    evaluated.add(int(row["epoch"]))
        return evaluated

    def find_new_checkpoints(self):
        found = []
        for fname in os.listdir(self.checkpoint_dir):
            match = CKPT_RE.match(fname)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch in self.evaluated_epochs:
                continue
            ckpt_path = os.path.join(self.checkpoint_dir, fname)
            # Skip files still being written by the training process.
            size_before = os.path.getsize(ckpt_path)
            time.sleep(2.0)
            if os.path.getsize(ckpt_path) != size_before:
                continue
            found.append((epoch, ckpt_path))
        return sorted(found)

    def eval_checkpoint(self, epoch, ckpt_path):
        print(
            f"\n===== [{self.__class__.__name__}] Evaluating epoch {epoch}: "
            f"{ckpt_path} =====",
            flush=True,
        )
        result_path = os.path.join(self.eval_dir, f"result_epoch{epoch:0>4}.yaml")

        cmd = (
            [
                "python3",
                "bin/Rollout.py",
                self.policy,
                self.env,
                "--checkpoint",
                ckpt_path,
                "--world_idx_list",
                *[str(idx) for idx in self.world_idx_list],
                "--world_idx_repeat_count",
                str(self.world_idx_repeat_count),
                "--max_duration",
                str(self.max_duration),
                "--result_filename",
                result_path,
                "--auto_exit",
            ]
            + ([] if self.render else ["--no_render", "--no_plot"])
            + self.rollout_extra_args
        )
        # Force CPU-based OSMesa rendering: MuJoCo's default GLFW backend needs
        # an X11 display, which headless machines don't have (the container's
        # MUJOCO_GL=glfw default is broken here, so setdefault isn't enough).
        # Only applies to offscreen camera capture; --render still needs a real
        # display and will fail here regardless, since the on-screen viewer
        # always uses GLFW directly (see WindowViewer in gymnasium's mujoco
        # renderer), independent of MUJOCO_GL.
        env = os.environ.copy()
        if not self.render:
            env["MUJOCO_GL"] = "osmesa"

        try:
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[{self.__class__.__name__}] Rollout failed for epoch {epoch}: {e}",
                flush=True,
            )
            return None

        with open(result_path, "r") as f:
            result = yaml.safe_load(f)
        successes = result.get("success", [])
        if len(successes) == 0:
            print(
                f"[{self.__class__.__name__}] No episodes recorded for epoch {epoch}",
                flush=True,
            )
            return None

        success_rate = sum(successes) / len(successes)
        self.log_result(epoch, success_rate, len(successes))
        return success_rate

    def log_result(self, epoch, success_rate, num_episodes):
        is_new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["epoch", "success_rate", "num_episodes", "timestamp"])
            writer.writerow([epoch, success_rate, num_episodes, time.time()])

        self.writer.add_scalar("success_rate/rollout", success_rate, epoch)
        self.evaluated_epochs.add(epoch)
        print(
            f"[{self.__class__.__name__}] epoch {epoch}: "
            f"success rate {success_rate:.2f} ({num_episodes} episodes)",
            flush=True,
        )

    def run(self):
        while True:
            for epoch, ckpt_path in self.find_new_checkpoints():
                self.eval_checkpoint(epoch, ckpt_path)

            if not self.watch:
                break
            time.sleep(self.poll_interval)


if __name__ == "__main__":
    EvalRolloutSuccessRate(**vars(parse_argument())).run()
