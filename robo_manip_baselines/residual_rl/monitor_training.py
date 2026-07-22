#!/usr/bin/env python3
"""Periodically polls one or more SAC training runs' tensorboard logs and
prints a status line when something is worth a human look (sustained
critic-loss growth, entropy-coefficient collapse, success-rate regression,
or accelerating actor-loss magnitude), plus an occasional heartbeat so
"no news" doesn't look identical to "not running". Read-only, CPU-only --
negligible overhead on the training processes themselves.

Not a replacement for actually reading the curves -- a cheap tripwire.
"""

import argparse
import glob
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["rollout/success_rate", "train/ent_coef", "train/actor_loss", "train/critic_loss"]


def load_scalars(checkpoint_dir):
    paths = glob.glob(f"{checkpoint_dir}/sac_1/events.out.tfevents.*")
    if not paths:
        return None
    ea = EventAccumulator(paths[0])
    ea.Reload()
    out = {}
    for tag in TAGS:
        if tag in ea.Tags()["scalars"]:
            out[tag] = ea.Scalars(tag)
    return out


def check_run(name, checkpoint_dir, state, now):
    scalars = load_scalars(checkpoint_dir)
    if not scalars or "rollout/success_rate" not in scalars:
        return None

    step = scalars["rollout/success_rate"][-1].step
    if state.get("last_step") == step:
        return None  # nothing new since last poll
    state["last_step"] = step

    success = scalars["rollout/success_rate"][-1].value
    ent_coef = scalars["train/ent_coef"][-1].value if "train/ent_coef" in scalars else None
    critic_losses = [e.value for e in scalars["train/critic_loss"][-4:]] if "train/critic_loss" in scalars else []
    actor_losses = [e.value for e in scalars["train/actor_loss"][-4:]] if "train/actor_loss" in scalars else []

    flags = []
    if len(critic_losses) >= 3:
        recent = critic_losses[-3:]
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)) and recent[-1] > 200:
            flags.append(f"critic_loss rising 3+ points in a row, now {recent[-1]:.1f}")
    if len(actor_losses) >= 4 and all(
        actor_losses[i] > actor_losses[i + 1] for i in range(len(actor_losses) - 1)
    ):
        # actor_loss growing more negative every one of the last 4 points --
        # accelerating, not just noisy.
        flags.append(f"actor_loss magnitude accelerating, now {actor_losses[-1]:.1f}")
    if ent_coef is not None and ent_coef < 0.01:
        flags.append(f"ent_coef collapsed to {ent_coef:.4f}")

    peak = state.get("peak_success", success)
    state["peak_success"] = max(peak, success)
    if success < state["peak_success"] - 0.25:
        flags.append(f"success_rate dropped from peak {state['peak_success']:.2f} to {success:.2f}")

    summary = (
        f"[{name}] step={step} success_rate={success:.3f} "
        f"ent_coef={ent_coef:.3f} critic_loss={critic_losses[-1] if critic_losses else float('nan'):.1f} "
        f"actor_loss={actor_losses[-1] if actor_losses else float('nan'):.1f}"
    )

    is_flagged = bool(flags)
    heartbeat_due = now - state.get("last_print", 0) >= state["heartbeat_interval"]
    if is_flagged or heartbeat_due:
        state["last_print"] = now
        lines = [summary]
        for f in flags:
            lines.append(f"  FLAG: {f}")
        return "\n".join(lines)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=checkpoint_dir pairs")
    parser.add_argument("--poll_interval", type=float, default=60.0, help="how often to check for new data [s]")
    parser.add_argument(
        "--heartbeat_interval", type=float, default=900.0, help="max seconds between prints even if nothing flagged"
    )
    args = parser.parse_args()

    runs = []
    for item in args.runs:
        name, checkpoint_dir = item.split("=", 1)
        runs.append((name, checkpoint_dir, {"heartbeat_interval": args.heartbeat_interval}))

    while True:
        now = time.time()
        for name, checkpoint_dir, state in runs:
            line = check_run(name, checkpoint_dir, state, now)
            if line:
                print(line, flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
