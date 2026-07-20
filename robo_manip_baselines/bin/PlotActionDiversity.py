"""Plot per-dimension action distributions over rollout time, from a *.npz
file saved by bin/CheckActionDiversity.py --diversity_output.

Usage:
    python bin/PlotActionDiversity.py <diversity_npz> [--skip N] \
        [--n_steps_per_chunk N] [--action_dim_labels LABEL ...] \
        [--output PATH.png]

Example:
    python3 bin/PlotActionDiversity.py ./diversity/hand_eef_world0.npz \
        --skip 2 \
        --n_steps_per_chunk 8
"""

import argparse
import os

import matplotlib.pylab as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Plot per-dimension action distributions over rollout time, from a "
            "*.npz file saved by bin/CheckActionDiversity.py --diversity_output. "
            "For each replan step, all repeated policy samples are drawn as faint "
            "lines (plus their mean), stitched across the whole rollout, so you "
            "can see at which point in time the sampled actions fan out."
        ),
    )
    parser.add_argument(
        "diversity_npz",
        type=str,
        help="path to the *.npz file produced by CheckActionDiversity.py --diversity_output",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="environment steps between consecutive predicted action-chunk "
        "timesteps (matches the --skip used for the rollout, printed as "
        "'skip: N' at the start of CheckActionDiversity.py's output); only "
        "affects the x-axis scaling",
    )
    parser.add_argument(
        "--n_steps_per_chunk",
        type=int,
        default=None,
        help="only plot the first N predicted timesteps of each chunk (e.g. "
        "matching --n_action_steps, the portion that is actually executed "
        "before replanning, so chunks tile the timeline without overlapping); "
        "default: plot the full predicted horizon, including chunks' overlap "
        "with the next replan",
    )
    parser.add_argument(
        "--action_dim_labels",
        type=str,
        nargs="*",
        default=None,
        help="optional labels for each action dimension (default: action[0], action[1], ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output image path (default: <diversity_npz>_actions.png)",
    )
    args = parser.parse_args()

    data = np.load(args.diversity_npz)
    actions = data["actions"]  # [num_infer_steps, num_samples, horizon, action_dim]
    infer_t = data["rollout_time_idx"]  # [num_infer_steps]

    num_infer_steps, num_samples, horizon, action_dim = actions.shape
    n_steps = args.n_steps_per_chunk or horizon
    n_steps = min(n_steps, horizon)

    if args.action_dim_labels is not None:
        assert len(args.action_dim_labels) == action_dim
        labels = args.action_dim_labels
    else:
        labels = [f"action[{d}]" for d in range(action_dim)]

    fig, axes = plt.subplots(
        action_dim, 1, figsize=(14, 2.2 * action_dim), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for infer_idx in range(num_infer_steps):
        t = infer_t[infer_idx] + args.skip * np.arange(n_steps)
        chunk = actions[infer_idx, :, :n_steps, :]  # [num_samples, n_steps, action_dim]
        chunk_mean = chunk.mean(axis=0)  # [n_steps, action_dim]
        for d in range(action_dim):
            for sample_idx in range(num_samples):
                axes[d].plot(
                    t, chunk[sample_idx, :, d], color="tab:blue", alpha=0.25, linewidth=1
                )
            axes[d].plot(t, chunk_mean[:, d], color="tab:red", linewidth=1.2)
        # Mark the start of each replan
        for d in range(action_dim):
            axes[d].axvline(infer_t[infer_idx], color="gray", alpha=0.15, linewidth=0.8)

    for d in range(action_dim):
        axes[d].set_ylabel(labels[d], fontsize=10)
        axes[d].grid(alpha=0.3)
    axes[-1].set_xlabel("rollout step")
    fig.suptitle(
        f"{os.path.basename(args.diversity_npz)}  "
        f"(blue = {num_samples} samples per replan, red = mean, gray = replan boundary)"
    )
    fig.tight_layout()

    output = args.output or os.path.splitext(args.diversity_npz)[0] + "_actions.png"
    fig.savefig(output, dpi=150)
    print(f"[PlotActionDiversity] Saved plot to {output}")


if __name__ == "__main__":
    main()
