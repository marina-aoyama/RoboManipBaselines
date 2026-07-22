import dataclasses

import yaml

POLICY_MODES = ("residual", "full")
FULL_RL_ACTION_SPACES = ("joint", "cartesian")


@dataclasses.dataclass
class EarlyTruncationConfig:
    """Truncates an episode after `patience_seconds` with no improvement in
    the episode's best-so-far reward. Deliberately reward-based (not
    distance-based) so it stays generic across envs -- unlike gripper-to-
    object distance, reward is available from every env's own
    `_get_reward()` with no task-specific plumbing required. For a sparse
    binary-success task (e.g. Insert) this is close to a no-op (reward only
    ever "improves" at the moment of success, which already terminates the
    episode); for a shaped/continuous reward (e.g. Door's reach+open
    blend) it truncates episodes that have stalled without progress."""

    enabled: bool = False
    patience_seconds: float = 5.0


@dataclasses.dataclass
class FullRlConfig:
    """Only used when policy_mode == 'full'."""

    action_space: str = "joint"  # "joint" (raw actuator targets) or "cartesian"
    cartesian_pos_scale: float = 0.02  # meters of eef translation per RL step
    cartesian_rot_scale: float = 0.1  # radians of eef rotation per RL step


@dataclasses.dataclass
class WorkspaceBoundsConfig:
    """Truncate the episode if the gripper leaves this box -- mainly useful
    with policy_mode='full', where nothing (no base-policy prior) otherwise
    keeps a randomly-initialized policy's actions anywhere near sensible
    early in training. Env-agnostic: only reads the eef position via FK, no
    object-specific state."""

    enabled: bool = False
    x_min: float = -0.5
    x_max: float = 0.5
    y_min: float = -0.6
    y_max: float = 0.6
    z_min: float = 0.7
    z_max: float = 1.3


@dataclasses.dataclass
class ResidualRlConfig:
    """Experiment-shaped configuration for `ResidualEnv`: policy mode
    (residual-on-base-policy vs. full RL), success/termination behavior, and
    early truncation. Everything that changes what is being trained (not how
    the run is operated) lives here, loaded from YAML, so different
    experiments are a `--config` flag rather than a code change.

    Deliberately env-agnostic: earlier versions of this config carried
    object-pose-dependent dense reward terms (reach/dist progress rewards)
    and a diverse-reset system, both mined for a specific task's object
    geometry (Toolbox pick-and-place). Neither generalizes to an arbitrary
    env without per-task code, so both were dropped when this was
    generalized beyond Toolbox -- see git history
    (residual_rl/configs/baseline.yaml etc.) if that machinery is needed
    again for a specific task."""

    policy_mode: str = "residual"  # "residual" (small delta on base policy) or "full"
    success_reward_threshold: float = 1.0
    early_truncation: EarlyTruncationConfig = dataclasses.field(
        default_factory=EarlyTruncationConfig
    )
    full_rl: FullRlConfig = dataclasses.field(default_factory=FullRlConfig)
    workspace_bounds: WorkspaceBoundsConfig = dataclasses.field(
        default_factory=WorkspaceBoundsConfig
    )

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        policy_mode = raw.get("policy_mode", "residual")
        if policy_mode not in POLICY_MODES:
            raise ValueError(
                f"Invalid policy_mode in {path}: {policy_mode!r}. "
                f"Expected one of {POLICY_MODES}."
            )

        success_reward_threshold = raw.get("success_reward_threshold", 1.0)

        early_truncation = EarlyTruncationConfig(**raw.get("early_truncation", {}))

        full_rl = FullRlConfig(**raw.get("full_rl", {}))
        if full_rl.action_space not in FULL_RL_ACTION_SPACES:
            raise ValueError(
                f"Invalid full_rl.action_space in {path}: "
                f"{full_rl.action_space!r}. Expected one of {FULL_RL_ACTION_SPACES}."
            )

        workspace_bounds = WorkspaceBoundsConfig(**raw.get("workspace_bounds", {}))

        return cls(
            policy_mode=policy_mode,
            success_reward_threshold=success_reward_threshold,
            early_truncation=early_truncation,
            full_rl=full_rl,
            workspace_bounds=workspace_bounds,
        )
