import dataclasses

import yaml

RESET_CATEGORIES = ("default", "near_object", "stable_grasp", "near_goal")
POLICY_MODES = ("residual", "full")
FULL_RL_ACTION_SPACES = ("joint", "cartesian")


@dataclasses.dataclass
class RewardTermConfig:
    """reach/dist are progress rewards: weight * (prev_dist - curr_dist), not
    absolute-proximity rewards. This matters -- an absolute-proximity reward
    (weight * f(dist), e.g. exp/tanh-shaped) pays out every single step just
    for *being* close, which a discounted RL objective can exploit by holding
    position near a good-but-not-quite-successful state indefinitely rather
    than pushing through to the sparse success reward and ending the episode.
    Verified quantitatively for this task: with gamma=0.99 and the previous
    absolute formulation, looping near the goal for a full ~30s episode paid
    ~5.5x more discounted return than succeeding within ~1s. Progress reward
    can't be farmed this way -- holding still yields exactly zero regardless
    of duration, since prev_dist == curr_dist. Default weight is larger than
    the old absolute-reward default since per-step deltas are cm-scale, not
    the old formulation's bounded [0,1]-ish scale -- treat as a starting
    point that needs empirical retuning, not a derived value."""

    enabled: bool = False
    weight: float = 5.0


@dataclasses.dataclass
class EarlyTruncationConfig:
    enabled: bool = False
    patience_seconds: float = 5.0
    reach_threshold: float = 0.3


@dataclasses.dataclass
class FullRlConfig:
    """Only used when policy_mode == 'full'."""

    action_space: str = "joint"  # "joint" (raw actuator targets) or "cartesian"
    cartesian_pos_scale: float = 0.02  # meters of eef translation per RL step
    cartesian_rot_scale: float = 0.1  # radians of eef rotation per RL step


@dataclasses.dataclass
class WorkspaceBoundsConfig:
    """Truncate the episode if the gripper leaves this box -- mainly useful
    with policy_mode='full', where nothing (no ACT prior) otherwise keeps a
    randomly-initialized policy's actions anywhere near sensible early in
    training."""

    enabled: bool = False
    x_min: float = -0.5
    x_max: float = 0.5
    y_min: float = -0.6
    y_max: float = 0.6
    z_min: float = 0.7
    z_max: float = 1.3


@dataclasses.dataclass
class ResidualRlConfig:
    """Experiment-shaped configuration for `ResidualToolboxEnv`: policy mode
    (residual-on-ACT vs. full RL), reward composition, reset-category mix,
    and early truncation. Everything that changes what is being trained (not
    how the run is operated) lives here, loaded from YAML, so different
    experiments are a `--config` flag rather than a code change."""

    policy_mode: str = "residual"  # "residual" (small delta on ACT) or "full"
    reach: RewardTermConfig = dataclasses.field(default_factory=RewardTermConfig)
    dist: RewardTermConfig = dataclasses.field(default_factory=RewardTermConfig)
    reset_weights: dict = dataclasses.field(default_factory=lambda: {"default": 1.0})
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

        rewards = raw.get("rewards", {})
        reach = RewardTermConfig(**rewards.get("reach", {}))
        dist = RewardTermConfig(**rewards.get("dist", {}))

        reset_weights = raw.get("resets", {"default": 1.0})
        unknown_categories = set(reset_weights) - set(RESET_CATEGORIES)
        if unknown_categories:
            raise ValueError(
                f"Unknown reset categories in {path}: {unknown_categories}. "
                f"Expected a subset of {RESET_CATEGORIES}."
            )

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
            reach=reach,
            dist=dist,
            reset_weights=reset_weights,
            early_truncation=early_truncation,
            full_rl=full_rl,
            workspace_bounds=workspace_bounds,
        )
