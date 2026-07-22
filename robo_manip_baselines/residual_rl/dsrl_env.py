import numpy as np
from gymnasium import spaces

from robo_manip_baselines.common import DataKey, convert_data_to_policy

from .config import ResidualRlConfig
from .frozen_maniflow_policy import FrozenManiFlowPolicy
from .frozen_policy_env_base import FrozenPolicyEnvBase


class DsrlEnv(FrozenPolicyEnvBase):
    """DSRL (Diffusion/flow Steering via RL): instead of adding an
    action-space residual on top of the frozen ManiFlow policy's decoded
    action (see `ResidualEnv`), the RL policy chooses the *initial noise*
    fed into ManiFlow's flow-matching sampler, steering which trajectory the
    deterministic denoising process produces. The frozen policy's own
    weights are never touched -- this is RL purely over its noise input,
    per "Steering Your Diffusion Policy with Latent Space Reinforcement
    Learning" (Wagenmaker et al., CoRL 2025).

    Requires the forked ManiFlow's `predict_action(..., noise=...)` override
    (see `third_party/ManiFlow_Policy/ManiFlow/maniflow/policy/
    maniflow_image_policy.py`) -- upstream ManiFlow always samples its own
    `torch.randn` internally and has no hook for this.

    Same fine-grained cadence as `ResidualEnv`: one `step()` call = `skip`
    raw sim sub-steps, at the same "control tick" granularity ManiFlow is
    normally deployed at (matches nakamoto/dsrl_pi0's reference
    implementation, which inserts one replay-buffer transition per raw
    decision point, not per chunk). `raw_action` (this step's proposed
    noise) is passed to `FrozenManiFlowPolicy.get_action_with_noise` on
    every call, but that method only actually *consumes* it once its
    internal chunk buffer runs empty -- every other call's noise is
    silently ignored in favor of the already-decided cached action, exactly
    like ManiFlow's own `n_action_steps`-paced re-inference. So
    `n_action_steps` (constructor arg, overridable) *is* the DSRL paper's
    `query_frequency`: how many `step()` calls occur between noise
    decisions that actually take effect. An earlier version of this class
    instead bundled a whole chunk into one `step()` call, reasoning that
    was the "no decision ever wasted" design -- that turned out to be a
    mistake: it isn't how the reference implementation works, it makes
    `--total_timesteps` mean something like 16x more real experience here
    than for `ResidualEnv` (very surprising if you don't know it), and with
    rendering on it looks like the sim has frozen between the rare visible
    steps. This version fixes all three by not being clever about it.

    Two noise modes (`noise_mode` in `__init__`):
    - "shared" (default): one `action_dim`-sized noise vector, broadcast
      across all `horizon` timesteps of the chunk. Much smaller action
      space than "full" (matches `ResidualEnv`'s action dimensionality,
      for a fair comparison) at the cost of not letting different chunk
      timesteps be steered independently -- the "basics first" starting
      point.
    - "full": independent noise per chunk timestep
      (`horizon * action_dim`-dim action), matching the DSRL paper's own
      formulation. Higher-dimensional, harder for SAC to explore
      efficiently -- try "shared" first.

    Observation is proprioception only by default (the same state the frozen
    policy itself conditions on) -- no base-action concatenation the way
    `ResidualEnv` does, since there's no direct "residual on top of X" to
    contextualize; the noise doesn't have a natural action-space
    interpretation to compare against.

    `include_object_pose` (opt-in, off by default) appends the manipulated
    object's *absolute* world-frame pose (`DataKey.MEASURED_OBJECT_POSE`,
    requires the wrapped env to implement `get_object_pose()` -- see
    `MujocoUR5eInsertEnv.get_object_pose`). Deliberately absolute, not
    relative to the eef: the peg here is attached via a soft MuJoCo weld
    (not a rigid grasp) and can genuinely slip/separate under contact, so
    "relative to eef" would conflate arm motion with the actual slip signal
    of interest; absolute pose against a *fixed* hole location (single
    `world_idx`) already carries that signal directly. This is meant to
    test a specific hypothesis: that the persistent "same edge, every time"
    failure reflects a partially-observed state (the policy can't tell,
    from proprioception alone, that the peg has shifted in the gripper)
    rather than an unconverged or fundamentally uncapable policy.
    """

    def __init__(
        self,
        env_id,
        config=None,
        maniflow_checkpoint=None,
        world_idx_list=None,
        noise_mode="shared",
        noise_scale=2.0,
        n_action_steps=None,
        include_object_pose=False,
        max_episode_duration=30.0,
        render_mode=None,
        device="cuda",
    ):
        config = config if config is not None else ResidualRlConfig()
        super().__init__(env_id, config, world_idx_list, max_episode_duration, render_mode)

        if maniflow_checkpoint is None:
            raise ValueError("DsrlEnv requires maniflow_checkpoint")
        if noise_mode not in ("shared", "full"):
            raise ValueError(f"Invalid noise_mode: {noise_mode!r}, expected 'shared' or 'full'")
        if include_object_pose and not hasattr(self.base_env.unwrapped, "get_object_pose"):
            raise ValueError(
                f"include_object_pose=True requires {env_id} to implement "
                f"get_object_pose() -- it doesn't."
            )
        self.include_object_pose = include_object_pose

        self.frozen_policy = FrozenManiFlowPolicy(
            maniflow_checkpoint,
            self.motion_manager,
            device=device,
            n_action_steps=n_action_steps,
        )
        self.skip = self.frozen_policy.skip
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale

        action_dim = self.frozen_policy.action_dim
        horizon = self.frozen_policy.horizon
        noise_action_dim = action_dim if noise_mode == "shared" else horizon * action_dim
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(noise_action_dim,), dtype=np.float64
        )

        state_dim = sum(self.frozen_policy.state_dims)
        if self.include_object_pose:
            state_dim += DataKey.get_dim_for_policy(DataKey.MEASURED_OBJECT_POSE, self.base_env)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float64
        )

    def _build_noise(self, raw_action):
        horizon = self.frozen_policy.horizon
        action_dim = self.frozen_policy.action_dim
        if self.noise_mode == "shared":
            noise = np.tile(raw_action, (horizon, 1))
        else:
            noise = raw_action.reshape(horizon, action_dim)
        return noise * self.noise_scale

    def _make_obs(self, base_obs):
        state = self.frozen_policy.get_raw_state(base_obs)
        if self.include_object_pose:
            object_pose = convert_data_to_policy(
                self.motion_manager.get_data(DataKey.MEASURED_OBJECT_POSE, base_obs),
                DataKey.MEASURED_OBJECT_POSE,
            )
            state = np.concatenate([state, object_pose])
        return state

    def reset(self, *, seed=None, options=None):
        base_obs, info = super().reset(seed=seed, options=options)
        self.frozen_policy.reset()
        return self._make_obs(base_obs), info

    def step(self, raw_action):
        raw_action = np.clip(raw_action, -1.0, 1.0)
        noise = self._build_noise(raw_action)

        base_obs, info = self._base_obs, self._info
        reward = 0.0
        for sub_idx in range(self.skip):
            is_skip = sub_idx != 0
            if not is_skip:
                # Consumed only if the frozen policy's chunk buffer is
                # empty right now -- otherwise silently ignored in favor of
                # the already-decided cached action. See class docstring.
                action_vec = self.frozen_policy.get_action_with_noise(
                    base_obs, info["rgb_images"], noise
                )
            self.frozen_policy.apply_action(action_vec, is_skip)
            env_action = self.motion_manager.get_command_data(DataKey.COMMAND_JOINT_POS)
            env_action = np.clip(
                env_action, self.base_env.action_space.low, self.base_env.action_space.high
            )
            base_obs, step_reward, _, _, info = self.base_env.step(env_action)
            reward = max(reward, step_reward)

        self._base_obs = base_obs
        self._info = info

        terminated, truncated, info = self._finalize_step(reward, base_obs, info)

        return self._make_obs(base_obs), float(reward), terminated, truncated, info
