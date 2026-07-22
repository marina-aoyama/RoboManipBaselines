"""SAC actor/critic with LayerNorm after each hidden layer.

SB3's own `SACPolicy`/`Actor`/`ContinuousCritic` build plain Linear+activation
MLPs -- `create_mlp` actually supports inserting extra modules
(`post_linear_modules`) after each hidden layer, but that isn't threaded
through `Actor`/`ContinuousCritic`'s constructors, so it's unusable without
subclassing. Worth doing here specifically because of `--gradient_steps=20`
in train_dsrl_sac.py (a high update-to-data ratio): plain MLP critics are
well documented to be prone to runaway Q-value/critic-loss divergence at
high UTD (see e.g. CrossQ, "Bigger, Better, Faster"), which is exactly the
symptom observed in early DSRL runs (critic_loss growing ~580x over 6k
steps with no sign of plateauing). LayerNorm in the critic (and, for
consistency, the actor) is the standard mitigation.

Usage: pass the `LayerNormSACPolicy` class (not a string) directly to
`SAC(...)`, e.g. `SAC(LayerNormSACPolicy, vec_env, ...)`.
"""

import torch.nn as nn
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy


class LayerNormActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebuild latent_pi with LayerNorm after each hidden Linear -- the
        # separate self.mu/self.log_std output heads (built by Actor.__init__
        # after this) are left untouched, matching standard practice of not
        # normalizing the final output layer.
        latent_pi_net = create_mlp(
            self.features_dim,
            -1,
            self.net_arch,
            self.activation_fn,
            post_linear_modules=[nn.LayerNorm],
        )
        self.latent_pi = nn.Sequential(*latent_pi_net)


class LayerNormContinuousCritic(ContinuousCritic):
    def __init__(self, **kwargs):
        # Unlike Actor, ContinuousCritic.__init__ doesn't store net_arch/
        # features_dim/activation_fn as self attributes -- SACPolicy.make_critic
        # always calls this with everything as keywords (via **critic_kwargs),
        # so pull what's needed from kwargs directly rather than after super().
        net_arch = kwargs["net_arch"]
        features_dim = kwargs["features_dim"]
        activation_fn = kwargs.get("activation_fn", nn.ReLU)

        super().__init__(**kwargs)
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net_list = create_mlp(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
                post_linear_modules=[nn.LayerNorm],
            )
            q_net = nn.Sequential(*q_net_list)
            setattr(self, f"qf{idx}", q_net)  # replaces the module already
            # registered under this name by ContinuousCritic.__init__
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return LayerNormActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)
