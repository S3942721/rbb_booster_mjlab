"""RL configuration for Booster K1 kick task."""

from mjlab.rl import RslRlModelCfg

from booster_mjlab.rl.config import (
    RslRlMuonPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def booster_k1_kick_ppo_runner_cfg(use_muon: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster K1 kick task.

    Plain PPO, no AMP style reward, per the "pure task reward first" decision --
    revisit once the outcome-tracking reward has converged.
    """
    algorithm_cls = RslRlMuonPpoAlgorithmCfg if use_muon else RslRlPpoAlgorithmCfg
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                # The unbounded scalar Gaussian reached a mean std of 4.01
                # (individual joints 5.42) in the first full kick run. Bound
                # exploration while leaving the mean action free for a strike.
                "init_std": 0.45,
                "std_type": "log",
                "std_range": (0.05, 0.75),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=algorithm_cls(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="k1_kick" + ("_muon" if use_muon else ""),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
