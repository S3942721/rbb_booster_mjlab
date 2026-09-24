"""Kick task termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab.tasks.kick.mdp._kick_state import advance_kick_progress

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def kick_complete(
    env: ManagerBasedRlEnv,
    command_name: str,
    contact_sensor_name: str,
    ball_asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    measure_delay_steps: int = 3,
) -> torch.Tensor:
    """Ends the episode the step the kick outcome is measured (single kick/episode).

    Runs before rewards each step (mjlab computes terminations first), so this term
    owns advancing the contact->measurement countdown and calling
    ``KickCommand.register_kick()``; ``mdp/rewards.py``'s ``kick_outcome`` reward
    only reads the result this same step.
    """
    resolving_ids = advance_kick_progress(
        env, contact_sensor_name, command_name, ball_asset_cfg, measure_delay_steps
    )
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done[resolving_ids] = True
    return done
