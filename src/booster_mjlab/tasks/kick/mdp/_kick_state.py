"""Shared kick-progress state, advanced by mdp/terminations.py (runs before rewards).

mjlab computes terminations before rewards each step (see
``ManagerBasedRlEnv.step()``), so the contact-detection / outcome-registration
side effect lives in the termination term (``kick_complete``); the reward term
(``kick_outcome``) only reads the result computed earlier in the same step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_STATE_KEY = "_kick_progress"


def get_progress_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    if not hasattr(env, _STATE_KEY):
        setattr(
            env,
            _STATE_KEY,
            {
                # -1 = no contact yet this episode; else counts down to 0 on the
                # step the outcome is measured/registered (and the episode ends).
                "countdown": torch.full(
                    (env.num_envs,), -1, dtype=torch.long, device=env.device
                ),
            },
        )
    state = getattr(env, _STATE_KEY)
    just_reset = env.episode_length_buf == 0
    state["countdown"][just_reset] = -1
    return state


def advance_kick_progress(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    command_name: str,
    ball_asset_cfg: SceneEntityCfg,
    measure_delay_steps: int,
) -> torch.Tensor:
    """Advance the per-env contact->measurement countdown by one step.

    Call exactly once per step (from the ``kick_complete`` termination term, which
    runs before rewards). Returns the env ids resolving (outcome measured and
    registered) this step.
    """
    ball = env.scene[ball_asset_cfg.name]
    state = get_progress_state(env)
    countdown = state["countdown"]

    contact_sensor = env.scene.sensors[contact_sensor_name]
    contacted = contact_sensor.data.found.any(dim=-1)

    fresh_contact = contacted & (countdown < 0)
    countdown[fresh_contact] = measure_delay_steps

    active = countdown >= 0
    countdown[active] -= 1

    resolving_ids = (countdown == 0).nonzero(as_tuple=False).squeeze(-1)
    if resolving_ids.numel() > 0:
        kick_cmd = env.command_manager.get_term(command_name)
        kick_cmd.register_kick(resolving_ids, ball.data.root_link_lin_vel_w)
    return resolving_ids
