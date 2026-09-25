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
    recovery_steps: int = 20,
    recovery_max_tilt: float = 0.35,
    recovery_max_ang_vel: float = 1.5,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link"),
    min_forward_contact: float | None = None,
    invalid_termination_names: tuple[str, ...] = (
        "fell_over",
        "illegal_contact",
        "mis_kick",
    ),
) -> torch.Tensor:
    """End after a measured failed attempt or recovered accepted launch.

    Runs before rewards each step (mjlab computes terminations first), so this term
    owns advancing the contact->measurement countdown and calling
    ``KickCommand.register_kick()``; event rewards only read the result this same
    step. Accepted launches remain active through a recovery window before this
    term reports completion.
    """
    return advance_kick_progress(
        env,
        contact_sensor_name,
        command_name,
        ball_asset_cfg,
        measure_delay_steps,
        recovery_steps,
        invalid_termination_names,
        recovery_max_tilt,
        recovery_max_ang_vel,
        robot_asset_cfg,
        foot_body_names,
        min_forward_contact,
    )
