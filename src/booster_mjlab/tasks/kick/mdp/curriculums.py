"""Kick task curriculum terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from booster_mjlab.tasks.kick.mdp.commands import KickCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class KickStage(TypedDict, total=False):
    step: int
    speed: tuple[float, float]
    yaw: tuple[float, float]
    chip_angle: tuple[float, float]
    spawn_distance: tuple[float, float]


def kick_envelope(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[KickStage],
    spawn_event_name: str = "spawn_ball",
) -> dict[str, torch.Tensor]:
    """Widen the commanded speed/direction/chip envelope and spawn distance over training.

    Mirrors ``amp/curriculums.py``'s stage-annealing pattern, applied to the kick
    command's ranges and the ball-spawn event's distance range instead of an AMP
    style-reward weight.
    """
    del env_ids  # Unused; this curriculum applies globally, not per-env.
    command_term = env.command_manager.get_term(command_name)
    cfg = cast(KickCommandCfg, command_term.cfg)
    spawn_event_cfg = env.event_manager.get_term_cfg(spawn_event_name)

    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            if "speed" in stage:
                cfg.ranges.speed = stage["speed"]
            if "yaw" in stage:
                cfg.ranges.yaw = stage["yaw"]
            if "chip_angle" in stage:
                cfg.ranges.chip_angle = stage["chip_angle"]
            if "spawn_distance" in stage:
                spawn_event_cfg.params["distance_range"] = stage["spawn_distance"]

    return {
        "speed_min": torch.tensor(cfg.ranges.speed[0]),
        "speed_max": torch.tensor(cfg.ranges.speed[1]),
        "spawn_distance_max": torch.tensor(
            spawn_event_cfg.params["distance_range"][1]
        ),
    }
