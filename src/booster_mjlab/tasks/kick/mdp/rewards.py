"""Kick task reward terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.managers.scene_entity_config import SceneEntityCfg

from booster_mjlab.tasks.kick.mdp._kick_state import get_progress_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DISTANCE_KEY = "_kick_prev_distance"


def kick_outcome(
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_std: float = 1.0,
    direction_std: float = math.radians(20.0),
) -> torch.Tensor:
    """Sparse reward comparing post-contact ball velocity to the kick command.

    Reads the outcome registered this step by the ``kick_complete`` termination
    term (which runs before rewards and owns contact detection/countdown
    advancement -- see mdp/terminations.py and mdp/_kick_state.py). Zero on all
    steps that don't resolve a kick.
    """
    state = get_progress_state(env)
    resolving_ids = (state["countdown"] == 0).nonzero(as_tuple=False).squeeze(-1)
    reward = torch.zeros(env.num_envs, device=env.device)
    if resolving_ids.numel() == 0:
        return reward

    kick_cmd = env.command_manager.get_term(command_name)
    achieved = kick_cmd.achieved_vel_w[resolving_ids]
    target = kick_cmd.target_vel_w[resolving_ids]

    speed_err = torch.linalg.norm(achieved, dim=-1) - torch.linalg.norm(target, dim=-1)
    target_dir = F.normalize(target, dim=-1, eps=1e-6)
    achieved_dir = F.normalize(achieved, dim=-1, eps=1e-6)
    cos_sim = (target_dir * achieved_dir).sum(dim=-1).clamp(-1.0, 1.0)
    direction_err = torch.acos(cos_sim)

    speed_reward = torch.exp(-(speed_err**2) / (speed_std**2))
    direction_reward = torch.exp(-(direction_err**2) / (direction_std**2))
    reward[resolving_ids] = speed_reward * direction_reward
    return reward


def closing_velocity(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_asset_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Dense reward for reducing robot-to-ball planar distance; zero after contact.

    Rewards the *decrease* in distance since the previous step rather than the raw
    distance, so it doesn't dominate the sparse kick-outcome term and naturally goes
    to zero once the robot parks next to the ball.
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_asset_cfg.name]
    state = get_progress_state(env)
    contacted = state["countdown"] >= 0

    to_ball = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    distance = torch.linalg.norm(to_ball, dim=-1)

    if not hasattr(env, _DISTANCE_KEY):
        setattr(env, _DISTANCE_KEY, distance.clone())
    prev_distance = getattr(env, _DISTANCE_KEY)
    just_reset = env.episode_length_buf == 0
    prev_distance = torch.where(just_reset, distance, prev_distance)

    closing = (prev_distance - distance).clamp(min=0.0)
    setattr(env, _DISTANCE_KEY, distance.clone())
    return torch.where(contacted, torch.zeros_like(closing), closing)


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant per-step penalty; combine with a negative weight to reward speed."""
    return torch.ones(env.num_envs, device=env.device)
