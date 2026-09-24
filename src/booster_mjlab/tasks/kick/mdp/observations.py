"""Kick task observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")


def ball_state_relative_b(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Ground-truth ball position/velocity relative to the robot, in its body frame.

    Shape (num_envs, 6): ``[rel_pos(3), rel_vel(3)]``. Used directly for the critic;
    the actor instead sees ``noisy_ball_state_relative_b`` below.
    """
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]

    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    rel_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)
    rel_vel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_vel_w)
    return torch.cat([rel_pos_b, rel_vel_b], dim=-1)


def noisy_ball_state_relative_b(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    pos_noise_std: float = 0.07,
    vel_noise_std: float = 0.4,
    latency_steps: int = 2,
    dropout_prob: float = 0.05,
) -> torch.Tensor:
    """Vision-corrupted ball state relative to the robot, in its body frame.

    Placeholder noise model (per user decision, tune once real vision stats are
    available): Gaussian position/velocity noise, a fixed-length latency buffer, and
    a dropout probability that holds the last delayed estimate instead of the fresh
    one (models a stale/missed vision update rather than a spike to zero).

    Shape (num_envs, 6): ``[rel_pos(3), rel_vel(3)]``.
    """
    ground_truth = ball_state_relative_b(env, robot_cfg, ball_cfg)
    noise = torch.zeros_like(ground_truth)
    noise[:, 0:3] = torch.randn_like(ground_truth[:, 0:3]) * pos_noise_std
    noise[:, 3:6] = torch.randn_like(ground_truth[:, 3:6]) * vel_noise_std
    sample = ground_truth + noise

    just_reset = env.episode_length_buf == 0

    buf_key = "_kick_ball_obs_buffer"
    if not hasattr(env, buf_key):
        setattr(env, buf_key, sample.unsqueeze(1).repeat(1, latency_steps + 1, 1))
    buf = getattr(env, buf_key)
    buf = torch.where(just_reset.view(-1, 1, 1), sample.unsqueeze(1), buf)
    buf = torch.cat([buf[:, 1:], sample.unsqueeze(1)], dim=1)
    setattr(env, buf_key, buf)
    delayed = buf[:, 0]

    hold_key = "_kick_ball_obs_hold"
    if not hasattr(env, hold_key):
        setattr(env, hold_key, delayed.clone())
    held = getattr(env, hold_key)
    held = torch.where(just_reset.view(-1, 1), delayed, held)

    dropout = torch.rand(env.num_envs, device=env.device) < dropout_prob
    output = torch.where(dropout.view(-1, 1), held, delayed)
    setattr(env, hold_key, output)
    return output
