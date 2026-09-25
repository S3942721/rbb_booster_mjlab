"""Ground-truth and explicitly stateful ball observations."""

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
    """Relative ball position/velocity in the robot's full body frame."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    rel_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_pos_w)
    rel_vel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_vel_w)
    return torch.cat([rel_pos_b, rel_vel_b], dim=-1)


def clean_ball_state_relative_b(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> torch.Tensor:
    """Eight-value clean observation with age=0 and valid=1."""
    state = ball_state_relative_b(env, robot_cfg, ball_cfg)
    status = torch.zeros(env.num_envs, 2, device=env.device)
    status[:, 1] = 1.0
    return torch.cat([state, status], dim=-1)


class noisy_ball_state_relative_b:
    """Reset-aware vision model with noise, fixed latency, dropout, age and validity.

    Output is ``[rel_pos(3), rel_vel(3), age_seconds, measurement_valid]``.
    The term advances at most once per environment control step even if an
    observation is queried multiple times by tooling.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        self.robot_cfg = params.get("robot_cfg", _DEFAULT_ROBOT_CFG)
        self.ball_cfg = params.get("ball_cfg", _DEFAULT_BALL_CFG)
        self.pos_noise_std = params.get("pos_noise_std", 0.07)
        self.vel_noise_std = params.get("vel_noise_std", 0.4)
        self.latency_steps = params.get("latency_steps", 2)
        self.dropout_prob = params.get("dropout_prob", 0.05)
        self.enable_noise = params.get("enable_noise", True)
        if self.latency_steps < 0:
            raise ValueError("latency_steps must be non-negative")

        self.buffer = torch.zeros(
            env.num_envs, self.latency_steps + 1, 6, device=env.device
        )
        self.output = torch.zeros(env.num_envs, 8, device=env.device)
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.last_update_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        self.history_age_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self.initialized[env_ids] = False
        self.buffer[env_ids] = 0.0
        self.output[env_ids] = 0.0
        self.last_update_step[env_ids] = -1
        self.history_age_steps[env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        truth = ball_state_relative_b(env, self.robot_cfg, self.ball_cfg)
        if not self.enable_noise:
            status = torch.zeros(env.num_envs, 2, device=env.device)
            status[:, 1] = 1.0
            return torch.cat([truth, status], dim=-1)

        current_step = int(env.common_step_counter)
        update_ids = (
            (self.last_update_step != current_step).nonzero(as_tuple=False).squeeze(-1)
        )
        if update_ids.numel() == 0:
            return self.output

        sample = truth[update_ids].clone()
        sample[:, :3] += torch.randn_like(sample[:, :3]) * self.pos_noise_std
        sample[:, 3:] += torch.randn_like(sample[:, 3:]) * self.vel_noise_std
        fresh_mask = ~self.initialized[update_ids]
        fresh_ids = update_ids[fresh_mask]
        if fresh_ids.numel() > 0:
            fresh_sample = sample[fresh_mask]
            self.buffer[fresh_ids] = fresh_sample[:, None, :]
            self.output[fresh_ids, :6] = fresh_sample
            self.output[fresh_ids, 6] = self.latency_steps * env.step_dt
            self.output[fresh_ids, 7] = 1.0
            self.initialized[fresh_ids] = True

        continuing_ids = update_ids[~fresh_mask]
        self.history_age_steps[continuing_ids] = (
            self.history_age_steps[continuing_ids] + 1
        ).clamp_max(self.latency_steps)

        self.buffer[update_ids, :-1] = self.buffer[update_ids, 1:].clone()
        self.buffer[update_ids, -1] = sample
        delayed = self.buffer[update_ids, 0]
        dropout = torch.rand(update_ids.numel(), device=env.device) < self.dropout_prob
        self.output[update_ids, :6] = torch.where(
            dropout[:, None], self.output[update_ids, :6], delayed
        )
        base_age = self.history_age_steps[update_ids] * env.step_dt
        self.output[update_ids, 6] = torch.where(
            dropout, self.output[update_ids, 6] + env.step_dt, base_age
        )
        self.output[update_ids, 7] = (~dropout).float()
        self.last_update_step[update_ids] = current_step
        return self.output
