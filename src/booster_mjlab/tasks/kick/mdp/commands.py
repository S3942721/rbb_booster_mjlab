"""Kick command: desired ball-launch velocity vector, exposed robot-relative.

The command is a 3D vector describing how the robot should send the ball on
contact: direction (world-frame yaw), power (exit speed), and chip angle
(launch angle above horizontal; 0 = ground kick). It's sampled and stored in
world frame internally (stable across a step regardless of robot heading), but
the observation exposed to the policy via ``command`` is rotated into the
robot's current body frame each step -- the policy should never need world-frame
localisation, since that's handled upstream by the localisation stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class KickCommand(CommandTerm):
    """Desired ball-launch velocity; see module docstring."""

    cfg: KickCommandCfg

    def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.ball: Entity = env.scene[cfg.ball_entity_name]

        self.target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Latched by register_kick() the first time each episode a foot-ball
        # contact is detected (see tasks/kick/mdp/rewards.py / terminations.py).
        self.achieved_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.has_kicked = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["direction_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["kicked"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Commanded vector rotated into the robot's current body frame."""
        return quat_apply_inverse(self.robot.data.root_link_quat_w, self.target_vel_w)

    def register_kick(self, env_ids: torch.Tensor, ball_vel_w: torch.Tensor) -> None:
        """Latch the resulting ball velocity the first time contact is detected.

        Safe to call every step with the full contacting set: envs that already
        have ``has_kicked`` set are left alone, so only the *first* contact each
        episode is recorded (later touches, e.g. a second nudge, don't overwrite
        the measured outcome).
        """
        fresh = env_ids[~self.has_kicked[env_ids]]
        if len(fresh) == 0:
            return
        self.achieved_vel_w[fresh] = ball_vel_w[fresh]
        self.has_kicked[fresh] = True

    def _update_metrics(self) -> None:
        target_speed = torch.linalg.norm(self.target_vel_w, dim=-1)
        achieved_speed = torch.linalg.norm(self.achieved_vel_w, dim=-1)
        target_dir = F.normalize(self.target_vel_w, dim=-1, eps=1e-6)
        achieved_dir = F.normalize(self.achieved_vel_w, dim=-1, eps=1e-6)
        cos_sim = (target_dir * achieved_dir).sum(dim=-1).clamp(-1.0, 1.0)

        zeros = torch.zeros_like(target_speed)
        self.metrics["speed_error"] = torch.where(
            self.has_kicked, achieved_speed - target_speed, zeros
        )
        self.metrics["direction_error"] = torch.where(
            self.has_kicked, torch.acos(cos_sim), zeros
        )
        self.metrics["kicked"] = self.has_kicked.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        speed = r.uniform_(*self.cfg.ranges.speed)
        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.yaw)
        # Launch angle above horizontal; 0 = ground kick, >0 = chip.
        chip_angle = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.chip_angle
        )

        horizontal = speed * torch.cos(chip_angle)
        vx = horizontal * torch.cos(yaw)
        vy = horizontal * torch.sin(yaw)
        vz = speed * torch.sin(chip_angle)

        self.target_vel_w[env_ids] = torch.stack([vx, vy, vz], dim=-1)
        self.has_kicked[env_ids] = False
        self.achieved_vel_w[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        # `command` is a pure function of target_vel_w + current heading; nothing
        # to advance per-step.
        del env_ids

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw desired (blue) vs. actual ball velocity (cyan) arrows at the ball."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        ball_pos_ws = self.ball.data.root_link_pos_w.cpu().numpy()
        ball_vel_ws = self.ball.data.root_link_lin_vel_w.cpu().numpy()
        target_vel_ws = self.target_vel_w.cpu().numpy()
        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            origin = ball_pos_ws[batch].copy()
            origin[2] += z_offset

            visualizer.add_arrow(
                origin,
                origin + target_vel_ws[batch] * scale,
                color=self.cfg.viz.target_color,
                width=0.015,
                label=f"kick_target_{batch}",
            )
            visualizer.add_arrow(
                origin,
                origin + ball_vel_ws[batch] * scale,
                color=self.cfg.viz.actual_color,
                width=0.015,
                label=f"kick_actual_{batch}",
            )


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
    entity_name: str
    """Name of the robot entity in the scene."""
    ball_entity_name: str
    """Name of the ball entity in the scene."""

    @dataclass
    class Ranges:
        speed: tuple[float, float]
        """Commanded ball exit speed range, m/s."""
        yaw: tuple[float, float]
        """Commanded kick direction (world-frame yaw), radians. Widen via curriculum."""
        chip_angle: tuple[float, float]
        """Launch angle above horizontal, radians. 0 = ground kick, >0 = chip."""

    ranges: Ranges

    @dataclass
    class VizCfg:
        z_offset: float = 0.15
        scale: float = 0.3
        target_color: tuple[float, float, float, float] = (0.2, 0.2, 0.9, 0.8)
        actual_color: tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.8)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> KickCommand:
        return KickCommand(self, env)
