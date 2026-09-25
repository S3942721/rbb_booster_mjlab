"""Tolerance-conditioned kick command and per-episode kick state.

The preferred launch velocity is stored in world coordinates so it remains fixed
while the robot walks around the ball. The actor receives the vector in the
robot's full body frame plus the accepted speed interval, angular tolerances and
urgency. A command therefore represents both accurate passes and broad,
time-critical clearances without changing policy interfaces at runtime.
"""

from __future__ import annotations

import math
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


def _heading_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return world yaw for wxyz quaternions."""
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_between_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=1.0e-6)
    b = F.normalize(b, dim=-1, eps=1.0e-6)
    return torch.acos((a * b).sum(dim=-1).clamp(-1.0, 1.0))


class KickCommand(CommandTerm):
    """Preferred launch and acceptable envelope, with reset-owned task state."""

    cfg: KickCommandCfg

    def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.ball: Entity = env.scene[cfg.ball_entity_name]

        self.target_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.achieved_vel_w = torch.zeros_like(self.target_vel_w)
        self.speed_min = torch.zeros(self.num_envs, device=self.device)
        self.speed_max = torch.zeros(self.num_envs, device=self.device)
        self.direction_tolerance = torch.zeros(self.num_envs, device=self.device)
        self.elevation_tolerance = torch.zeros(self.num_envs, device=self.device)
        self.urgency = torch.zeros(self.num_envs, device=self.device)

        # CommandManager.reset runs after terminal rewards and logging and gives
        # us the exact reset IDs, so it is the reliable owner for episode state.
        self.measure_countdown = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.recovery_countdown = torch.full_like(self.measure_countdown, -1)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.has_contacted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.has_kicked = torch.zeros_like(self.has_contacted)
        self.valid_strike_contact = torch.zeros_like(self.has_contacted)
        self.accepted_launch = torch.zeros_like(self.has_contacted)
        self.stable_success = torch.zeros_like(self.has_contacted)
        self.attempt_finished = torch.zeros_like(self.has_contacted)
        self.just_contacted = torch.zeros_like(self.has_contacted)
        self.just_measured = torch.zeros_like(self.has_contacted)
        self.just_completed = torch.zeros_like(self.has_contacted)
        self.episode_active = torch.zeros_like(self.has_contacted)
        self.has_sampled_once = torch.zeros_like(self.has_contacted)
        # 0=forward, 1=side, 2=reverse, measured at episode reset.
        self.turn_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for name in (
            "contacted",
            "proper_contact",
            "accepted_launch",
            "stable_success",
            "speed_error",
            "direction_error",
            "elevation_error",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        env._kick_command_schema = (
            "v2:target_velocity_body_xyz,speed_min,speed_max,"
            "direction_tolerance,elevation_tolerance,urgency"
        )

    @property
    def command(self) -> torch.Tensor:
        target_b = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self.target_vel_w
        )
        return torch.cat(
            [
                target_b,
                self.speed_min[:, None],
                self.speed_max[:, None],
                self.direction_tolerance[:, None],
                self.elevation_tolerance[:, None],
                self.urgency[:, None],
            ],
            dim=-1,
        )

    def begin_step(self) -> None:
        """Clear one-control-step event flags before advancing task state."""
        self.just_contacted.zero_()
        self.just_measured.zero_()
        self.just_completed.zero_()

    def register_contact(
        self, env_ids: torch.Tensor, contact_is_forward: torch.Tensor | None = None
    ) -> None:
        fresh = env_ids[~self.has_contacted[env_ids]]
        if fresh.numel() == 0:
            return
        self.has_contacted[fresh] = True
        if contact_is_forward is None:
            contact_is_forward = torch.ones_like(fresh, dtype=torch.bool)
        self.valid_strike_contact[fresh] = contact_is_forward
        self.just_contacted[fresh] = True
        self.phase[fresh] = 2
        self.metrics["contacted"][fresh] = 1.0
        self.metrics["proper_contact"][fresh] = contact_is_forward.float()

    def register_kick(self, env_ids: torch.Tensor, ball_vel_w: torch.Tensor) -> None:
        """Latch the first launch measurement and evaluate its accepted envelope."""
        fresh = env_ids[~self.has_kicked[env_ids]]
        if fresh.numel() == 0:
            return
        achieved = ball_vel_w[fresh]
        target = self.target_vel_w[fresh]
        self.achieved_vel_w[fresh] = achieved
        self.has_kicked[fresh] = True
        self.just_measured[fresh] = True

        achieved_h = achieved[:, :2]
        target_h = target[:, :2]
        achieved_h_speed = torch.linalg.vector_norm(achieved_h, dim=-1)
        target_h_speed = torch.linalg.vector_norm(target_h, dim=-1)
        direction_error = _angle_between_2d(achieved_h, target_h)
        achieved_elevation = torch.atan2(
            achieved[:, 2], achieved_h_speed.clamp_min(1.0e-6)
        )
        target_elevation = torch.atan2(target[:, 2], target_h_speed.clamp_min(1.0e-6))
        elevation_error = torch.abs(achieved_elevation - target_elevation)
        speed_error = achieved_h_speed - target_h_speed

        accepted = (
            (achieved_h_speed >= self.speed_min[fresh])
            & (achieved_h_speed <= self.speed_max[fresh])
            & (achieved_h_speed >= self.cfg.min_useful_speed)
            & (direction_error <= self.direction_tolerance[fresh])
            & (elevation_error <= self.elevation_tolerance[fresh])
            & self.valid_strike_contact[fresh]
        )
        self.accepted_launch[fresh] = accepted
        self.metrics["accepted_launch"][fresh] = accepted.float()
        self.metrics["speed_error"][fresh] = speed_error
        self.metrics["direction_error"][fresh] = direction_error
        self.metrics["elevation_error"][fresh] = elevation_error

    def mark_stable_success(self, env_ids: torch.Tensor) -> None:
        self.stable_success[env_ids] = True
        self.attempt_finished[env_ids] = True
        self.just_completed[env_ids] = True
        self.metrics["stable_success"][env_ids] = 1.0

    def mark_failed_attempt(self, env_ids: torch.Tensor) -> None:
        self.attempt_finished[env_ids] = True

    def _update_metrics(self) -> None:
        # Event methods update metrics before CommandManager.reset logs them.
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        speed = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.speed)
        yaw_offset = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.yaw)
        chip = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.chip_angle)
        tolerance_fraction = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.speed_tolerance_fraction
        )
        direction_tolerance = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.direction_tolerance
        )
        elevation_tolerance = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.elevation_tolerance
        )
        urgency = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.urgency)

        reset_pose = getattr(self._env, "_kick_reset_root_pose_w", None)
        quat = (
            reset_pose[env_ids, 3:7]
            if reset_pose is not None
            else self.robot.data.root_link_quat_w[env_ids]
        )
        yaw = _heading_from_quat(quat) + yaw_offset
        horizontal = speed * torch.cos(chip)
        target = torch.stack(
            [
                horizontal * torch.cos(yaw),
                horizontal * torch.sin(yaw),
                speed * torch.sin(chip),
            ],
            dim=-1,
        )
        self.target_vel_w[env_ids] = target
        self.speed_min[env_ids] = (horizontal * (1.0 - tolerance_fraction)).clamp_min(
            self.cfg.min_useful_speed
        )
        self.speed_max[env_ids] = horizontal * (1.0 + tolerance_fraction)
        self.direction_tolerance[env_ids] = direction_tolerance
        self.elevation_tolerance[env_ids] = elevation_tolerance
        self.urgency[env_ids] = urgency

        reset_ball_pos = getattr(self._env, "_kick_reset_ball_pos_w", None)
        if reset_ball_pos is not None and reset_pose is not None:
            to_ball = reset_ball_pos[env_ids, :2] - reset_pose[env_ids, :2]
            demand = _angle_between_2d(target[:, :2], to_ball)
            self.turn_bin[env_ids] = torch.where(
                demand < torch.pi / 4,
                torch.zeros_like(demand, dtype=torch.long),
                torch.where(
                    demand < 3 * torch.pi / 4,
                    torch.ones_like(demand, dtype=torch.long),
                    torch.full_like(demand, 2, dtype=torch.long),
                ),
            )

        self.achieved_vel_w[env_ids] = 0.0
        self.measure_countdown[env_ids] = -1
        self.recovery_countdown[env_ids] = -1
        self.phase[env_ids] = 0
        self.has_contacted[env_ids] = False
        self.has_kicked[env_ids] = False
        self.valid_strike_contact[env_ids] = False
        self.accepted_launch[env_ids] = False
        self.stable_success[env_ids] = False
        self.attempt_finished[env_ids] = False
        self.just_contacted[env_ids] = False
        self.just_measured[env_ids] = False
        self.just_completed[env_ids] = False
        # Skip the synthetic first episode created by the wrapper's initial
        # reset. This also prevents randomised initial episode lengths from
        # contaminating the curriculum statistics.
        self.episode_active[env_ids] = self.has_sampled_once[env_ids]
        self.has_sampled_once[env_ids] = True
        for metric in self.metrics.values():
            metric[env_ids] = 0.0

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        del env_ids

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the preferred launch and its accepted velocity envelope.

        Blue is the preferred launch, cyan is the measured launch, yellow marks
        the azimuth cone, magenta marks the elevation limits, and orange marks
        the inner and outer speed limits. The arrows are velocity vectors scaled
        into world-space display lengths.
        """
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        ball_pos_ws = self.ball.data.root_link_pos_w.cpu().numpy()
        ball_vel_ws = self.ball.data.root_link_lin_vel_w.cpu().numpy()
        target_vel_ws = self.target_vel_w.cpu().numpy()
        for batch in env_indices:
            origin = ball_pos_ws[batch].copy()
            origin[2] += self.cfg.viz.z_offset
            target = target_vel_ws[batch]
            target_tensor = self.target_vel_w[batch]
            horizontal_speed = float(torch.linalg.vector_norm(target_tensor[:2]))
            target_speed = float(torch.linalg.vector_norm(target_tensor))
            yaw = float(torch.atan2(target_tensor[1], target_tensor[0]))
            elevation = math.atan2(
                float(target_tensor[2]), max(horizontal_speed, 1.0e-6)
            )

            def velocity_vector(
                speed: float, vector_yaw: float, vector_elevation: float
            ) -> object:
                horizontal = speed * math.cos(vector_elevation)
                return torch.tensor(
                    [
                        horizontal * math.cos(vector_yaw),
                        horizontal * math.sin(vector_yaw),
                        speed * math.sin(vector_elevation),
                    ]
                ).numpy()

            visualizer.add_arrow(
                origin,
                origin + target * self.cfg.viz.scale,
                color=self.cfg.viz.target_color,
                width=0.015,
                label=f"kick_target_{batch}",
            )
            visualizer.add_arrow(
                origin,
                origin + ball_vel_ws[batch] * self.cfg.viz.scale,
                color=self.cfg.viz.actual_color,
                width=0.015,
                label=f"kick_actual_{batch}",
            )
            for sign, name in ((-1.0, "left"), (1.0, "right")):
                cone = velocity_vector(
                    target_speed,
                    yaw + sign * float(self.direction_tolerance[batch]),
                    elevation,
                )
                visualizer.add_arrow(
                    origin,
                    origin + cone * self.cfg.viz.scale,
                    color=(1.0, 0.8, 0.0, 0.5),
                    width=0.008,
                    label=f"kick_azimuth_{name}_{batch}",
                )
            for sign, name in ((-1.0, "low"), (1.0, "high")):
                elevation_bound = velocity_vector(
                    target_speed,
                    yaw,
                    elevation + sign * float(self.elevation_tolerance[batch]),
                )
                visualizer.add_arrow(
                    origin,
                    origin + elevation_bound * self.cfg.viz.scale,
                    color=(0.9, 0.2, 0.9, 0.5),
                    width=0.008,
                    label=f"kick_elevation_{name}_{batch}",
                )
            for speed, name in (
                (float(self.speed_min[batch]), "minimum"),
                (float(self.speed_max[batch]), "maximum"),
            ):
                power = velocity_vector(speed, yaw, elevation)
                visualizer.add_arrow(
                    origin,
                    origin + power * self.cfg.viz.scale,
                    color=(1.0, 0.4, 0.0, 0.55),
                    width=0.008,
                    label=f"kick_speed_{name}_{batch}",
                )


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
    entity_name: str
    ball_entity_name: str
    min_useful_speed: float = 0.5

    @dataclass
    class Ranges:
        speed: tuple[float, float]
        """Preferred total launch speed, m/s."""
        yaw: tuple[float, float]
        """Preferred yaw offset from the robot's reset heading, radians."""
        chip_angle: tuple[float, float]
        speed_tolerance_fraction: tuple[float, float] = (0.15, 0.5)
        direction_tolerance: tuple[float, float] = (0.174533, 1.047198)
        elevation_tolerance: tuple[float, float] = (0.087266, 0.261799)
        urgency: tuple[float, float] = (0.0, 1.0)

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
