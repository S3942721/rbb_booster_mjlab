"""Rewards for directional approach, tolerance-aware launch and recovery."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from booster_mjlab.tasks.kick.mdp._kick_state import get_kick_command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _event_value(env: ManagerBasedRlEnv, value: torch.Tensor) -> torch.Tensor:
    """Make a one-step event retain its configured return under dt scaling."""
    return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def facing_ball(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = math.radians(25.0),
) -> torch.Tensor:
    """Reward keeping the ball in front of the torso before contact.

    The policy observes ball state in its body frame. This term makes that
    visibility assumption explicit, and switches off once the kick starts so
    the departing ball cannot pull the torso out of recovery.
    """
    command = get_kick_command(env, command_name)
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    to_ball_w = ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2]
    w, x, y, z = robot.data.root_link_quat_w.unbind(dim=-1)
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    ball_heading = torch.atan2(to_ball_w[:, 1], to_ball_w[:, 0])
    error = torch.abs(_wrap_angle(ball_heading - heading))
    return torch.exp(-torch.square(error / std)) * (command.phase < 2).float()


def backward_lean_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    free_tilt: float = math.radians(5.0),
    std: float = math.radians(10.0),
) -> torch.Tensor:
    """Penalize torso pitch away from the ball-facing direction.

    The robot body's x axis points forward. In body coordinates, backward lean
    makes the projected gravity x component negative. A small neutral range
    preserves normal gait variation while strongly discouraging the unstable
    backward posture that also hides the feet and ball from the head camera.
    """
    robot = env.scene[asset_cfg.name]
    gravity_b = quat_apply_inverse(
        robot.data.root_link_quat_w, robot.data.gravity_vec_w
    )
    backward_component = (-gravity_b[:, 0] - math.sin(free_tilt)).clamp_min(0.0)
    return torch.square(backward_component / math.sin(std))


class directional_approach_progress:
    """Signed progress to a behind-ball staging pose, then toward the ball.

    Before alignment, a waypoint to either side of the ball provides a detour when
    the direct robot-to-staging segment crosses a clearance disk. The shorter side
    is latched per episode. Once close and aligned at the staging pose, the term
    switches to strike progress. Phase transitions pay zero to avoid reward jumps.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.command_name = cfg.params.get("command_name", "kick")
        self.robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.ball_cfg = cfg.params.get("ball_cfg", SceneEntityCfg("ball"))
        self.staging_distance = cfg.params.get("staging_distance", 0.45)
        self.clearance_radius = cfg.params.get("clearance_radius", 0.35)
        self.staging_threshold = cfg.params.get("staging_threshold", 0.22)
        self.heading_threshold = cfg.params.get("heading_threshold", math.radians(35.0))
        self.heading_cost_scale = cfg.params.get("heading_cost_scale", 0.20)
        self.max_progress_per_step = cfg.params.get("max_progress_per_step", 0.12)
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.route_sign = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.selected_direction = torch.zeros(env.num_envs, 2, device=env.device)
        self.last_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        self.initialized[env_ids] = False
        self.route_sign[env_ids] = 0
        self.selected_direction[env_ids] = 0.0
        self.last_phase[env_ids] = 0
        self.previous[env_ids] = 0.0

    def _select_direction(
        self,
        preferred: torch.Tensor,
        tolerance: torch.Tensor,
        robot_xy: torch.Tensor,
        ball_xy: torch.Tensor,
    ) -> torch.Tensor:
        preferred_yaw = torch.atan2(preferred[:, 1], preferred[:, 0])
        direct_yaw = torch.atan2(
            ball_xy[:, 1] - robot_xy[:, 1], ball_xy[:, 0] - robot_xy[:, 0]
        )
        offset = _wrap_angle(direct_yaw - preferred_yaw).clamp(
            min=-tolerance, max=tolerance
        )
        chosen = preferred_yaw + offset
        return torch.stack([torch.cos(chosen), torch.sin(chosen)], dim=-1)

    def _approach_cost(
        self,
        robot_xy: torch.Tensor,
        ball_xy: torch.Tensor,
        direction: torch.Tensor,
        route_sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        staging = ball_xy - direction * self.staging_distance
        to_stage = staging - robot_xy
        direct_cost = torch.linalg.vector_norm(to_stage, dim=-1)

        segment_len_sq = (to_stage * to_stage).sum(dim=-1).clamp_min(1.0e-6)
        projection = ((ball_xy - robot_xy) * to_stage).sum(dim=-1) / segment_len_sq
        closest = robot_xy + projection.clamp(0.0, 1.0)[:, None] * to_stage
        crosses = (
            (projection > 0.0)
            & (projection < 1.0)
            & (
                torch.linalg.vector_norm(closest - ball_xy, dim=-1)
                < self.clearance_radius
            )
        )

        perpendicular = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
        behind = ball_xy - direction * (0.35 * self.staging_distance)
        waypoint_left = behind + perpendicular * self.clearance_radius
        waypoint_right = behind - perpendicular * self.clearance_radius
        left_cost = torch.linalg.vector_norm(waypoint_left - robot_xy, dim=-1)
        left_cost += torch.linalg.vector_norm(staging - waypoint_left, dim=-1)
        right_cost = torch.linalg.vector_norm(waypoint_right - robot_xy, dim=-1)
        right_cost += torch.linalg.vector_norm(staging - waypoint_right, dim=-1)

        new_sign = torch.where(left_cost <= right_cost, 1, -1)
        route_sign = torch.where((route_sign == 0) & crosses, new_sign, route_sign)
        detour_cost = torch.where(route_sign > 0, left_cost, right_cost)
        cost = torch.where(route_sign == 0, direct_cost, detour_cost)
        return cost, staging, route_sign

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        command = get_kick_command(env, self.command_name)
        robot = env.scene[self.robot_cfg.name]
        ball = env.scene[self.ball_cfg.name]
        robot_xy = robot.data.root_link_pos_w[:, :2]
        ball_xy = ball.data.root_link_pos_w[:, :2]
        preferred = F.normalize(command.target_vel_w[:, :2], dim=-1, eps=1.0e-6)

        uninitialized = ~self.initialized
        if uninitialized.any():
            self.selected_direction[uninitialized] = self._select_direction(
                preferred[uninitialized],
                command.direction_tolerance[uninitialized],
                robot_xy[uninitialized],
                ball_xy[uninitialized],
            )

        route_cost, staging, route_sign = self._approach_cost(
            robot_xy,
            ball_xy,
            self.selected_direction,
            self.route_sign,
        )
        self.route_sign = route_sign

        quat = robot.data.root_link_quat_w
        w, x, y, z = quat.unbind(dim=-1)
        heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        desired_heading = torch.atan2(
            self.selected_direction[:, 1], self.selected_direction[:, 0]
        )
        heading_error = torch.abs(_wrap_angle(heading - desired_heading))
        staging_error = torch.linalg.vector_norm(staging - robot_xy, dim=-1)

        aligned = (
            (command.phase == 0)
            & (staging_error < self.staging_threshold)
            & (heading_error < self.heading_threshold)
        )
        command.phase[aligned] = 1

        approach_potential = -route_cost - self.heading_cost_scale * heading_error
        strike_potential = -torch.linalg.vector_norm(ball_xy - robot_xy, dim=-1)
        current = torch.where(command.phase == 0, approach_potential, strike_potential)
        active = command.phase < 2
        phase_changed = command.phase != self.last_phase
        valid = self.initialized & ~phase_changed & active
        progress = torch.where(
            valid, current - self.previous, torch.zeros_like(current)
        )
        progress = progress.clamp(
            -self.max_progress_per_step, self.max_progress_per_step
        )

        self.previous = current.detach().clone()
        self.last_phase = command.phase.clone()
        self.initialized[:] = True
        return _event_value(env, progress)


def launch_quality(
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_scale: float = 0.75,
    direction_scale: float = math.radians(30.0),
    elevation_scale: float = math.radians(15.0),
) -> torch.Tensor:
    """Preferred-target feedback, softened for broad urgent clearances.

    An accepted launch gets a separate binary bonus. This term retains a
    gradient inside the accepted envelope, so precision demands actually change
    the learned strike instead of merely defining a pass/fail boundary.
    """
    command = get_kick_command(env, command_name)
    achieved = command.achieved_vel_w
    horizontal_speed = torch.linalg.vector_norm(achieved[:, :2], dim=-1)
    speed_miss = F.relu(command.speed_min - horizontal_speed)
    speed_miss += F.relu(horizontal_speed - command.speed_max)
    direction_error = command.metrics["direction_error"]
    elevation_error = command.metrics["elevation_error"]
    direction_miss = F.relu(direction_error - command.direction_tolerance)
    elevation_miss = F.relu(elevation_error - command.elevation_tolerance)
    envelope_quality = torch.exp(
        -((speed_miss / speed_scale) ** 2)
        - ((direction_miss / direction_scale) ** 2)
        - ((elevation_miss / elevation_scale) ** 2)
    )
    preferred_speed = torch.linalg.vector_norm(command.target_vel_w[:, :2], dim=-1)
    preferred_speed_error = (horizontal_speed - preferred_speed).abs()
    preferred_quality = torch.exp(
        -((preferred_speed_error / speed_scale) ** 2)
        - ((direction_error / direction_scale) ** 2)
        - ((elevation_error / elevation_scale) ** 2)
    )
    tolerance_width = (command.direction_tolerance / math.radians(60.0)).clamp(0.0, 1.0)
    precision_weight = (1.0 - tolerance_width) * (1.0 - command.urgency)
    quality = envelope_quality * (
        (1.0 - precision_weight) + precision_weight * preferred_quality
    )
    return _event_value(env, command.just_measured.float() * quality)


def useful_contact(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = get_kick_command(env, command_name)
    event = command.just_contacted & command.valid_strike_contact
    return _event_value(env, event.float())


def accepted_launch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = get_kick_command(env, command_name)
    event = command.just_measured & command.accepted_launch
    return _event_value(env, event.float())


def stable_completion(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = get_kick_command(env, command_name)
    return _event_value(env, command.just_completed.float())


def failure_event(
    env: ManagerBasedRlEnv,
    termination_names: tuple[str, ...] = ("fell_over", "illegal_contact", "mis_kick"),
) -> torch.Tensor:
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active = set(env.termination_manager.active_terms)
    for name in termination_names:
        if name in active:
            failed |= env.termination_manager.get_term(name)
    return _event_value(env, failed.float())


def time_penalty(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Per-second cost, stronger for urgent commands."""
    command = get_kick_command(env, command_name)
    return 0.5 + command.urgency


# Backward-compatible symbol used by old saved configs. New configs use
# launch_quality and directional_approach_progress.
kick_outcome = launch_quality
