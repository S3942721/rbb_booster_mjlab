"""Kick lifecycle advancement shared by termination and reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from booster_mjlab.tasks.kick.mdp.commands import KickCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def get_kick_command(env: ManagerBasedRlEnv, command_name: str = "kick") -> KickCommand:
    return cast(KickCommand, env.command_manager.get_term(command_name))


def get_progress_state(
    env: ManagerBasedRlEnv, command_name: str = "kick"
) -> dict[str, torch.Tensor]:
    """Compatibility/read-only view of state now owned by :class:`KickCommand`."""
    command = get_kick_command(env, command_name)
    return {
        "countdown": command.measure_countdown,
        "recovery_countdown": command.recovery_countdown,
        "just_measured": command.just_measured,
        "just_completed": command.just_completed,
    }


def _current_invalid_termination(
    env: ManagerBasedRlEnv, names: tuple[str, ...]
) -> torch.Tensor:
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active = set(env.termination_manager.active_terms)
    for name in names:
        if name in active:
            invalid |= env.termination_manager.get_term(name)
    return invalid


def _recovery_is_stable(
    env: ManagerBasedRlEnv,
    robot_asset_cfg: SceneEntityCfg,
    max_tilt: float,
    max_ang_vel: float,
) -> torch.Tensor:
    """Return whether the torso is upright and no longer rotating violently."""
    robot = env.scene[robot_asset_cfg.name]
    quat = robot.data.root_link_quat_w
    x, y = quat[:, 1], quat[:, 2]
    tilt = torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))
    angular_speed = torch.linalg.vector_norm(
        robot.data.root_link_ang_vel_w[:, :2], dim=-1
    )
    return (tilt <= max_tilt) & (angular_speed <= max_ang_vel)


def _forward_foot_contact(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    ball_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg,
    foot_body_names: tuple[str, str],
    min_forward_contact: float,
) -> torch.Tensor:
    """Return whether a contacting foot has the ball in front of its ankle.

    K1 currently has one collision mesh per foot, so MuJoCo cannot name a toe
    contact separately. The ball centre in the contacting foot frame is a stable
    proxy: a non-negative local x coordinate is the instep/toe half of the foot,
    while a negative coordinate is a heel-first strike.
    """
    sensor = env.scene.sensors[contact_sensor_name]
    found = sensor.data.found
    assert found is not None
    robot = env.scene[robot_asset_cfg.name]
    ball = env.scene[ball_asset_cfg.name]
    foot_ids, _ = robot.find_bodies(foot_body_names)
    if found.shape[1] != len(foot_ids):
        raise ValueError(
            f"{contact_sensor_name} has {found.shape[1]} contact slots but "
            f"{len(foot_ids)} foot bodies were configured"
        )
    foot_pos = robot.data.body_link_pos_w[:, foot_ids]
    foot_quat = robot.data.body_link_quat_w[:, foot_ids]
    ball_relative = quat_apply_inverse(
        foot_quat, ball.data.root_link_pos_w[:, None, :] - foot_pos
    )
    return ((found > 0) & (ball_relative[..., 0] >= min_forward_contact)).any(dim=-1)


def advance_kick_progress(
    env: ManagerBasedRlEnv,
    contact_sensor_name: str,
    command_name: str,
    ball_asset_cfg: SceneEntityCfg,
    measure_delay_steps: int,
    recovery_steps: int,
    invalid_termination_names: tuple[str, ...],
    recovery_max_tilt: float = 0.35,
    recovery_max_ang_vel: float = 1.5,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link"),
    min_forward_contact: float | None = None,
) -> torch.Tensor:
    """Advance contact, measurement and recovery exactly once per control step.

    ``measure_delay_steps=N`` means measure after N subsequent control steps;
    zero measures on the contact step. A failed launch finishes immediately. An
    accepted launch finishes only after the recovery window and only if no earlier
    failure termination fired on that step.
    """
    if measure_delay_steps < 0 or recovery_steps < 0:
        raise ValueError("measure_delay_steps and recovery_steps must be non-negative")
    if recovery_max_tilt <= 0.0 or recovery_max_ang_vel <= 0.0:
        raise ValueError("recovery stability limits must be positive")

    command = get_kick_command(env, command_name)
    command.begin_step()
    ball = env.scene[ball_asset_cfg.name]
    found = env.scene.sensors[contact_sensor_name].data.found
    assert found is not None
    contacted = found.any(dim=-1)

    fresh_contact = contacted & ~command.has_contacted
    fresh_ids = fresh_contact.nonzero(as_tuple=False).squeeze(-1)
    if fresh_ids.numel() > 0:
        contact_is_forward = (
            _forward_foot_contact(
                env,
                contact_sensor_name,
                ball_asset_cfg,
                robot_asset_cfg,
                foot_body_names,
                min_forward_contact,
            )
            if min_forward_contact is not None
            else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        )
        command.register_contact(fresh_ids, contact_is_forward[fresh_ids])
        command.measure_countdown[fresh_ids] = measure_delay_steps

    active_measurement = (command.measure_countdown > 0) & ~fresh_contact
    command.measure_countdown[active_measurement] -= 1
    resolve = (fresh_contact & (measure_delay_steps == 0)) | (
        (command.measure_countdown == 0) & ~fresh_contact & ~command.has_kicked
    )
    resolving_ids = resolve.nonzero(as_tuple=False).squeeze(-1)
    if resolving_ids.numel() > 0:
        command.register_kick(resolving_ids, ball.data.root_link_lin_vel_w)
        command.measure_countdown[resolving_ids] = -1

    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    failed_measurement = command.just_measured & ~command.accepted_launch
    failed_ids = failed_measurement.nonzero(as_tuple=False).squeeze(-1)
    if failed_ids.numel() > 0:
        command.mark_failed_attempt(failed_ids)
        done[failed_ids] = True

    accepted_now = command.just_measured & command.accepted_launch
    accepted_ids = accepted_now.nonzero(as_tuple=False).squeeze(-1)
    if accepted_ids.numel() > 0:
        command.phase[accepted_ids] = 3
        command.recovery_countdown[accepted_ids] = recovery_steps

    active_recovery = (command.recovery_countdown > 0) & ~accepted_now
    command.recovery_countdown[active_recovery] -= 1
    recovery_finished = (
        (
            (accepted_now & (recovery_steps == 0))
            | ((command.recovery_countdown == 0) & ~accepted_now)
        )
        & command.accepted_launch
        & ~command.attempt_finished
    )

    invalid = _current_invalid_termination(env, invalid_termination_names)
    stable = _recovery_is_stable(
        env, robot_asset_cfg, recovery_max_tilt, recovery_max_ang_vel
    )
    successful = recovery_finished & ~invalid & stable
    successful_ids = successful.nonzero(as_tuple=False).squeeze(-1)
    if successful_ids.numel() > 0:
        command.mark_stable_success(successful_ids)
        command.recovery_countdown[successful_ids] = -1
        done[successful_ids] = True

    return done
