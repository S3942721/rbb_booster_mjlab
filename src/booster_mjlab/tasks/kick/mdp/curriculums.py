"""Competence-gated kick curriculum with checkpointable state."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from booster_mjlab.tasks.kick.mdp.commands import KickCommand, KickCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class KickStage(TypedDict, total=False):
    speed: tuple[float, float]
    yaw: tuple[float, float]
    chip_angle: tuple[float, float]
    speed_tolerance_fraction: tuple[float, float]
    direction_tolerance: tuple[float, float]
    elevation_tolerance: tuple[float, float]
    urgency: tuple[float, float]
    spawn_distance: tuple[float, float]
    ball_speed: tuple[float, float]
    min_episodes_per_bin: int
    promotion_success_rate: float


class kick_envelope:
    """Promote only after forward, side and reverse bins meet the stage gate."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.command_name = cfg.params["command_name"]
        self.spawn_event_name = cfg.params.get("spawn_event_name", "spawn_ball")
        self.stages: list[KickStage] = cfg.params["stages"]
        if not self.stages:
            raise ValueError("kick curriculum requires at least one stage")
        self.stage_index = 0
        self.completed = torch.zeros(3, dtype=torch.long, device=env.device)
        self.successes = torch.zeros(3, dtype=torch.long, device=env.device)
        env._kick_curriculum_term = self

    def _apply_stage(self, env: ManagerBasedRlEnv) -> None:
        command = cast(KickCommand, env.command_manager.get_term(self.command_name))
        cfg = cast(KickCommandCfg, command.cfg)
        stage = self.stages[self.stage_index]
        for key in (
            "speed",
            "yaw",
            "chip_angle",
            "speed_tolerance_fraction",
            "direction_tolerance",
            "elevation_tolerance",
            "urgency",
        ):
            if key in stage:
                setattr(cfg.ranges, key, stage[key])
        spawn_cfg = env.event_manager.get_term_cfg(self.spawn_event_name)
        if "spawn_distance" in stage:
            spawn_cfg.params["distance_range"] = stage["spawn_distance"]
        if "ball_speed" in stage:
            spawn_cfg.params["speed_range"] = stage["ball_speed"]

    def __call__(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice, **_params
    ) -> dict[str, torch.Tensor]:
        command = cast(KickCommand, env.command_manager.get_term(self.command_name))
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

        # Curriculum runs before reset and command resampling. Ignore the initial
        # reset, whose episode length is zero, and account for every completed
        # episode in its initial turn-demand bin.
        valid_ids = env_ids[
            (env.episode_length_buf[env_ids] > 0) & command.episode_active[env_ids]
        ]
        for bin_index in range(3):
            in_bin = valid_ids[command.turn_bin[valid_ids] == bin_index]
            self.completed[bin_index] += in_bin.numel()
            if in_bin.numel() > 0:
                self.successes[bin_index] += command.stable_success[in_bin].sum()

        stage = self.stages[self.stage_index]
        minimum = int(stage.get("min_episodes_per_bin", 500))
        required_rate = float(stage.get("promotion_success_rate", 0.8))
        rates = self.successes.float() / self.completed.clamp_min(1).float()
        may_promote = self.stage_index + 1 < len(self.stages)
        if (
            may_promote
            and bool(torch.all(self.completed >= minimum))
            and bool(torch.all(rates >= required_rate))
        ):
            self.stage_index += 1
            self.completed.zero_()
            self.successes.zero_()
            rates.zero_()

        self._apply_stage(env)
        current = self.stages[self.stage_index]
        return {
            "stage": torch.tensor(self.stage_index, device=env.device),
            "success_forward": rates[0],
            "success_side": rates[1],
            "success_reverse": rates[2],
            "episodes_min_bin": self.completed.min(),
            "spawn_distance_max": torch.tensor(
                current.get("spawn_distance", (0.0, 0.0))[1], device=env.device
            ),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "completed": self.completed.cpu(),
            "successes": self.successes.cpu(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.stage_index = min(int(state["stage_index"]), len(self.stages) - 1)
        self.completed.copy_(
            torch.as_tensor(state["completed"], device=self.completed.device)
        )
        self.successes.copy_(
            torch.as_tensor(state["successes"], device=self.successes.device)
        )


def export_kick_curriculum_state(env: ManagerBasedRlEnv) -> dict[str, object] | None:
    term = getattr(env, "_kick_curriculum_term", None)
    return term.state_dict() if term is not None else None


def restore_kick_curriculum_state(
    env: ManagerBasedRlEnv, state: dict[str, object] | None
) -> None:
    term = getattr(env, "_kick_curriculum_term", None)
    if term is not None and state is not None:
        term.load_state_dict(state)
        term._apply_stage(env)
