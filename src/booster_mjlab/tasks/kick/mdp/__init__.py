"""Kick task MDP terms."""

from booster_mjlab.tasks.kick.mdp.commands import KickCommand, KickCommandCfg
from booster_mjlab.tasks.kick.mdp.curriculums import kick_envelope
from booster_mjlab.tasks.kick.mdp.events import (
    reset_from_state_pool,
    spawn_ball_relative_to_robot,
)
from booster_mjlab.tasks.kick.mdp.observations import (
    ball_state_relative_b,
    noisy_ball_state_relative_b,
)
from booster_mjlab.tasks.kick.mdp.rewards import (
    closing_velocity,
    kick_outcome,
    time_penalty,
)
from booster_mjlab.tasks.kick.mdp.terminations import kick_complete

__all__ = (
    "KickCommand",
    "KickCommandCfg",
    "kick_envelope",
    "reset_from_state_pool",
    "spawn_ball_relative_to_robot",
    "ball_state_relative_b",
    "noisy_ball_state_relative_b",
    "closing_velocity",
    "kick_outcome",
    "time_penalty",
    "kick_complete",
)
