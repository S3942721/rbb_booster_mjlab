"""Kick task MDP terms."""

from booster_mjlab.tasks.kick.mdp.commands import KickCommand, KickCommandCfg
from booster_mjlab.tasks.kick.mdp.curriculums import (
    export_kick_curriculum_state,
    kick_envelope,
    restore_kick_curriculum_state,
)
from booster_mjlab.tasks.kick.mdp.events import (
    reset_from_state_pool,
    spawn_ball_relative_to_robot,
)
from booster_mjlab.tasks.kick.mdp.observations import (
    ball_state_relative_b,
    clean_ball_state_relative_b,
    noisy_ball_state_relative_b,
)
from booster_mjlab.tasks.kick.mdp.rewards import (
    accepted_launch,
    directional_approach_progress,
    facing_ball,
    failure_event,
    kick_outcome,
    launch_quality,
    stable_completion,
    time_penalty,
    useful_contact,
)
from booster_mjlab.tasks.kick.mdp.terminations import kick_complete

__all__ = (
    "KickCommand",
    "KickCommandCfg",
    "kick_envelope",
    "export_kick_curriculum_state",
    "restore_kick_curriculum_state",
    "reset_from_state_pool",
    "spawn_ball_relative_to_robot",
    "ball_state_relative_b",
    "clean_ball_state_relative_b",
    "noisy_ball_state_relative_b",
    "directional_approach_progress",
    "facing_ball",
    "useful_contact",
    "launch_quality",
    "accepted_launch",
    "stable_completion",
    "failure_event",
    "kick_outcome",
    "time_penalty",
    "kick_complete",
)
