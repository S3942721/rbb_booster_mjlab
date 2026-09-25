"""Thin wrapper around mjlab's play script.

Swaps the viser viewer for :class:`RecordingViserPlayViewer`, which
adds a record / stop-recording button that captures the browser view to an mp4
under ``--record-dir``. Frames are timed by the sim clock, so the video plays
back in real time even when the machine can't step the policy at full speed.
"""

import functools
import math
import sys
from typing import Any, cast

import mjlab.scripts.play as mjlab_play

from booster_mjlab.tasks.kick.mdp.commands import KickCommandCfg
from booster_mjlab.viewer import DEFAULT_VIDEO_DIR, RecordingViserPlayViewer


_KICK_SCENARIOS: dict[str, dict[str, float]] = {
    # This is deliberately identical to the stage-0 command distribution except
    # that each value is fixed, making a replay comparable across checkpoints.
    "stage0": {
        "speed": 2.0,
        "yaw_deg": 0.0,
        "chip_deg": 0.0,
        "speed_tolerance": 0.40,
        "direction_tolerance_deg": 45.0,
        "elevation_tolerance_deg": 12.0,
        "urgency": 0.5,
        "distance": 0.75,
        "bearing_deg": 0.0,
    },
    "precision-pass": {
        "speed": 2.0,
        "yaw_deg": 0.0,
        "chip_deg": 0.0,
        "speed_tolerance": 0.10,
        "direction_tolerance_deg": 5.0,
        "elevation_tolerance_deg": 3.0,
        "urgency": 0.1,
        "distance": 1.0,
        "bearing_deg": 0.0,
    },
    "wide-clearance": {
        "speed": 3.5,
        "yaw_deg": 0.0,
        "chip_deg": 0.0,
        "speed_tolerance": 0.50,
        "direction_tolerance_deg": 60.0,
        "elevation_tolerance_deg": 15.0,
        "urgency": 1.0,
        "distance": 1.5,
        "bearing_deg": 0.0,
    },
    "chip": {
        "speed": 2.5,
        "yaw_deg": 0.0,
        "chip_deg": 15.0,
        "speed_tolerance": 0.25,
        "direction_tolerance_deg": 12.0,
        "elevation_tolerance_deg": 6.0,
        "urgency": 0.3,
        "distance": 1.5,
        "bearing_deg": 0.0,
    },
    "reverse": {
        "speed": 2.0,
        "yaw_deg": 180.0,
        "chip_deg": 0.0,
        "speed_tolerance": 0.35,
        "direction_tolerance_deg": 30.0,
        "elevation_tolerance_deg": 10.0,
        "urgency": 0.5,
        "distance": 1.0,
        "bearing_deg": 0.0,
    },
    "soft-touch": {
        "speed": 0.8,
        "yaw_deg": 0.0,
        "chip_deg": 0.0,
        "speed_tolerance": 0.20,
        "direction_tolerance_deg": 15.0,
        "elevation_tolerance_deg": 5.0,
        "urgency": 0.2,
        "distance": 0.5,
        "bearing_deg": 0.0,
    },
}


def _pop_flag(argv: list[str], flag: str) -> str | None:
    """Remove ``--flag VALUE`` or ``--flag=VALUE`` from argv; return VALUE."""
    for i, tok in enumerate(argv):
        if tok == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value")
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if tok.startswith(flag + "="):
            value = tok.split("=", 1)[1]
            del argv[i]
            return value
    return None


def _pop_float(argv: list[str], flag: str) -> float | None:
    value = _pop_flag(argv, flag)
    return float(value) if value is not None else None


def _set_singleton_range(cfg: KickCommandCfg, name: str, value: float) -> None:
    setattr(cfg.ranges, name, (value, value))


def _configure_kick_scenario(
    env_cfg: Any,
    scenario: str | None,
    overrides: dict[str, float | None],
) -> None:
    """Apply a reproducible kick command and spawn pose before play starts."""
    if scenario is None and not any(value is not None for value in overrides.values()):
        return
    if "kick" not in env_cfg.commands:
        raise ValueError("--kick-* options are available only for kick tasks")
    if scenario is not None and scenario not in _KICK_SCENARIOS:
        choices = ", ".join(_KICK_SCENARIOS)
        raise ValueError(
            f"Unknown kick scenario {scenario!r}; choose one of: {choices}"
        )

    values = dict(_KICK_SCENARIOS[scenario or "stage0"])
    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    command_cfg = cast(KickCommandCfg, env_cfg.commands["kick"])
    _set_singleton_range(command_cfg, "speed", values["speed"])
    _set_singleton_range(command_cfg, "yaw", math.radians(values["yaw_deg"]))
    _set_singleton_range(command_cfg, "chip_angle", math.radians(values["chip_deg"]))
    _set_singleton_range(
        command_cfg, "speed_tolerance_fraction", values["speed_tolerance"]
    )
    _set_singleton_range(
        command_cfg,
        "direction_tolerance",
        math.radians(values["direction_tolerance_deg"]),
    )
    _set_singleton_range(
        command_cfg,
        "elevation_tolerance",
        math.radians(values["elevation_tolerance_deg"]),
    )
    _set_singleton_range(command_cfg, "urgency", values["urgency"])

    spawn_cfg = env_cfg.events["spawn_ball"]
    spawn_cfg.params["distance_range"] = (values["distance"], values["distance"])
    bearing = math.radians(values["bearing_deg"])
    spawn_cfg.params["bearing_range"] = (bearing, bearing)
    spawn_cfg.params["speed_range"] = (0.0, 0.0)
    print(
        "[kick play] "
        f"scenario={scenario or 'custom'} distance={values['distance']:.2f}m "
        f"bearing={values['bearing_deg']:.0f}deg launch={values['speed']:.2f}m/s "
        f"yaw={values['yaw_deg']:.0f}deg chip={values['chip_deg']:.0f}deg "
        f"cone=+/-{values['direction_tolerance_deg']:.0f}deg "
        f"speed_tol=+/-{100 * values['speed_tolerance']:.0f}% "
        f"urgency={values['urgency']:.2f}"
    )


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        # mjlab's help won't list --record-dir (we pop it before mjlab parses),
        # so surface it here.
        print(
            "booster_mjlab play adds:\n"
            "  --record-dir STR  (viser viewer: where recorded mp4s are written, "
            f"default {DEFAULT_VIDEO_DIR})\n"
            "  --kick-scenario NAME  (stage0, precision-pass, wide-clearance, "
            "chip, reverse, soft-touch)\n"
            "  --kick-distance M --kick-bearing-deg DEG --kick-speed MPS\n"
            "  --kick-yaw-deg DEG --kick-chip-deg DEG --kick-speed-tolerance FRAC\n"
            "  --kick-direction-tolerance-deg DEG --kick-elevation-tolerance-deg DEG\n"
            "  --kick-urgency VALUE\n",
            file=sys.stderr,
        )
    record_dir = _pop_flag(argv, "--record-dir")
    scenario = _pop_flag(argv, "--kick-scenario")
    overrides = {
        "distance": _pop_float(argv, "--kick-distance"),
        "bearing_deg": _pop_float(argv, "--kick-bearing-deg"),
        "speed": _pop_float(argv, "--kick-speed"),
        "yaw_deg": _pop_float(argv, "--kick-yaw-deg"),
        "chip_deg": _pop_float(argv, "--kick-chip-deg"),
        "speed_tolerance": _pop_float(argv, "--kick-speed-tolerance"),
        "direction_tolerance_deg": _pop_float(argv, "--kick-direction-tolerance-deg"),
        "elevation_tolerance_deg": _pop_float(argv, "--kick-elevation-tolerance-deg"),
        "urgency": _pop_float(argv, "--kick-urgency"),
    }
    original_load_env_cfg = mjlab_play.load_env_cfg

    def load_env_cfg_with_kick_scenario(task_id: str, *args, **kwargs):
        cfg = original_load_env_cfg(task_id, *args, **kwargs)
        _configure_kick_scenario(cfg, scenario, overrides)
        return cfg

    mjlab_play.load_env_cfg = load_env_cfg_with_kick_scenario
    sys.argv = [sys.argv[0], *argv]

    # Viser viewer with a record button.
    mjlab_play.ViserPlayViewer = functools.partial(  # type: ignore[assignment]
        RecordingViserPlayViewer,
        video_dir=record_dir or DEFAULT_VIDEO_DIR,
    )

    mjlab_play.main()


if __name__ == "__main__":
    main()
