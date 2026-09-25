"""Headless, fixed-stage evaluation for the Booster K1 approach-and-kick task.

This evaluator preserves the training horizon and failure terminations, disables
automatic reset so terminal kick state can be recorded, and writes one CSV row
per completed episode plus a JSON summary. Run clean and noisy perception as
separate evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends

from booster_mjlab.tasks.kick.mdp.commands import KickCommand, KickCommandCfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", nargs="?", default="Mjlab-Kick-Booster-K1")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint-file", type=Path)
    checkpoint.add_argument("--wandb-run-path")
    parser.add_argument("--wandb-checkpoint-name")
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--perception", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/kick/evaluation"))
    return parser.parse_args()


def _fix_curriculum_stage(env_cfg: Any, stage_index: int) -> None:
    term_cfg = env_cfg.curriculum["kick_envelope"]
    stages = term_cfg.params["stages"]
    if not 0 <= stage_index < len(stages):
        raise ValueError(f"stage must be in [0, {len(stages) - 1}]")
    stage = stages[stage_index]
    command_cfg = cast(KickCommandCfg, env_cfg.commands["kick"])
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
            setattr(command_cfg.ranges, key, stage[key])
    spawn_cfg = env_cfg.events[term_cfg.params.get("spawn_event_name", "spawn_ball")]
    if "spawn_distance" in stage:
        spawn_cfg.params["distance_range"] = stage["spawn_distance"]
    if "ball_speed" in stage:
        spawn_cfg.params["speed_range"] = stage["ball_speed"]
    env_cfg.curriculum = {}


def _scalar(value: torch.Tensor, index: int) -> float:
    return float(value[index].detach().cpu().item())


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"episodes": len(rows)}
    for field in (
        "contacted",
        "measured_launch",
        "accepted_launch",
        "stable_success",
        "fell_over",
        "illegal_contact",
        "mis_kick",
        "time_out",
    ):
        result[f"{field}_rate"] = sum(int(row[field]) for row in rows) / len(rows)
    result["mean_duration_s"] = sum(row["duration_s"] for row in rows) / len(rows)
    result["mean_return"] = sum(row["return"] for row in rows) / len(rows)
    names = ("forward", "side", "reverse")
    result["turn_bins"] = {}
    for index, name in enumerate(names):
        subset = [row for row in rows if row["turn_bin"] == index]
        result["turn_bins"][name] = {
            "episodes": len(subset),
            "stable_success_rate": (
                sum(int(row["stable_success"]) for row in subset) / len(subset)
                if subset
                else None
            ),
        }
    return result


def main() -> None:
    args = _parse_args()
    import mjlab.tasks  # noqa: F401

    if args.task_id not in list_tasks():
        raise SystemExit(
            f"Unknown task '{args.task_id}'. Run list_envs to see options."
        )
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num-envs must be positive")

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task_id)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    _fix_curriculum_stage(env_cfg, args.stage)
    actor_cfg = env_cfg.observations["actor"]
    if args.perception == "clean":
        actor_cfg.enable_corruption = False
        actor_cfg.terms["ball_state"].params["enable_noise"] = False

    if args.checkpoint_file is not None:
        resume_path = args.checkpoint_file.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
    else:
        log_root = (Path(args.log_root) / agent_cfg.experiment_name).resolve()
        resume_path, _ = get_wandb_checkpoint_path(
            log_root, Path(args.wandb_run_path), args.wandb_checkpoint_name
        )

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    returns = torch.zeros(args.num_envs, device=device)
    lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    rows: list[dict[str, Any]] = []
    obs = env.get_observations()

    try:
        with torch.inference_mode():
            while len(rows) < args.episodes:
                obs, reward, dones, _ = env.step(policy(obs))
                returns += reward
                lengths += 1
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() == 0:
                    continue

                command = cast(KickCommand, raw_env.command_manager.get_term("kick"))
                terminal_terms = {
                    name: raw_env.termination_manager.get_term(name)
                    for name in raw_env.termination_manager.active_terms
                }
                remaining = args.episodes - len(rows)
                for env_id in done_ids[:remaining].tolist():
                    rows.append(
                        {
                            "episode": len(rows),
                            "env_id": env_id,
                            "turn_bin": int(command.turn_bin[env_id].item()),
                            "duration_s": _scalar(
                                lengths.float() * raw_env.step_dt, env_id
                            ),
                            "return": _scalar(returns, env_id),
                            "contacted": bool(command.has_contacted[env_id].item()),
                            "measured_launch": bool(command.has_kicked[env_id].item()),
                            "accepted_launch": bool(
                                command.accepted_launch[env_id].item()
                            ),
                            "stable_success": bool(
                                command.stable_success[env_id].item()
                            ),
                            "speed_error": _scalar(
                                command.metrics["speed_error"], env_id
                            ),
                            "direction_error_rad": _scalar(
                                command.metrics["direction_error"], env_id
                            ),
                            "elevation_error_rad": _scalar(
                                command.metrics["elevation_error"], env_id
                            ),
                            **{
                                name: bool(values[env_id].item())
                                for name, values in terminal_terms.items()
                            },
                        }
                    )

                raw_env.reset(env_ids=done_ids)
                returns[done_ids] = 0.0
                lengths[done_ids] = 0
                obs = env.get_observations()
                if len(rows) % max(args.num_envs, 256) < done_ids.numel():
                    print(f"[evaluate_kick] {len(rows)}/{args.episodes} episodes")
    finally:
        env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stage{args.stage}_{args.perception}_seed{args.seed}"
    csv_path = args.output_dir / f"episodes_{stem}.csv"
    json_path = args.output_dir / f"summary_{stem}.json"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "task_id": args.task_id,
        "checkpoint": str(resume_path),
        "stage": args.stage,
        "perception": args.perception,
        "seed": args.seed,
        **_summary(rows),
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[evaluate_kick] wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
