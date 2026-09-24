"""Harvest believable robot states from a running walk policy for kick-episode seeding.

Unlike ``uv run play``, this does **not** launch the interactive viewer -- the viewer
paces itself to real time for human viewing, which is far too slow for collecting many
thousands of states (an earlier version of this script wrapped the viewer and hit
exactly that wall). Instead this reimplements just the env/checkpoint/policy loading
mjlab's own play script does (see ``mjlab/src/mjlab/scripts/play.py::run_play``) and
runs a plain step loop as fast as the GPU allows, across many parallel envs, with no
rendering at all.

Usage::

    uv run collect-walk-states Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \\
        --wandb-run-path your-org/mjlab/run-id \\
        --num-envs 4096 \\
        --collect-out logs/kick/walk_state_pool.pt \\
        --collect-num-states 20000

Progress is printed periodically; with a few thousand parallel envs this should take
seconds to low minutes rather than hours.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="Registered task id, e.g. Mjlab-Velocity-...")
    parser.add_argument("--wandb-run-path", default=None)
    parser.add_argument(
        "--wandb-checkpoint-name",
        default=None,
        help="Checkpoint name within the W&B run (e.g. 'model_4000.pt'); latest if omitted.",
    )
    parser.add_argument(
        "--checkpoint-file", default=None, help="Local checkpoint path (alternative to W&B)."
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs/rsl_rl")
    parser.add_argument("--collect-out", required=True, type=Path)
    parser.add_argument("--collect-num-states", type=int, default=20_000)
    parser.add_argument("--collect-min-interval", type=float, default=0.5)
    parser.add_argument("--collect-max-interval", type=float, default=2.0)
    parser.add_argument("--collect-asset-name", default="robot")
    parser.add_argument(
        "--collect-progress-every",
        type=float,
        default=2.0,
        help="Seconds between progress prints.",
    )
    args = parser.parse_args()
    if args.wandb_run_path is None and args.checkpoint_file is None:
        parser.error("one of --wandb-run-path or --checkpoint-file is required")
    return args


def _read_root_state_w(data) -> torch.Tensor:
    """Combined (num_envs, 13) [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)]."""
    return torch.cat(
        [
            data.root_link_pos_w,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
        ],
        dim=-1,
    )


def main() -> None:
    args = _parse_args()

    # Import tasks to populate the registry (also triggers the "mjlab.tasks" entry
    # point discovery that pulls in booster_mjlab.tasks -- see mjlab/__init__.py).
    import mjlab.tasks  # noqa: F401

    if args.task_id not in list_tasks():
        raise SystemExit(f"Unknown task '{args.task_id}'. Run list_envs to see options.")

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = args.num_envs

    log_root_path = (Path(args.log_root) / agent_cfg.experiment_name).resolve()
    if args.checkpoint_file is not None:
        resume_path = Path(args.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    else:
        resume_path, was_cached = get_wandb_checkpoint_path(
            log_root_path, Path(args.wandb_run_path), args.wandb_checkpoint_name
        )
        print(
            f"[collect_walk_states] Loading checkpoint: {resume_path.name} "
            f"({'cached' if was_cached else 'downloaded'})"
        )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    asset = env.unwrapped.scene[args.collect_asset_name]
    num_envs = env.unwrapped.num_envs
    step_dt = env.unwrapped.step_dt

    def sample_intervals(n: int) -> torch.Tensor:
        return torch.randint(
            low=max(1, int(args.collect_min_interval / step_dt)),
            high=max(2, int(args.collect_max_interval / step_dt) + 1),
            size=(n,),
            device=device,
        )

    next_snapshot_step = sample_intervals(num_envs)
    step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
    rows: list[torch.Tensor] = []
    num_joints: int | None = None

    print(
        f"[collect_walk_states] Running {num_envs} envs headlessly, target "
        f"{args.collect_num_states} states..."
    )
    start_time = time.perf_counter()
    last_print = start_time

    with torch.no_grad():
        while sum(r.shape[0] for r in rows) < args.collect_num_states:
            obs = env.get_observations()
            actions = policy(obs)
            env.step(actions)
            step_count += 1

            due = step_count >= next_snapshot_step
            env_ids = due.nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                data = asset.data
                root_state = _read_root_state_w(data)[env_ids]
                joint_pos = data.joint_pos[env_ids]
                joint_vel = data.joint_vel[env_ids]
                if num_joints is None:
                    num_joints = joint_pos.shape[-1]

                env_origins = env.unwrapped.scene.env_origins[env_ids]
                root_pos_rel = root_state[:, 0:3] - env_origins
                row = torch.cat(
                    [
                        root_pos_rel,
                        root_state[:, 3:7],
                        joint_pos,
                        root_state[:, 7:10],
                        root_state[:, 10:13],
                        joint_vel,
                    ],
                    dim=-1,
                )
                rows.append(row.detach().cpu())
                next_snapshot_step[env_ids] = (
                    step_count[env_ids] + sample_intervals(env_ids.numel())
                )

            now = time.perf_counter()
            if now - last_print >= args.collect_progress_every:
                last_print = now
                collected = sum(r.shape[0] for r in rows)
                elapsed = now - start_time
                rate = collected / elapsed if elapsed > 0 else 0.0
                print(
                    f"[collect_walk_states] {collected}/{args.collect_num_states} "
                    f"states ({rate:.0f}/s, {elapsed:.0f}s elapsed)"
                )

    pool = torch.cat(rows, dim=0)[: args.collect_num_states]
    args.collect_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pool": pool,
            "num_joints": num_joints,
            "layout": (
                "root_pos_rel(3), root_quat_wxyz(4), joint_pos(nj), "
                "root_lin_vel(3), root_ang_vel(3), joint_vel(nj)"
            ),
            "asset_name": args.collect_asset_name,
        },
        args.collect_out,
    )
    print(f"[collect_walk_states] Saved {pool.shape[0]} states to {args.collect_out}")


if __name__ == "__main__":
    main()
