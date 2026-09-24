"""Harvest believable robot states from a running walk policy for kick-episode seeding.

Wraps mjlab's play script exactly like ``scripts/play.py`` does (same
monkey-patch-the-viewer technique), so checkpoint loading, wandb resolution and
policy inference all go through the *exact* codepath you already use for
``uv run play Mjlab-Velocity-... --wandb-run-path ...``. This script never touches
those internals directly -- it only observes sim state after each step the base
viewer already takes.

Usage (mirrors ``uv run play``, plus a few extra flags this module pops before
handing off)::

    uv run collect-walk-states Mjlab-Velocity-Flat-Amp-DA-Muon-Booster-K1 \\
        --wandb-run-path your-org/mjlab/run-id \\
        --collect-out logs/kick/walk_state_pool.pt \\
        --collect-num-states 20000 \\
        --env.scene.num-envs 2048

First run with ``--collect-inspect-api`` to verify the assumed ``Entity.data``
attribute names against your installed mjlab version before trusting a full
collection run (see the ASSUMPTION comments below) -- this repo has no network
access to mjlab's source, so these names are inferred from sibling read/write
call sites in ``mdp/events.py``, not verified directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import mjlab.scripts.play as mjlab_play
from mjlab.viewer.viser.viewer import ViserPlayViewer


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


def _pop_bool_flag(argv: list[str], flag: str) -> bool:
    for i, tok in enumerate(argv):
        if tok == flag:
            del argv[i]
            return True
    return False


def _read_root_state_w(data) -> torch.Tensor:
    """Combined (num_envs, 13) [pos(3), quat_wxyz(4), lin_vel(3), ang_vel(3)].

    Verified against mjlab/src/mjlab/entity/data.py's EntityData: root_link_pos_w,
    root_link_quat_w, root_link_lin_vel_w, root_link_ang_vel_w are all world-frame
    properties on Entity.data.
    """
    return torch.cat(
        [
            data.root_link_pos_w,
            data.root_link_quat_w,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_w,
        ],
        dim=-1,
    )


@dataclass
class _CollectConfig:
    out_path: Path
    num_states: int
    min_interval_s: float
    max_interval_s: float
    asset_name: str
    inspect_api: bool


class StateHarvestingViserPlayViewer(ViserPlayViewer):
    """Snapshots robot rigid-body state at randomized intervals while playing.

    State layout matches ``mdp/events.py``'s ``reset_from_pose_pool`` pool rows so
    the saved file can seed an equivalent ``reset_from_state_pool`` event later:
    ``[root_pos_rel(3), root_quat_wxyz(4), joint_pos(nj), root_lin_vel(3),
    root_ang_vel(3), joint_vel(nj)]``, all in the entity's local (env-origin
    relative) frame.
    """

    _collect_cfg: _CollectConfig

    def __init__(self, *args, collect_cfg: _CollectConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._collect_cfg = collect_cfg
        self._rows: list[torch.Tensor] = []
        self._next_snapshot_step: torch.Tensor | None = None
        self._did_inspect = False
        self._num_joints: int | None = None
        self._saved = False

    def _asset(self):
        env = self.env.unwrapped
        return env.scene[self._collect_cfg.asset_name], env

    def _inspect_api_once(self) -> None:
        if self._did_inspect:
            return
        self._did_inspect = True
        asset, _ = self._asset()
        names = sorted(n for n in dir(asset.data) if not n.startswith("_"))
        print("=" * 72)
        print(f"[collect_walk_states] Entity.data attributes for '{asset.__class__.__name__}':")
        for n in names:
            print(f"  {n}")
        print("=" * 72)
        print(
            "[collect_walk_states] Confirm the root-pose/velocity attribute names "
            "used below (search for 'ASSUMPTION' in collect_walk_states.py) match "
            "one of the names printed above, then rerun without --collect-inspect-api."
        )

    def _sample_intervals(self, num_envs: int, device: torch.device) -> torch.Tensor:
        cfg = self._collect_cfg
        return torch.randint(
            low=max(1, int(cfg.min_interval_s / self._step_dt_safe())),
            high=max(2, int(cfg.max_interval_s / self._step_dt_safe()) + 1),
            size=(num_envs,),
            device=device,
        )

    def _step_dt_safe(self) -> float:
        env = self.env.unwrapped
        dt = float(getattr(env, "step_dt", 0.0) or 0.0)
        return dt if dt > 0 else 0.02

    def _snapshot(self, env_ids: torch.Tensor) -> None:
        asset, env = self._asset()
        data = asset.data

        root_state = _read_root_state_w(data)[env_ids]  # (n, 13)
        joint_pos = data.joint_pos[env_ids]
        joint_vel = data.joint_vel[env_ids]

        if self._num_joints is None:
            self._num_joints = joint_pos.shape[-1]

        env_origins = env.scene.env_origins[env_ids]
        root_pos_rel = root_state[:, 0:3] - env_origins
        root_quat = root_state[:, 3:7]
        root_lin_vel = root_state[:, 7:10]
        root_ang_vel = root_state[:, 10:13]

        row = torch.cat(
            [root_pos_rel, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel],
            dim=-1,
        )
        self._rows.append(row.detach().cpu())

    def _save(self) -> None:
        if self._saved or not self._rows:
            return
        self._saved = True
        pool = torch.cat(self._rows, dim=0)
        cfg = self._collect_cfg
        cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pool": pool,  # (N, 3+4+nj+3+3+nj)
                "num_joints": self._num_joints,
                "layout": (
                    "root_pos_rel(3), root_quat_wxyz(4), joint_pos(nj), "
                    "root_lin_vel(3), root_ang_vel(3), joint_vel(nj)"
                ),
                "asset_name": cfg.asset_name,
            },
            cfg.out_path,
        )
        print(
            f"[collect_walk_states] Saved {pool.shape[0]} states "
            f"({self._num_joints} joints) to {cfg.out_path}"
        )

    def tick(self) -> bool:  # noqa: D102 - overrides base viewer loop hook.
        result = super().tick()

        if self._collect_cfg.inspect_api:
            self._inspect_api_once()
            return result

        asset, env = self._asset()
        num_envs = asset.data.joint_pos.shape[0]
        device = asset.data.joint_pos.device

        if self._next_snapshot_step is None:
            self._next_snapshot_step = self._sample_intervals(num_envs, device)

        step_count = torch.full(
            (num_envs,), self._step_count, device=device, dtype=torch.long
        )
        due = step_count >= self._next_snapshot_step
        env_ids = due.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self._snapshot(env_ids)
            self._next_snapshot_step[env_ids] = (
                step_count[env_ids] + self._sample_intervals(env_ids.numel(), device)
            )

        total = sum(r.shape[0] for r in self._rows)
        if total >= self._collect_cfg.num_states:
            self._save()
            return False  # Stop the viewer loop.

        return result


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(
            "booster_mjlab collect-walk-states adds:\n"
            "  --collect-out PATH          output .pt state-pool file (required)\n"
            "  --collect-num-states N      stop after collecting N states (default 20000)\n"
            "  --collect-min-interval S    min seconds between snapshots per env "
            "(default 0.5)\n"
            "  --collect-max-interval S    max seconds between snapshots per env "
            "(default 2.0)\n"
            "  --collect-asset-name NAME   scene entity to snapshot (default 'robot')\n"
            "  --collect-inspect-api       print Entity.data attributes and exit "
            "instead of collecting (run this first)\n",
            file=sys.stderr,
        )

    out_path_str = _pop_flag(argv, "--collect-out")
    num_states_str = _pop_flag(argv, "--collect-num-states")
    min_interval_str = _pop_flag(argv, "--collect-min-interval")
    max_interval_str = _pop_flag(argv, "--collect-max-interval")
    asset_name = _pop_flag(argv, "--collect-asset-name") or "robot"
    inspect_api = _pop_bool_flag(argv, "--collect-inspect-api")

    if not inspect_api and not out_path_str:
        raise SystemExit("--collect-out PATH is required (or pass --collect-inspect-api)")

    collect_cfg = _CollectConfig(
        out_path=Path(out_path_str) if out_path_str else Path("/dev/null"),
        num_states=int(num_states_str) if num_states_str else 20_000,
        min_interval_s=float(min_interval_str) if min_interval_str else 0.5,
        max_interval_s=float(max_interval_str) if max_interval_str else 2.0,
        asset_name=asset_name,
        inspect_api=inspect_api,
    )

    sys.argv = [sys.argv[0], *argv]

    def _make_viewer(*args, **kwargs):
        return StateHarvestingViserPlayViewer(*args, collect_cfg=collect_cfg, **kwargs)

    mjlab_play.ViserPlayViewer = _make_viewer  # type: ignore[assignment]

    mjlab_play.main()


if __name__ == "__main__":
    main()
