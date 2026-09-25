"""Kick task event terms: seeding episodes from harvested walk states + ball spawn."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.logging import print_info

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


class reset_from_state_pool:
    """Seed episodes from a state pool harvested by ``scripts/collect_walk_states.py``.

    Row layout must match that script's output exactly:
    ``[root_pos_rel(3), root_quat_wxyz(4), joint_pos(nj), root_lin_vel(3),
    root_ang_vel(3), joint_vel(nj)]``. Unlike ``mdp/events.py``'s
    ``reset_from_pose_pool``. Pools are validated again when loaded so old raw
    collections cannot silently seed fallen or schema-incompatible states.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        self._asset_cfg: SceneEntityCfg = params.get("asset_cfg", _DEFAULT_ROBOT_CFG)
        self._pool_path = Path(params["pool_path"])
        self._device = env.device
        self._pool: torch.Tensor | None = None
        self._pool_size = 0
        self._num_joints = 0
        self._metadata: dict[str, object] = {}

    def __call__(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **params
    ) -> None:
        if self._pool is None:
            self._load_pool()
        self._sample_from_pool(env, env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        pass

    def _load_pool(self) -> None:
        payload = torch.load(
            self._pool_path, map_location=self._device, weights_only=True
        )
        pool = payload["pool"].to(self._device)
        self._num_joints = int(payload["num_joints"])
        self._metadata = {key: value for key, value in payload.items() if key != "pool"}
        expected_columns = 13 + 2 * self._num_joints
        if pool.ndim != 2 or pool.shape[1] != expected_columns:
            raise ValueError(
                f"Invalid walk state pool shape {tuple(pool.shape)}; "
                f"expected (*, {expected_columns})"
            )
        finite = torch.isfinite(pool).all(dim=-1)
        quat = pool[:, 3:7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1)
        w, x, y, z = quat.unbind(dim=-1)
        tilt = torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))
        valid = (
            finite
            & (torch.abs(quat_norm - 1.0) < 1.0e-3)
            & (pool[:, 2] >= 0.35)
            & (tilt <= math.radians(45.0))
        )
        rejected = int((~valid).sum().item())
        self._pool = pool[valid]
        self._pool_size = self._pool.shape[0]
        if self._pool_size == 0:
            raise ValueError(f"No valid states remain in {self._pool_path}")
        print_info(
            f"[reset_from_state_pool] Loaded {self._pool_size} states "
            f"({self._num_joints} joints, rejected {rejected}) from {self._pool_path}"
        )

    def _sample_from_pool(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device, dtype=torch.int)

        asset = env.scene[self._asset_cfg.name]
        if len(asset.joint_names) != self._num_joints:
            raise ValueError(
                f"State pool has {self._num_joints} joints but "
                f"'{self._asset_cfg.name}' has {len(asset.joint_names)}"
            )
        pool_joint_names = self._metadata.get("joint_names")
        if pool_joint_names is not None and tuple(pool_joint_names) != tuple(
            asset.joint_names
        ):
            raise ValueError(
                "State-pool joint order does not match the configured robot: "
                f"pool={tuple(pool_joint_names)}, robot={tuple(asset.joint_names)}"
            )
        num_reset = len(env_ids)

        assert self._pool is not None
        pool_indices = torch.randint(
            0, self._pool_size, (num_reset,), device=self._device
        )
        sampled = self._pool[pool_indices]

        nj = self._num_joints
        root_pos_rel = sampled[:, 0:3]
        root_quat = sampled[:, 3:7]
        joint_pos = sampled[:, 7 : 7 + nj]
        root_lin_vel = sampled[:, 7 + nj : 7 + nj + 3]
        root_ang_vel = sampled[:, 7 + nj + 3 : 7 + nj + 6]
        joint_vel = sampled[:, 7 + nj + 6 : 7 + nj + 6 + nj]

        positions = root_pos_rel + env.scene.env_origins[env_ids]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_link_velocity_to_sim(
            torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
        )
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if not hasattr(env, "_kick_reset_root_pose_w"):
            env._kick_reset_root_pose_w = torch.zeros(
                env.num_envs, 7, device=self._device
            )
        env._kick_reset_root_pose_w[env_ids] = root_pose


def spawn_ball_relative_to_robot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    distance_range: tuple[float, float] = (1.5, 4.0),
    bearing_range: tuple[float, float] = (-math.pi, math.pi),
    speed_range: tuple[float, float] = (0.0, 1.5),
    ball_heading_noise_range: tuple[float, float] = (-0.3, 0.3),
    ball_height: float = 0.11,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Place the ball a randomized distance/bearing from the robot, optionally rolling.

    ``bearing_range`` is relative to the robot's current heading (0 = directly
    ahead), so the required approach/circling behaviour varies episode to episode.
    Robot position/heading come from the reset event's cached sampled pose rather
    than derived ``xpos/xquat``, which are stale until the post-reset forward pass.
    If a coupled ball randomizer provides per-env radii, spawn height and rolling
    angular velocity use them.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    device = env.device

    ball = env.scene[ball_cfg.name]

    r = torch.empty(n, device=device)
    distance = r.uniform_(*distance_range)
    bearing_offset = torch.empty(n, device=device).uniform_(*bearing_range)
    ball_heading_noise = torch.empty(n, device=device).uniform_(
        *ball_heading_noise_range
    )
    speed = torch.empty(n, device=device).uniform_(*speed_range)

    reset_pose = getattr(env, "_kick_reset_root_pose_w", None)
    if reset_pose is None:
        raise RuntimeError(
            "spawn_ball_relative_to_robot requires reset_from_state_pool to run first"
        )
    quat = reset_pose[env_ids, 3:7]
    w, x, y, z = quat.unbind(dim=-1)
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    bearing = heading + bearing_offset
    robot_pos = reset_pose[env_ids, :3]

    ball_pos = robot_pos.clone()
    ball_pos[:, 0] += distance * torch.cos(bearing)
    ball_pos[:, 1] += distance * torch.sin(bearing)
    radii = getattr(env, "_kick_ball_radius", None)
    radius = (
        radii[env_ids]
        if radii is not None
        else torch.full((n,), ball_height, device=device)
    )
    ball_pos[:, 2] = radius

    # Ball rolls roughly back toward the robot (plus noise), not away from it.
    roll_heading = bearing + math.pi + ball_heading_noise
    ball_lin_vel = torch.zeros(n, 3, device=device)
    ball_lin_vel[:, 0] = speed * torch.cos(roll_heading)
    ball_lin_vel[:, 1] = speed * torch.sin(roll_heading)

    ball_ang_vel = torch.zeros(n, 3, device=device)
    ball_ang_vel[:, 0] = ball_lin_vel[:, 1] / radius
    ball_ang_vel[:, 1] = -ball_lin_vel[:, 0] / radius

    identity_quat = torch.zeros(n, 4, device=device)
    identity_quat[:, 0] = 1.0

    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, identity_quat], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.cat([ball_lin_vel, ball_ang_vel], dim=-1),
        env_ids=env_ids,
    )

    if not hasattr(env, "_kick_reset_ball_pos_w"):
        env._kick_reset_ball_pos_w = torch.zeros(env.num_envs, 3, device=device)
    env._kick_reset_ball_pos_w[env_ids] = ball_pos
