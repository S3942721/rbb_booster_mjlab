"""Kick task base environment configuration."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from booster_mjlab.mdp.terminations import stochastic_bad_orientation
from booster_mjlab.tasks.kick import mdp as kick_mdp
from booster_mjlab.tasks.kick.mdp.commands import KickCommandCfg
from booster_mjlab.tasks.velocity.mdp.rewards import variable_upright


def make_kick_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base kick task configuration.

    Single dedicated policy (not the walk engine): the robot must approach,
    circle if needed, and strike a ball to match a commanded 3D launch vector
    (direction, power, chip height) as fast as possible. Episodes are seeded from
    real walk-policy rollout states (see ``scripts/collect_walk_states.py`` /
    ``mdp/events.py::reset_from_state_pool``) so the robot starts a few metres
    from the ball mid-gait, matching the trained-team pattern described in the
    plan. Pure task reward for v1 (no AMP style reward).
    """

    ##
    # Observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=velocity_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=velocity_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=velocity_mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=velocity_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=velocity_mdp.last_action),
        "ball_state": ObservationTermCfg(func=kick_mdp.noisy_ball_state_relative_b),
        "command": ObservationTermCfg(
            func=velocity_mdp.generated_commands,
            params={"command_name": "kick"},
        ),
    }

    critic_terms = {
        **actor_terms,
        "joint_pos": ObservationTermCfg(func=velocity_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=velocity_mdp.joint_vel_rel),
        "base_lin_vel": ObservationTermCfg(
            func=velocity_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "ball_state": ObservationTermCfg(func=kick_mdp.ball_state_relative_b),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Overridden per robot.
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "kick": KickCommandCfg(
            entity_name="robot",
            ball_entity_name="ball",
            resampling_time_range=(1.0e9, 1.0e9),  # Effectively: resample on reset only.
            debug_vis=True,
            ranges=KickCommandCfg.Ranges(
                # Curriculum widens these over training; see kick_envelope below.
                speed=(1.5, 2.5),
                yaw=(-math.pi / 2, math.pi / 2),
                chip_angle=(0.0, 0.0),
            ),
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_robot_from_walk_states": EventTermCfg(
            func=kick_mdp.reset_from_state_pool,
            mode="reset",
            params={
                # Path set per-robot in config/k1/env_cfgs.py once a pool has
                # been harvested via scripts/collect_walk_states.py.
                "pool_path": "logs/kick/walk_state_pool.pt",
            },
        ),
        "spawn_ball": EventTermCfg(
            func=kick_mdp.spawn_ball_relative_to_robot,
            mode="reset",
            params={
                # Curriculum widens this over training; see kick_envelope below.
                "distance_range": (1.0, 2.0),
                "bearing_range": (-math.pi, math.pi),
                "speed_range": (0.0, 1.5),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=velocity_mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "operation": "abs",
                "ranges": (0.75, 1.25),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=velocity_mdp.dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "pd_gains": EventTermCfg(
            mode="startup",
            func=velocity_mdp.dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "operation": "scale",
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "kick_outcome": RewardTermCfg(
            func=kick_mdp.kick_outcome,
            weight=20.0,
            params={"command_name": "kick"},
        ),
        "closing_velocity": RewardTermCfg(
            func=kick_mdp.closing_velocity,
            weight=2.0,
        ),
        "time_penalty": RewardTermCfg(
            func=kick_mdp.time_penalty,
            weight=-0.05,
        ),
        "upright": RewardTermCfg(
            func=variable_upright,
            weight=1.0,
            params={
                "command_name": "kick",
                "std_standing": math.sqrt(0.20),
                "std_walking": math.sqrt(0.30),
                "std_running": math.sqrt(0.45),
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=velocity_mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.1),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": "self_collision"},
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=stochastic_bad_orientation,
            params={"limit_angle": math.radians(63.0), "probability": 0.02},
        ),
        "illegal_contact": TerminationTermCfg(
            func=velocity_mdp.illegal_contact,
            params={"sensor_name": "non_foot_ground_contact"},
        ),
        "mis_kick": TerminationTermCfg(
            func=velocity_mdp.illegal_contact,
            params={"sensor_name": "non_foot_ball_contact"},
        ),
        "kick_complete": TerminationTermCfg(
            func=kick_mdp.kick_complete,
            params={
                "command_name": "kick",
                "contact_sensor_name": "foot_ball_contact",
                "measure_delay_steps": 3,
            },
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        # Widen the commanded speed/direction/chip envelope and spawn distance as
        # competence improves, mirroring amp/curriculums.py's stage-annealing pattern.
        "kick_envelope": CurriculumTermCfg(
            func=kick_mdp.kick_envelope,
            params={
                "command_name": "kick",
                "spawn_event_name": "spawn_ball",
                "stages": [
                    {
                        "step": 0,
                        "speed": (1.5, 2.5),
                        "yaw": (-math.pi / 2, math.pi / 2),
                        "chip_angle": (0.0, 0.0),
                        "spawn_distance": (1.0, 2.0),
                    },
                    {
                        "step": 3000 * 24,
                        "speed": (1.5, 3.0),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, math.radians(10.0)),
                        "spawn_distance": (1.0, 3.0),
                    },
                    {
                        "step": 8000 * 24,
                        "speed": (1.5, 4.0),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, math.radians(20.0)),
                        "spawn_distance": (1.5, 4.0),
                    },
                ],
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",
            distance=3.0,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=80,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=6.0,
    )
