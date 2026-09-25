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

from booster_mjlab.tasks.kick import mdp as kick_mdp
from booster_mjlab.tasks.kick.mdp.commands import KickCommandCfg


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
        "ball_state": ObservationTermCfg(
            func=kick_mdp.noisy_ball_state_relative_b,
            params={
                "pos_noise_std": 0.07,
                "vel_noise_std": 0.4,
                "latency_steps": 2,
                "dropout_prob": 0.05,
                "enable_noise": True,
            },
        ),
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
        "ball_state": ObservationTermCfg(func=kick_mdp.clean_ball_state_relative_b),
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
            resampling_time_range=(
                1.0e9,
                1.0e9,
            ),  # Effectively: resample on reset only.
            debug_vis=True,
            ranges=KickCommandCfg.Ranges(
                # Curriculum widens these over training; see kick_envelope below.
                speed=(1.5, 2.5),
                yaw=(-math.pi, math.pi),
                chip_angle=(0.0, 0.0),
                speed_tolerance_fraction=(0.30, 0.50),
                direction_tolerance=(math.radians(30.0), math.radians(60.0)),
                elevation_tolerance=(math.radians(8.0), math.radians(15.0)),
                urgency=(0.2, 0.8),
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
                "distance_range": (0.5, 1.0),
                "bearing_range": (-math.pi, math.pi),
                "speed_range": (0.0, 0.0),
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
        "directional_approach": RewardTermCfg(
            func=kick_mdp.directional_approach_progress,
            weight=1.5,
            params={
                "command_name": "kick",
                "staging_distance": 0.45,
                "clearance_radius": 0.35,
                "staging_threshold": 0.22,
                "heading_threshold": math.radians(35.0),
            },
        ),
        "facing_ball": RewardTermCfg(
            func=kick_mdp.facing_ball,
            weight=0.25,
            params={"command_name": "kick", "std": math.radians(25.0)},
        ),
        "useful_contact": RewardTermCfg(
            func=kick_mdp.useful_contact,
            weight=1.0,
            params={"command_name": "kick"},
        ),
        "launch_quality": RewardTermCfg(
            func=kick_mdp.launch_quality,
            weight=2.0,
            params={"command_name": "kick"},
        ),
        "accepted_launch": RewardTermCfg(
            func=kick_mdp.accepted_launch,
            weight=7.0,
            params={"command_name": "kick"},
        ),
        "stable_completion": RewardTermCfg(
            func=kick_mdp.stable_completion,
            weight=8.0,
            params={"command_name": "kick"},
        ),
        "time_penalty": RewardTermCfg(
            func=kick_mdp.time_penalty,
            weight=-0.1,
            params={"command_name": "kick"},
        ),
        "upright": RewardTermCfg(
            func=velocity_mdp.upright,
            weight=0.5,
            params={
                "std": math.sqrt(0.30),
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=velocity_mdp.body_angular_velocity_penalty,
            weight=-0.03,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per robot.
            },
        ),
        "upper_body_posture": RewardTermCfg(
            func=velocity_mdp.posture,
            weight=0.25,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*"),
                ),
                "std": {
                    r"Head_.*": 0.08,
                    r".*_Shoulder_.*": 0.30,
                    r".*_Elbow_.*": 0.35,
                },
            },
        ),
        "upper_body_joint_vel": RewardTermCfg(
            func=velocity_mdp.joint_vel_l2,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(r"Head_.*", r".*_Shoulder_.*", r".*_Elbow_.*"),
                )
            },
        ),
        "failure": RewardTermCfg(func=kick_mdp.failure_event, weight=-10.0),
        "dof_pos_limits": RewardTermCfg(
            func=velocity_mdp.joint_pos_limits, weight=-0.5
        ),
        "action_rate_l2": RewardTermCfg(func=velocity_mdp.action_rate_l2, weight=-0.10),
        "action_acc_l2": RewardTermCfg(func=velocity_mdp.action_acc_l2, weight=-0.02),
        "self_collisions": RewardTermCfg(
            func=velocity_mdp.self_collision_cost,
            weight=-0.5,
            params={"sensor_name": "self_collision"},
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=velocity_mdp.bad_orientation,
            params={"limit_angle": math.radians(50.0)},
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
                "recovery_steps": 50,
                "recovery_max_tilt": math.radians(20.0),
                "recovery_max_ang_vel": 1.5,
            },
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        # Advance only after forward, side and reverse stable-success bins pass.
        "kick_envelope": CurriculumTermCfg(
            func=kick_mdp.kick_envelope,
            params={
                "command_name": "kick",
                "spawn_event_name": "spawn_ball",
                "stages": [
                    {
                        "speed": (1.5, 2.5),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, 0.0),
                        "speed_tolerance_fraction": (0.30, 0.50),
                        "direction_tolerance": (
                            math.radians(30.0),
                            math.radians(60.0),
                        ),
                        "spawn_distance": (0.5, 1.0),
                        "ball_speed": (0.0, 0.0),
                        "min_episodes_per_bin": 500,
                        "promotion_success_rate": 0.80,
                    },
                    {
                        "speed": (1.5, 3.0),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, 0.0),
                        "speed_tolerance_fraction": (0.15, 0.50),
                        "direction_tolerance": (
                            math.radians(10.0),
                            math.radians(60.0),
                        ),
                        "spawn_distance": (1.0, 2.0),
                        "ball_speed": (0.0, 0.3),
                        "min_episodes_per_bin": 500,
                        "promotion_success_rate": 0.80,
                    },
                    {
                        "speed": (1.5, 4.0),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, math.radians(10.0)),
                        "speed_tolerance_fraction": (0.15, 0.50),
                        "direction_tolerance": (
                            math.radians(10.0),
                            math.radians(60.0),
                        ),
                        "spawn_distance": (1.0, 3.0),
                        "ball_speed": (0.0, 0.8),
                        "min_episodes_per_bin": 500,
                        "promotion_success_rate": 0.80,
                    },
                    {
                        "speed": (1.5, 4.0),
                        "yaw": (-math.pi, math.pi),
                        "chip_angle": (0.0, math.radians(20.0)),
                        "speed_tolerance_fraction": (0.15, 0.50),
                        "direction_tolerance": (
                            math.radians(10.0),
                            math.radians(60.0),
                        ),
                        "spawn_distance": (1.5, 4.0),
                        "ball_speed": (0.0, 1.5),
                        "min_episodes_per_bin": 500,
                        "promotion_success_rate": 0.80,
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
                cone="elliptic",
            ),
        ),
        decimation=4,
        episode_length_s=10.0,
    )
