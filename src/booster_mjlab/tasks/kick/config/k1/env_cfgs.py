"""Booster K1-specific kick task environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from booster_mjlab.environment.ball.ball_constants import get_ball_cfg
from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from booster_mjlab.tasks.kick.kick_env_cfg import make_kick_env_cfg


def booster_k1_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 kick task configuration."""
    cfg = make_kick_env_cfg()

    cfg.scene.entities = {
        "robot": get_k1_robot_cfg(),
        "ball": get_ball_cfg(size=5, ball_pos=(2.0, 0.0, None)),
    }

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            exclude=("left_foot_link", "right_foot_link"),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    foot_ball_cfg = ContactSensorCfg(
        name="foot_ball_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_foot_link|right_foot_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    non_foot_ball_cfg = ContactSensorCfg(
        name="non_foot_ball_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            exclude=("left_foot_link", "right_foot_link"),
        ),
        secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (
        feet_ground_cfg,
        nonfoot_ground_cfg,
        self_collision_cfg,
        foot_ball_cfg,
        non_foot_ball_cfg,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # Head control belongs to the perception/head controller at deployment. Keep
    # it out of this policy so the kick cannot learn to thrash it. Arms remain
    # available for balance and are posture-regularised by the base config.
    joint_pos_action.actuator_names = (
        r".*_Knee_Pitch",
        r".*_Hip_Yaw",
        r".*_Ankle_.*",
        r".*_Hip_Pitch",
        r".*_Hip_Roll",
        r".*_Shoulder_.*",
        r".*_Elbow_.*",
    )
    joint_pos_action.scale = {
        pattern: scale
        for pattern, scale in K1_ACTION_SCALE.items()
        if pattern != r"Head_.*"
    }

    cfg.viewer.body_name = "Trunk"

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Keep the first curriculum stages on the coherent size-5 model compiled by
    # get_ball_cfg. Radius-only geom scaling and independent pseudo-inertia
    # scaling violate the sphere inertia relation and leave friction/visuals
    # inconsistent. Reintroduce size variation only through a coupled randomizer.

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)

    # Play is also the strict visual evaluation config: preserve normal horizon
    # and failures, while bypassing the custom stateful vision corruption.
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.observations["actor"].terms["ball_state"].params["enable_noise"] = False

    return cfg
