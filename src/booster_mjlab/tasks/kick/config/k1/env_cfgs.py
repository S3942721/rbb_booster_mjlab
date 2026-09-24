"""Booster K1-specific kick task environment configuration."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from booster_mjlab.environment.ball.ball_constants import (
    BALL_SIZE_SCALES,
    get_ball_cfg,
)
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
    joint_pos_action.scale = K1_ACTION_SCALE

    cfg.viewer.body_name = "Trunk"

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )

    # Ball size/mass DR: built at size 5 (radius 0.11), scaled down toward size 1
    # (per user decision: randomize the full FIFA range from the start).
    size_1_scale = BALL_SIZE_SCALES[1]["size"] / BALL_SIZE_SCALES[5]["size"]
    weight_1_scale = BALL_SIZE_SCALES[1]["weight"] / BALL_SIZE_SCALES[5]["weight"]
    cfg.events["ball_size"] = EventTermCfg(
        func=velocity_mdp.dr.geom_size,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
            "operation": "scale",
            "ranges": {0: (size_1_scale, 1.0)},
        },
    )
    # dr.pseudo_inertia (not dr.body_mass) so mass and inertia scale together
    # consistently, per its own docstring's recommendation for uniform density
    # changes. alpha is a log-scale: mass/inertia scale by e^(2*alpha).
    weight_1_alpha = 0.5 * math.log(weight_1_scale)
    cfg.events["ball_mass"] = EventTermCfg(
        func=velocity_mdp.dr.pseudo_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball", body_names=("ball",)),
            "alpha_range": (weight_1_alpha, 0.0),
        },
    )

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("illegal_contact", None)
        cfg.terminations.pop("mis_kick", None)

    return cfg
