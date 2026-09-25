"""ONNX metadata helpers for passive joints and Python-PD actuators."""

from __future__ import annotations

import onnx

from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx as _mjlab_attach_metadata_to_onnx,
)
from mjlab.rl.exporter_utils import (
    get_base_metadata as _mjlab_get_base_metadata,
)

__all__ = (
    "attach_base_metadata",
    "attach_metadata_to_onnx",
    "attach_exclusion_metadata",
    "get_base_metadata",
)


def attach_metadata_to_onnx(
    onnx_path: str, metadata: dict[str, list | str | float]
) -> None:
    """Attach metadata, replacing properties that already use the same key."""
    model = onnx.load(onnx_path)
    replaced_keys = metadata.keys()
    for index in range(len(model.metadata_props) - 1, -1, -1):
        if model.metadata_props[index].key in replaced_keys:
            del model.metadata_props[index]
    onnx.save(model, onnx_path)
    _mjlab_attach_metadata_to_onnx(onnx_path, metadata)


def attach_base_metadata(env: ManagerBasedRlEnv, onnx_path: str) -> None:
    """Replace mjlab's base metadata while preserving the exported run path."""
    model = onnx.load(onnx_path)
    run_path = next(
        (entry.value for entry in model.metadata_props if entry.key == "run_path"),
        "local",
    )
    attach_metadata_to_onnx(onnx_path, get_base_metadata(env, run_path))


def attach_exclusion_metadata(env: ManagerBasedRlEnv, onnx_path: str) -> None:
    """Append joint/EE exclusion metadata to an exported ONNX, if the env has any.

    Excluded joints are held at their default pose by their PD controller and are
    not actuated by the policy, so the deployment side needs to know them to keep
    the exported action dim and the joint mapping consistent. Reads the generic
    ``_excluded_joint_names`` / ``_excluded_ee_names`` env attributes. No-op when the env
    declares no exclusions, so it is safe to call from any runner's ``save``.
    """
    joints = list(getattr(env, "_excluded_joint_names", None) or [])
    ee = list(getattr(env, "_excluded_ee_names", None) or [])
    if not (joints or ee):
        return
    extra: dict[str, list | str | float] = {}
    if joints:
        extra["excluded_joint_names"] = joints
    if ee:
        extra["excluded_ee_names"] = ee
    attach_metadata_to_onnx(str(onnx_path), extra)


def get_base_metadata(
    env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
    """Build ONNX metadata, fixing PD gains for python-PD-backed actuators."""
    robot: Entity = env.scene["robot"]
    md = _mjlab_get_base_metadata(env, run_path)
    _align_joint_metadata_with_actions(env, robot, md)
    kick_schema = getattr(env, "_kick_command_schema", None)
    if kick_schema is not None:
        md["kick_command_schema"] = kick_schema

    if not any(isinstance(a, IdealPdActuator) for a in robot.actuators):
        return md

    overrides: dict[str, tuple[float, float]] = {}
    for act in robot.actuators:
        if not isinstance(act, IdealPdActuator):
            continue
        # stiffness/damping have shape (num_envs, num_joints); env 0 is canonical.
        kp = act.stiffness[0].cpu().tolist()
        kd = act.damping[0].cpu().tolist()
        for name, p, d in zip(act.target_names, kp, kd):
            overrides[name] = (p, d)

    new_kp: list[float] = []
    new_kd: list[float] = []
    for jname, kp_default, kd_default in zip(
        md["joint_names"], md["joint_stiffness"], md["joint_damping"]
    ):
        if jname in overrides:
            p, d = overrides[jname]
            new_kp.append(p)
            new_kd.append(d)
        else:
            new_kp.append(kp_default)
            new_kd.append(kd_default)
    md["joint_stiffness"] = new_kp
    md["joint_damping"] = new_kd
    return md


def _align_joint_metadata_with_actions(
    env: ManagerBasedRlEnv, robot: Entity, md: dict
) -> None:
    """Align joint metadata with the exact action ordering used by the policy.

    An articulated robot can expose actuators which are deliberately not driven
    by a policy. In particular, the kick policy leaves K1's head joints to a
    perception controller. Export metadata must follow the action term rather
    than every actuator, otherwise its joint fields disagree with action_scale.
    """
    controlled_names: list[str] = []
    for term_name in env.action_manager.active_terms:
        action_term = env.action_manager.get_term(term_name)
        target_names = getattr(action_term, "target_names", ())
        controlled_names.extend(
            name for name in target_names if name in robot.joint_names
        )

    metadata_joint_names = list(md["joint_names"])
    if controlled_names:
        if len(set(controlled_names)) != len(controlled_names):
            raise ValueError(
                "ONNX metadata cannot represent duplicate controlled joint names"
            )
        joint_names = controlled_names
    else:
        actuated_names = {
            name for actuator in robot.actuators for name in actuator.target_names
        }
        joint_names = [name for name in robot.joint_names if name in actuated_names]

    try:
        robot_joint_ids = [robot.joint_names.index(name) for name in joint_names]
        metadata_joint_ids = [metadata_joint_names.index(name) for name in joint_names]
    except ValueError as error:
        raise ValueError(
            "ONNX metadata is missing a joint controlled by the policy"
        ) from error

    md["joint_names"] = joint_names
    md["default_joint_pos"] = (
        robot.data.default_joint_pos[0, robot_joint_ids].cpu().tolist()
    )
    for key in ("joint_stiffness", "joint_damping"):
        md[key] = [md[key][index] for index in metadata_joint_ids]

    joint_field_lengths = {
        key: len(md[key])
        for key in (
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "default_joint_pos",
            "action_scale",
        )
    }
    if len(set(joint_field_lengths.values())) != 1:
        raise ValueError(f"ONNX joint metadata is misaligned: {joint_field_lengths}")
