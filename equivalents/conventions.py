from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MimicInfo:
    """URDF `<mimic>` metadata, name-keyed and frame-independent.

    `relations[name]` is a `(target_name, multiplier, offset)` triple describing
    `q[name] = multiplier * q[target_name] + offset` (and analogous for v/a),
    matching the URDF convention. Used by the pinocchio backend to:

    * EXPAND a project-layout (mimic-collapsed) q to a pinocchio-layout q,
      injecting the mimic-mirrored values; and
    * REDUCE a pinocchio-layout v/q-space derivative back to project-layout
      by folding mimic columns into the mimicked column with the multiplier.
    """

    relations: Mapping[str, tuple] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.relations

    def mimic_names(self) -> tuple:
        return tuple(self.relations.keys())


@dataclass(frozen=True)
class ConventionMismatch:
    category: str
    detail: str


def as_float64(array_like) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float64)


def normalize_pin_compatible_quaternion(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.shape[0] < 7:
        raise ValueError("Floating-base configuration must have at least 7 position entries.")
    norm = np.linalg.norm(q[3:7])
    if norm == 0.0:
        raise ValueError("Floating-base quaternion norm was zero during normalization.")
    q[3:7] /= norm
    return q


def expand_continuous_joint_positions_for_pin(
    q: Sequence[float],
    joint_names: Sequence[str],
    joint_types_by_name: Mapping[str, str] | None,
) -> np.ndarray:
    q = as_float64(q)
    if not joint_types_by_name:
        return q

    expanded = []
    if len(q) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} scalar joint positions, got {len(q)}."
        )

    for index, joint_name in enumerate(joint_names):
        joint_type = joint_types_by_name.get(joint_name)
        value = float(q[index])
        if joint_type == "continuous":
            expanded.extend((np.cos(value), np.sin(value)))
        else:
            expanded.append(value)
    return np.asarray(expanded, dtype=np.float64)


def collapse_continuous_joint_positions_from_pin(
    q_pin: Sequence[float],
    joint_names: Sequence[str],
    joint_types_by_name: Mapping[str, str] | None,
) -> np.ndarray:
    """Inverse of `expand_continuous_joint_positions_for_pin`: collapse each
    continuous joint's Pinocchio [cos,sin] pair back to a scalar angle via
    atan2. Used to keep a Pinocchio-integrated configuration in the project's
    scalar-joint layout so it can be fed back into project-layout routines."""
    q_pin = as_float64(q_pin)
    if not joint_types_by_name:
        return q_pin

    collapsed = []
    cursor = 0
    for joint_name in joint_names:
        joint_type = joint_types_by_name.get(joint_name)
        if joint_type == "continuous":
            cos_v, sin_v = float(q_pin[cursor]), float(q_pin[cursor + 1])
            collapsed.append(np.arctan2(sin_v, cos_v))
            cursor += 2
        else:
            collapsed.append(float(q_pin[cursor]))
            cursor += 1
    if cursor != len(q_pin):
        raise ValueError(
            f"Consumed {cursor} Pinocchio position entries but received {len(q_pin)}."
        )
    return np.asarray(collapsed, dtype=np.float64)


def collapse_pin_q_to_project(
    base_mode: str,
    q_pin: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Inverse of `normalize_project_q_for_pin`: map a Pinocchio-layout
    configuration back to the project's scalar-joint layout (free-flyer prefix
    kept as xyzw, continuous joints collapsed [cos,sin] -> angle)."""
    q_pin = as_float64(q_pin)
    if base_mode == "floating":
        q_prefix = q_pin[:7]
        if joint_names is None:
            return q_prefix
        q_suffix = collapse_continuous_joint_positions_from_pin(
            q_pin[7:], joint_names, joint_types_by_name
        )
        return np.concatenate((q_prefix, q_suffix))
    if joint_names is None:
        return q_pin
    return collapse_continuous_joint_positions_from_pin(q_pin, joint_names, joint_types_by_name)


def normalize_project_q_for_pin(
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    q = as_float64(q)
    if base_mode == "floating":
        q_prefix = normalize_pin_compatible_quaternion(q[:7])
        if joint_names is None:
            return q_prefix
        q_suffix = expand_continuous_joint_positions_for_pin(
            q[7:],
            joint_names,
            joint_types_by_name,
        )
        return np.concatenate((q_prefix, q_suffix))
    if joint_names is None:
        return q
    return expand_continuous_joint_positions_for_pin(q, joint_names, joint_types_by_name)


def project_q_to_pin_q_jacobian(
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    q = as_float64(q)

    rows = []
    cols = q.shape[0]

    if base_mode == "floating":
        if q.shape[0] < 7:
            raise ValueError("Floating-base configuration must have at least 7 position entries.")
        base_jac = np.zeros((7, cols), dtype=np.float64)
        base_jac[:, :7] = np.eye(7, dtype=np.float64)
        base_quat = q[3:7]
        norm = np.linalg.norm(base_quat)
        if norm == 0.0:
            raise ValueError("Floating-base quaternion norm was zero during normalization.")
        rows.append(base_jac)
        q_joints = q[7:]
        joint_col_offset = 7
    else:
        q_joints = q
        joint_col_offset = 0

    if joint_names is None:
        if rows:
            return rows[0]
        return np.eye(q.shape[0], dtype=np.float64)

    if len(q_joints) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} scalar joint positions, got {len(q_joints)}."
        )

    joint_rows = []
    for local_index, joint_name in enumerate(joint_names):
        joint_type = joint_types_by_name.get(joint_name) if joint_types_by_name else None
        col_index = joint_col_offset + local_index
        if joint_type == "continuous":
            theta = float(q_joints[local_index])
            block = np.zeros((2, cols), dtype=np.float64)
            block[0, col_index] = -np.sin(theta)
            block[1, col_index] = np.cos(theta)
            joint_rows.append(block)
        else:
            block = np.zeros((1, cols), dtype=np.float64)
            block[0, col_index] = 1.0
            joint_rows.append(block)

    if joint_rows:
        rows.append(np.vstack(joint_rows))

    if not rows:
        return np.zeros((0, cols), dtype=np.float64)
    return np.vstack(rows)


def reduce_pinocchio_q_jacobian_to_project(
    jacobian,
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    jacobian = normalize_matrix(jacobian)
    project_q = as_float64(q)
    if jacobian.shape[1] == project_q.shape[0]:
        return jacobian
    if base_mode == "floating" and jacobian.shape[1] == project_q.shape[0] - 1:
        return jacobian
    chain = project_q_to_pin_q_jacobian(
        base_mode,
        q,
        joint_names=joint_names,
        joint_types_by_name=joint_types_by_name,
    )
    if jacobian.shape[1] != chain.shape[0]:
        raise ValueError(
            f"Cannot reduce Pinocchio q-Jacobian with shape {jacobian.shape}; expected "
            f"{project_q.shape[0]} or {chain.shape[0]} columns."
        )
    return normalize_matrix(jacobian @ chain)


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    return np.atleast_1d(as_float64(vector)).reshape(-1)


def normalize_matrix(matrix) -> np.ndarray:
    return np.atleast_2d(as_float64(matrix))


def movable_joint_names_excluding_floating_root(
    base_mode: str, joint_names: Iterable[str]
) -> list:
    joint_names = list(joint_names)
    if base_mode == "floating" and joint_names:
        return [
            name
            for name in joint_names
            if name not in {"floating_base_joint", "root_joint"}
        ]
    return joint_names


def expand_q_for_mimic(
    q: Sequence[float],
    project_joint_names: Sequence[str],
    pinocchio_joint_names: Sequence[str],
    mimic: "MimicInfo",
    floating_prefix_len: int,
) -> np.ndarray:
    """Expand a project-layout q (mimic-collapsed) into a pinocchio-layout q.

    For each pinocchio joint name in `pinocchio_joint_names`:
      - if the name is a mimic joint, write `multiplier * q_target + offset`,
        where `q_target` is read from the already-expanded array at the
        target's slot;
      - otherwise, copy the project slot for that joint.

    Both `project_joint_names` and `pinocchio_joint_names` are the SCALAR
    actuated joint name lists (i.e. excluding the floating-base root).
    `floating_prefix_len` is the number of leading entries reserved for the
    free-flyer (7 for quaternion, 0 for fixed-base).
    """
    q = np.asarray(q, dtype=np.float64)
    if mimic.is_empty():
        return q
    project_index = {name: i for i, name in enumerate(project_joint_names)}
    out_prefix = q[:floating_prefix_len].tolist()
    out_suffix = []
    pin_index = {name: i for i, name in enumerate(pinocchio_joint_names)}
    target_pin_pos = {}
    cursor = 0
    for name in pinocchio_joint_names:
        if name in mimic.relations:
            target_name, mult, offset = mimic.relations[name]
            tgt_pin_pos = target_pin_pos.get(target_name)
            if tgt_pin_pos is None:
                raise ValueError(
                    f"mimic joint '{name}' references '{target_name}' which is "
                    "not yet expanded; expansion expects target to appear "
                    "earlier in the pinocchio joint order."
                )
            value = mult * out_suffix[tgt_pin_pos] + offset
            out_suffix.append(value)
        else:
            project_pos = project_index.get(name)
            if project_pos is None:
                raise ValueError(
                    f"non-mimic pinocchio joint '{name}' is missing from "
                    "project joint list; mimic expansion cannot proceed."
                )
            out_suffix.append(float(q[floating_prefix_len + project_pos]))
            target_pin_pos[name] = cursor
        cursor += 1
    return np.asarray(out_prefix + out_suffix, dtype=np.float64)


def reduce_matrix_for_mimic(
    matrix,
    project_joint_names: Sequence[str],
    pinocchio_joint_names: Sequence[str],
    mimic: "MimicInfo",
    floating_prefix_len_q: int,
    floating_prefix_len_v: int,
    axes_to_reduce: Sequence[tuple],
) -> np.ndarray:
    """Reduce a pinocchio-layout (q or v) matrix to project layout by folding
    mimic columns/rows into the mimicked column/row with the multiplier.

    `axes_to_reduce` is a list of `(axis, space)` tuples, where `space` is
    either `"q"` or `"v"`. For each tuple we collapse the named axis from
    pinocchio's joint-space to project's joint-space.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if mimic.is_empty():
        return arr
    for axis, space in axes_to_reduce:
        if space == "q":
            prefix = floating_prefix_len_q
        elif space == "v":
            prefix = floating_prefix_len_v
        else:
            raise ValueError(f"unknown reduction space: {space}")
        # Build index mapping from pinocchio slot -> project slot (or None if
        # the slot is a mimic — its column folds into the target's column).
        pin_to_project = [None] * (prefix + len(pinocchio_joint_names))
        for i in range(prefix):
            pin_to_project[i] = i
        project_index = {name: i for i, name in enumerate(project_joint_names)}
        pin_index = {name: i for i, name in enumerate(pinocchio_joint_names)}
        # First non-mimic joints get a slot.
        for i, name in enumerate(pinocchio_joint_names):
            if name in mimic.relations:
                continue
            project_pos = project_index.get(name)
            if project_pos is None:
                raise ValueError(
                    f"non-mimic pinocchio joint '{name}' missing from project list"
                )
            pin_to_project[prefix + i] = prefix + project_pos
        # Build the reduced matrix by accumulating columns.
        new_shape = list(arr.shape)
        new_shape[axis] = prefix + len(project_joint_names)
        reduced = np.zeros(tuple(new_shape), dtype=np.float64)
        for pin_slot in range(prefix + len(pinocchio_joint_names)):
            if pin_slot < prefix:
                tgt = pin_slot
                scale = 1.0
            else:
                pin_joint_index = pin_slot - prefix
                pin_name = pinocchio_joint_names[pin_joint_index]
                if pin_name in mimic.relations:
                    target_name, mult, _ = mimic.relations[pin_name]
                    target_project_pos = project_index.get(target_name)
                    if target_project_pos is None:
                        raise ValueError(
                            f"mimic '{pin_name}' targets unknown project joint '{target_name}'"
                        )
                    tgt = prefix + target_project_pos
                    scale = float(mult)
                else:
                    tgt = pin_to_project[pin_slot]
                    scale = 1.0
            # Slice src column from arr and accumulate into reduced[tgt] along axis.
            src_idx = [slice(None)] * arr.ndim
            src_idx[axis] = pin_slot
            tgt_idx = [slice(None)] * reduced.ndim
            tgt_idx[axis] = tgt
            reduced[tuple(tgt_idx)] = reduced[tuple(tgt_idx)] + scale * arr[tuple(src_idx)]
        arr = reduced
    return arr
