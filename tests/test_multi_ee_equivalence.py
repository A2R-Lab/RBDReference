"""Multi-target end-effector pose + Jacobian equivalence (PS5 task 2).

`RBDReference.end_effector_pose` / `end_effector_pose_gradient` already accept a
LIST of `ee_joint_names` and return one result per target (and `None` => every
leaf), so "all end-effectors at once" needs no new code — this test pins that
multi-target surface:

  * the stacked list call returns exactly one pose / 6xnv Jacobian per requested
    EE, in request order;
  * each entry is bit-identical to the single-target call for that EE (the
    eventual multi-target codegen must reproduce this per-EE independence);
  * each EE pose (xyz + rotation matrix) and Jacobian matches pinocchio's
    per-frame FK / frame Jacobian.

The single-target path itself is already covered by
`test_kinematics_equivalence.py` / `test_kinematics_derivatives_equivalence.py`;
this file is specifically the multi-target / all-leaves contract.

rpy rows are compared via the rotation matrix (not the raw rpy triple) to dodge
the atan2 branch cut, mirroring `test_kinematics_equivalence.py`. The pinocchio
pose-gradient reference is a central FD on the d/dv Jacobian, so the gradient
check uses the `pose_gradient` bucket.

Gimbal lock: the [xyz; rpy] pose gradient uses the rpy-rate map E(rpy), which is
singular at pitch = +/-pi/2 (the rotation rows of the Jacobian then blow up on
BOTH the project oracle and the pinocchio FD reference — a representation
limitation of the rpy pose, not a multi-target bug). When an EE sample sits in
that band the per-EE GRADIENT cross-check is skipped, but the well-defined pose
position + rotation-matrix checks still run. rizon4's zero-inertia URDF is skipped
wholesale (consistent with the rest of the suite).
"""

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples


def build_case_params(base_mode: str):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            pytest.mark.robot_smoke,
        ]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


def _leaf_names(project_model):
    robot = project_model.robot
    return [robot.get_joint_by_id(jid).get_name() for jid in robot.get_leaf_nodes()]


def _near_gimbal_lock(rot, band=0.05):
    """True when the EE rotation sits within `band` rad of pitch = +/-pi/2, where
    the rpy-rate map E(rpy) is singular and the pose GRADIENT is ill-defined."""
    pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
    return abs(abs(pitch) - np.pi / 2.0) < band


def _check_multi_ee(spec, project_model, pinocchio_model):
    if spec.robot_id == "rizon4":
        pytest.skip("rizon4 ships a zero-inertia URDF; its FK/pose oracle is non-physical.")
    ref = project_model.reference
    leaf_names = _leaf_names(project_model)
    pin_frames = set(pinocchio_model.frame_names)
    mimic = set(getattr(pinocchio_model, "mimic_joint_names", []))

    for sample in build_dynamics_samples(project_model):
        q = sample.q

        # --- all end-effectors in ONE call (explicit list AND the None=all path)
        poses_list = ref.end_effector_pose(q, ee_joint_names=leaf_names)
        grads_list = ref.end_effector_pose_gradient(q, ee_joint_names=leaf_names)
        poses_default = ref.end_effector_pose(q)  # None -> all leaves
        assert len(poses_list) == len(leaf_names)
        assert len(grads_list) == len(leaf_names)
        assert len(poses_default) == len(leaf_names)

        for k, name in enumerate(leaf_names):
            pose_multi = np.asarray(poses_list[k], dtype=np.float64).reshape(-1)
            grad_multi = np.asarray(grads_list[k], dtype=np.float64)

            # (a) multi-target entry == single-target call (per-EE independence)
            pose_single = np.asarray(
                ref.end_effector_pose(q, ee_joint_names=name)[0], dtype=np.float64
            ).reshape(-1)
            grad_single = np.asarray(
                ref.end_effector_pose_gradient(q, ee_joint_names=name)[0], dtype=np.float64
            )
            assert_close(pose_multi, pose_single, algorithm="inverse_dynamics", robot_id=spec.robot_id)
            assert_close(grad_multi, grad_single, algorithm="inverse_dynamics", robot_id=spec.robot_id)
            # None=all path agrees too
            assert_close(
                np.asarray(poses_default[k], dtype=np.float64).reshape(-1),
                pose_multi, algorithm="inverse_dynamics", robot_id=spec.robot_id,
            )

            # (b) vs pinocchio per EE (skip mimic-named / non-frame leaves)
            if name in mimic or name not in pin_frames:
                continue
            pin_pose = pinocchio_model.end_effector_pose(q, name)
            pin_rot = pinocchio_model.end_effector_rotation_matrix(q, name)
            ref_rot = project_model.end_effector_rotation_matrix(q, name)
            # xyz position + rotation matrix (rpy via rotation to avoid branch cut)
            assert_close(pose_multi[:3], pin_pose[:3], algorithm="inverse_dynamics", robot_id=spec.robot_id)
            assert_close(ref_rot, pin_rot, algorithm="inverse_dynamics", robot_id=spec.robot_id)
            # gradient: skip at gimbal lock (singular rpy-rate map on both sides)
            if _near_gimbal_lock(ref_rot):
                continue
            pin_grad = pinocchio_model.end_effector_pose_gradient(q, name)
            assert_close(grad_multi, pin_grad, algorithm="pose_gradient", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_multi_ee_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_multi_ee(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_multi_ee_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_multi_ee(spec, project_model, pinocchio_model)
