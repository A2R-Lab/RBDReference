"""Equivalence tests for the runtime "welded tool" SE(3) tip offset.

Validates the SE(3)-generalized ``end_effector_pose`` / ``end_effector_pose_gradient``
(a tool/tip frame ``X_frame = X_target @ X_tool`` with its OWN orientation, not just a
point offset):

  * pose value vs pinocchio ``framePlacement * X_tool`` (rotated tool frame);
  * pose gradient (``[Jv; E^-1 Jw]``, rpy-rate) vs central-FD of the SAME pose along
    the tangent v (own-FD: the correct check for the rpy-rate form, independent of
    pinocchio's geometric-Jacobian frame conventions);
  * point-offset regression: a point ``[x,y,z]`` gives output identical to the SE(3)
    ``[[I,p],[0,1]]`` -- proving the generalization leaves the legacy point path byte-exact.

Fixed-base (iiwa14) + floating-base (go2), matching the frame-Jacobian suite.
"""

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.test_kinematics_equivalence import pose_vector_to_rotation_matrix


def _rot_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _tool_transform():
    """A nontrivial welded-tool offset: ~35 deg about a tilted axis + translation."""
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = _rot_axis_angle([0.3, -0.7, 0.65], 0.6108)  # ~35 deg
    X[:3, 3] = np.array([0.05, -0.02, 0.10], dtype=np.float64)
    return X


def _leaf_target(project_model, pinocchio_model):
    mimic = set(getattr(pinocchio_model, "mimic_joint_names", []))
    names = [n for n in project_model.joint_names
             if n not in mimic and n in pinocchio_model.frame_names]
    return names[-1] if names else None


def _make_params():
    params = []
    for base_mode in ("fixed", "floating"):
        for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
            spec = case["spec"]
            if spec.robot_id not in ("iiwa14", "go2"):
                continue
            marks = [pytest.mark.pinocchio_equivalence, pytest.mark.developer_only,
                     pytest.mark.robot_smoke]
            if base_mode == "floating":
                marks.append(pytest.mark.floating_base)
            params.append(pytest.param(spec, base_mode,
                                        id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


_PARAMS = _make_params()


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_tool_pose_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    target = _leaf_target(project_model, pinocchio_model)
    if target is None:
        pytest.skip(f"No leaf target for {spec.robot_id}")
    X_tool = _tool_transform()
    for sample in build_dynamics_samples(project_model):
        actual = project_model.end_effector_pose(sample.q, target, offset=X_tool)
        expected = pinocchio_model.end_effector_pose(sample.q, target, offset=X_tool)
        # position
        np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-8, atol=1e-9)
        # orientation via rotation matrix (avoids rpy branch-cut mismatch)
        np.testing.assert_allclose(pose_vector_to_rotation_matrix(actual),
                                   pose_vector_to_rotation_matrix(expected),
                                   rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_tool_pose_gradient_matches_own_fd(spec, base_mode, project_model, pinocchio_model):
    target = _leaf_target(project_model, pinocchio_model)
    if target is None:
        pytest.skip(f"No leaf target for {spec.robot_id}")
    X_tool = _tool_transform()
    ref = project_model.reference
    nv = ref.robot.get_num_vel()
    h = 1e-6
    for sample in build_dynamics_samples(project_model):
        q = sample.q
        J = project_model.end_effector_pose_gradient(q, target, offset=X_tool)
        Jfd = np.zeros((6, nv), dtype=np.float64)
        for i in range(nv):
            v = np.zeros(nv, dtype=np.float64); v[i] = h
            pose_p = np.asarray(ref.end_effector_pose(
                ref.integrate(q, v), ee_joint_names=target, ee_offsets=[X_tool])[0]).reshape(-1)
            pose_m = np.asarray(ref.end_effector_pose(
                ref.integrate(q, -v), ee_joint_names=target, ee_offsets=[X_tool])[0]).reshape(-1)
            Jfd[:, i] = (pose_p - pose_m) / (2.0 * h)
        np.testing.assert_allclose(J, Jfd, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_point_offset_regression(spec, base_mode, project_model, pinocchio_model):
    """A point offset must equal the SE(3) offset with R_tool = I (byte-level),
    proving the SE(3) generalization does not perturb the legacy point path."""
    target = _leaf_target(project_model, pinocchio_model)
    if target is None:
        pytest.skip(f"No leaf target for {spec.robot_id}")
    p = np.array([0.05, -0.02, 0.10], dtype=np.float64)
    X_point = np.eye(4, dtype=np.float64); X_point[:3, 3] = p
    for sample in build_dynamics_samples(project_model):
        q = sample.q
        pose_pt = project_model.end_effector_pose(q, target, offset=p)
        pose_se3 = project_model.end_effector_pose(q, target, offset=X_point)
        np.testing.assert_array_equal(pose_pt, pose_se3)
        J_pt = project_model.end_effector_pose_gradient(q, target, offset=p)
        J_se3 = project_model.end_effector_pose_gradient(q, target, offset=X_point)
        np.testing.assert_array_equal(J_pt, J_se3)
