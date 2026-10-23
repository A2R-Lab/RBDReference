"""The pinocchio backend's pose derivatives, which the CUDA receipt uses as the
oracle for the end-effector pose gradient and Hessian cells.

The gradient is analytic (pinocchio's local-world-aligned frame Jacobian and
the roll-pitch-yaw rate map); the Hessian is one central difference of that
gradient. Two checks per manifest robot and base, both internal to the
backend so that a truncated URDF angle (gen3 writes 3.1416, which the project
parser snaps to pi and pinocchio reads literally) cannot enter: the gradient is
the derivative of the backend's own pose, and the Hessian is its second
derivative. The backend-versus-project comparison with its per-robot geometry
buckets lives in test_kinematics_derivatives_equivalence.
"""
import numpy as np
import pytest

from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.test_kinematics_derivatives_equivalence import select_derivative_target


def _configuration(project_model, base_mode):
    rng = np.random.default_rng(20261023)
    q = rng.uniform(-1.0, 1.0, project_model.robot.get_num_pos())
    if base_mode == "floating":
        q[3:7] /= np.linalg.norm(q[3:7])
    return q


def _target(project_model, pinocchio_model):
    picked = select_derivative_target(project_model, pinocchio_model)
    assert picked is not None, "every manifest robot has an articulated leaf target"
    target, offset = picked
    leaf_names = [project_model.robot.get_joint_by_id(i).get_name()
                  for i in project_model.robot.get_leaf_nodes()]
    return target, offset, leaf_names.index(target)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_pinocchio_pose_gradient_differentiates_its_pose(spec, base_mode, project_model, pinocchio_model):
    target, offset, leaf = _target(project_model, pinocchio_model)
    q = _configuration(project_model, base_mode)
    gradient = pinocchio_model.end_effector_pose_gradient(q, target, offset=offset)
    nv = project_model.nv
    step = 1e-6
    differenced = np.zeros((6, nv), dtype=np.float64)
    for i in range(nv):
        v = np.zeros(nv, dtype=np.float64)
        v[i] = step
        delta = (pinocchio_model.end_effector_pose(pinocchio_model._pin_integrate(q, v), target, offset=offset)
                 - pinocchio_model.end_effector_pose(pinocchio_model._pin_integrate(q, -v), target, offset=offset))
        delta[3:6] = ((delta[3:6] + np.pi) % (2.0 * np.pi)) - np.pi
        differenced[:, i] = delta / (2.0 * step)
    np.testing.assert_allclose(gradient, differenced, rtol=1e-7, atol=1e-8)


def _wrapped_pose_difference(pinocchio_model, q, v_plus, v_minus, target, offset):
    delta = (pinocchio_model.end_effector_pose(pinocchio_model._pin_integrate(q, v_plus), target, offset=offset)
             - pinocchio_model.end_effector_pose(pinocchio_model._pin_integrate(q, v_minus), target, offset=offset))
    delta[3:6] = ((delta[3:6] + np.pi) % (2.0 * np.pi)) - np.pi
    return delta


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_pinocchio_pose_hessian_is_the_second_derivative_of_its_pose(spec, base_mode, project_model, pinocchio_model):
    target, offset, _leaf = _target(project_model, pinocchio_model)
    q = _configuration(project_model, base_mode)
    hessian = pinocchio_model.end_effector_pose_hessian(q, target, offset=offset)
    nv = project_model.nv
    step = 1e-4
    differenced = np.zeros((6, nv, nv), dtype=np.float64)
    for i in range(nv):
        for j in range(i, nv):
            e_i = np.zeros(nv, dtype=np.float64)
            e_j = np.zeros(nv, dtype=np.float64)
            e_i[i] = step
            e_j[j] = step
            upper = _wrapped_pose_difference(pinocchio_model, q, e_i + e_j, e_i - e_j, target, offset)
            lower = _wrapped_pose_difference(pinocchio_model, q, -e_i + e_j, -e_i - e_j, target, offset)
            differenced[:, i, j] = differenced[:, j, i] = (upper - lower) / (4.0 * step * step)
    np.testing.assert_allclose(hessian, differenced, rtol=1e-6, atol=1e-6)
