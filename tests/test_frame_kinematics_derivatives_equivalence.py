"""Frame velocity and acceleration derivatives against pinocchio and against the
reference's own tangent chart (2026-11-03, item C of 0.8).

``frame_velocity_derivatives`` is ``(dv/dq, dv/dqd)`` and
``frame_acceleration_derivatives`` is ``(dv/dq, da/dq, da/dqd, da/dqdd)``, in
the local tangent chart of ``integrate`` (the floating root and spherical
joints). They are pinocchio's ``getFrameVelocityDerivatives`` and
``getFrameAccelerationDerivatives`` in LOCAL and WORLD; in
LOCAL_WORLD_ALIGNED the acceleration rows differentiate the classical
acceleration that ``frame_acceleration`` returns there, which the pinocchio
backend forms from the spatial derivatives and ``w x v_lin``. The chart test
differentiates ``frame_velocity`` and ``frame_acceleration`` themselves
through ``integrate`` on every robot of the manifest, mimic robots included,
so the derivative is checked against the project's own values.
"""
import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.test_frame_jacobian_equivalence import _frame_targets, REFERENCE_FRAMES


def _params(robot_ids):
    params = []
    for base_mode in ("fixed", "floating"):
        for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
            spec = case["spec"]
            if robot_ids is not None and spec.robot_id not in robot_ids:
                continue
            marks = [pytest.mark.pinocchio_equivalence, pytest.mark.developer_only,
                     pytest.mark.robot_smoke]
            if base_mode == "floating":
                marks.append(pytest.mark.floating_base)
            params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


@pytest.mark.parametrize(("spec", "base_mode"), _params(("iiwa14", "go2", "gen3")))
def test_frame_velocity_derivatives_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    targets, _ = _frame_targets(project_model, pinocchio_model)
    assert targets, f"No frame targets for {spec.robot_id}"
    for target in targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_velocity_derivatives(sample.q, sample.qd, target, ref)
                expected = pinocchio_model.frame_velocity_derivatives(sample.q, sample.qd, target, ref)
                for a, e in zip(actual, expected):
                    assert_close(a, e, algorithm="frame_jacobian_dot", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), _params(("iiwa14", "go2", "gen3")))
def test_frame_acceleration_derivatives_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    targets, _ = _frame_targets(project_model, pinocchio_model)
    assert targets, f"No frame targets for {spec.robot_id}"
    for target in targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_acceleration_derivatives(sample.q, sample.qd, sample.qdd, target, ref)
                expected = pinocchio_model.frame_acceleration_derivatives(sample.q, sample.qd, sample.qdd, target, ref)
                for a, e in zip(actual, expected):
                    assert_close(a, e, algorithm="frame_jacobian_dot", robot_id=spec.robot_id)


def _chart_fd(ref, f, q, nv, h=1e-6):
    cols = []
    for k in range(nv):
        e = np.zeros(nv)
        e[k] = h
        cols.append((f(ref.integrate(q, e)) - f(ref.integrate(q, -e))) / (2.0 * h))
    return np.array(cols).T


def _plain_fd(f, x, h=1e-6):
    cols = []
    for k in range(len(x)):
        e = np.zeros(len(x))
        e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2.0 * h))
    return np.array(cols).T


@pytest.mark.parametrize(("spec", "base_mode"), _params(None))
def test_frame_kinematics_derivatives_match_own_chart_fd(spec, base_mode, project_model):
    """The analytic derivatives against central differences of the reference's
    own frame_velocity / frame_acceleration through its integrate chart."""
    ref = project_model.reference
    robot = ref.robot
    names = [robot.get_joint_by_id(j).get_name() for j in range(robot.get_num_joints())]
    leaf = robot.get_joint_by_id(int(robot.get_leaf_nodes()[0])).get_name()
    targets = sorted({leaf, names[len(names) // 2]})
    nv = robot.get_num_vel()
    sample = build_dynamics_samples(project_model)[0]
    q, qd, qdd = (np.asarray(x, dtype=np.float64) for x in (sample.q, sample.qd, sample.qdd))
    tol = 2e-6
    for target in targets:
        for rf in REFERENCE_FRAMES:
            dv_dq, dv_dqd = ref.frame_velocity_derivatives(q, qd, target, rf)
            dv_dq2, da_dq, da_dqd, da_dqdd = ref.frame_acceleration_derivatives(q, qd, qdd, target, rf)
            np.testing.assert_allclose(dv_dq2, dv_dq, rtol=0, atol=1e-12)
            fd = _chart_fd(ref, lambda qq: ref.frame_velocity(qq, qd, target, rf), q, nv)
            np.testing.assert_allclose(dv_dq, fd, rtol=0, atol=tol, err_msg=f"dv/dq {target} {rf}")
            fd = _plain_fd(lambda v: ref.frame_velocity(q, v, target, rf), qd)
            np.testing.assert_allclose(dv_dqd, fd, rtol=0, atol=tol, err_msg=f"dv/dqd {target} {rf}")
            fd = _chart_fd(ref, lambda qq: ref.frame_acceleration(qq, qd, qdd, target, rf), q, nv)
            np.testing.assert_allclose(da_dq, fd, rtol=0, atol=tol, err_msg=f"da/dq {target} {rf}")
            fd = _plain_fd(lambda v: ref.frame_acceleration(q, v, qdd, target, rf), qd)
            np.testing.assert_allclose(da_dqd, fd, rtol=0, atol=tol, err_msg=f"da/dqd {target} {rf}")
            fd = _plain_fd(lambda a: ref.frame_acceleration(q, qd, a, target, rf), qdd)
            np.testing.assert_allclose(da_dqdd, fd, rtol=0, atol=tol, err_msg=f"da/dqdd {target} {rf}")
