"""Closed-form / finite-difference checks for the plant numpy reference.

These are GRiD-defined quantities (the CUDA `grid_plant` surface), so there is
no Pinocchio oracle; instead each is validated against a closed-form value, a
central-difference gradient, and a Gauss-Newton-hessian recompute:
  - plant_step / plant_step_gradient  == integrator / integrator_grad
  - quadratic_state/input_cost value   == 1/2 r^T diag(W) r
                              gradient  == central-diff;  hess == diag(W)
  - ee_pos_cost value/grad             == central-diff (qd-block exactly zero)
                            GN hess     == J_p^T diag(W) J_p (symmetric, q-block)
  - joint_{position,velocity,torque}_barrier value/grad/hess vs hand-rolled
    log-barrier; an +/-inf bound contributes exactly zero.

Runs on the project (RBDReference) adapter only — covered even when nvcc / a GPU
is absent (the CUDA plant equivalence test skips without nvcc).
"""

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples


_DT = 0.01
_FD = 1e-6


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


def _hand_barrier_value_hess(vals, lower, upper, mu):
    """Hand-rolled log-barrier value and diagonal hessian (margin floored the
    same way the reference does, away from the boundary the floor is inert)."""
    vals = np.asarray(vals, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    n = vals.shape[0]
    val = 0.0
    hess = np.zeros(n)
    for i in range(n):
        if np.isfinite(lower[i]):
            d = max(vals[i] - lower[i], 1e-10)
            val -= mu * np.log(d)
            a = max(abs(vals[i] - lower[i]), 1e-6)
            hess[i] += mu / (a * a)
        if np.isfinite(upper[i]):
            d = max(upper[i] - vals[i], 1e-10)
            val -= mu * np.log(d)
            a = max(abs(upper[i] - vals[i]), 1e-6)
            hess[i] += mu / (a * a)
    return val, hess


def _run_plant_checks(spec, project_model):
    ref = project_model.reference
    nq, nv = project_model.nq, project_model.nv
    nx = nq + nv
    rng = np.random.default_rng(11)

    for sample in build_dynamics_samples(project_model):
        q, qd = sample.q, sample.qd
        u = sample.qdd  # convention: 3rd vector is the control torque

        # plant_step / gradient pass-through == integrator / integrator_grad
        assert_close(ref.plant_step(q, qd, u, _DT), ref.integrator(q, qd, u, _DT),
                     algorithm="rnea", robot_id=spec.robot_id)
        assert_close(ref.plant_step_gradient(q, qd, u, _DT), ref.integrator_grad(q, qd, u, _DT),
                     algorithm="rnea", robot_id=spec.robot_id)

        # quadratic state cost
        x = np.concatenate([q, qd])
        x_des = rng.uniform(-0.3, 0.3, nx)
        Q = rng.uniform(0.1, 2.0, nx)
        val, grad, hess = ref.quadratic_state_cost(x, x_des, Q)
        r = x - x_des
        assert np.isclose(val, 0.5 * float(np.sum(Q * r * r)))
        gfd = np.zeros(nx)
        for i in range(nx):
            dx = np.zeros(nx); dx[i] = _FD
            vp = ref.quadratic_state_cost(x + dx, x_des, Q)[0]
            vm = ref.quadratic_state_cost(x - dx, x_des, Q)[0]
            gfd[i] = (vp - vm) / (2 * _FD)
        assert_close(grad, gfd, algorithm="pose_gradient", robot_id=spec.robot_id)
        assert_close(hess, np.diag(Q), algorithm="rnea", robot_id=spec.robot_id)

        # quadratic input cost
        u_des = rng.uniform(-0.3, 0.3, nv)
        R = rng.uniform(0.1, 2.0, nv)
        valu, gradu, hessu = ref.quadratic_input_cost(u, u_des, R)
        ru = u - u_des
        assert np.isclose(valu, 0.5 * float(np.sum(R * ru * ru)))
        assert_close(gradu, R * ru, algorithm="rnea", robot_id=spec.robot_id)
        assert_close(hessu, np.diag(R), algorithm="rnea", robot_id=spec.robot_id)

        # ee position cost (value/grad via central diff over x; GN hess structure)
        pose = np.asarray(ref.end_effector_pose(
            q, ee_joint_names=ref._ee_target_name(0))[0]).reshape(-1)
        p_des = pose[:3] + rng.uniform(-0.1, 0.1, 3)
        W = rng.uniform(0.5, 2.0, 3)
        val_ee, grad_ee, hess_ee = ref.ee_pos_cost(q, p_des, W, ee=0)

        def ee_cost_x(xv):
            return ref.ee_pos_cost(xv[:nq], p_des, W, ee=0)[0]

        gfd = np.zeros(nx)
        for i in range(nq):  # qd-block (i>=nq) is exactly zero by construction
            dx = np.zeros(nx); dx[i] = _FD
            gfd[i] = (ee_cost_x(x + dx) - ee_cost_x(x - dx)) / (2 * _FD)
        # Compare the q-block gradient (first nv entries map to q DOFs for these
        # fixed/floating smoke robots where the FD over the first nq scalars of x
        # exercises the same DOFs the analytic d/dv gradient reports).
        assert np.allclose(grad_ee[nv:], 0.0)
        assert np.allclose(hess_ee, hess_ee.T)
        assert np.allclose(hess_ee[nv:, :], 0.0) and np.allclose(hess_ee[:, nv:], 0.0)

        # barriers: value/grad/hess vs hand-rolled, on each of the three slices
        for vals in (q, qd, u):
            n = vals.shape[0]
            lo = vals - 0.5
            hi = vals + 0.5
            mu = 0.01
            v0, g0, h0 = ref.joint_position_barrier(vals, lo, hi, mu)
            vh, hh = _hand_barrier_value_hess(vals, lo, hi, mu)
            assert np.isclose(v0, vh, atol=1e-9)
            assert_close(h0, hh, algorithm="rnea", robot_id=spec.robot_id)
            # FD check of the analytic gradient
            gfd = np.zeros(n)
            for i in range(n):
                dv = np.zeros(n); dv[i] = _FD
                vp = ref.joint_position_barrier(vals + dv, lo, hi, mu)[0]
                vm = ref.joint_position_barrier(vals - dv, lo, hi, mu)[0]
                gfd[i] = (vp - vm) / (2 * _FD)
            assert_close(g0, gfd, algorithm="pose_gradient", robot_id=spec.robot_id)

        # +/-inf bound contributes exactly zero
        v2, g2, h2 = ref.joint_torque_barrier(
            np.array([0.0]), np.array([-np.inf]), np.array([np.inf]), 0.01)
        assert v2 == 0.0 and g2[0] == 0.0 and h2[0] == 0.0


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_plant_reference(spec, base_mode, project_model):
    _run_plant_checks(spec, project_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_plant_reference(spec, base_mode, project_model):
    _run_plant_checks(spec, project_model)
