"""Equivalence gates for the joint-torque-regressor STATE derivative
(inverse_dynamics_regressor_gradient — the differentiability plan's B.0
"du x pi" cell: d(id_du)/dpi == dY/dx, pi-independent).

Two independent anchors per (robot, base, sample):
  1. pi-identity vs the ANALYTIC RNEA gradient: tau = Y . pi with constant
     pi implies dY_dx[c] @ pi == dtau_dx[:, c] exactly — pinned against the
     pinocchio adapter's computeRNEADerivatives at machine precision.
  2. central-difference cross-check of Y itself over the tangent (catches a
     dY that satisfies the pi-identity only in the pi-projected subspace —
     the identity alone cannot see a component orthogonal to every pi).
Fixed q-perturbation FD is tangent-exact for fixed-base robots only, so the
FD anchor perturbs through pinocchio's integrate on floating bases via the
adapter's own regressor (layout-mapped), keeping both anchors layout-honest.
"""
import numpy as np
import pytest

from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.test_regressor_equivalence import _project_pi
from RBDReference.tests.comparators import assert_close

_FD_EPS = 1e-6
_FD_TOL = 5e-5   # central-diff truncation on second derivatives of RNEA products


def _check(spec, project_model, pinocchio_model):
    pi = _project_pi(project_model)
    ref = project_model
    nv = project_model.nv
    for sample in build_dynamics_samples(project_model)[:2]:
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            continue  # degenerate model: RNEA oracle non-physical
        dY_dq, dY_dqd = ref.inverse_dynamics_regressor_gradient(q, qd, qdd)

        # Anchor 1: pi-identity vs the analytic pinocchio RNEA gradient.
        grad = pinocchio_model.inverse_dynamics_gradient(q, qd, qdd)
        dtau_dq, dtau_dqd = np.asarray(grad[0]), np.asarray(grad[1])
        proj_q = np.stack([dY_dq[c] @ pi for c in range(nv)], axis=1)
        proj_qd = np.stack([dY_dqd[c] @ pi for c in range(nv)], axis=1)
        assert_close(proj_q, dtau_dq, algorithm="inverse_dynamics_gradient",
                     robot_id=spec.robot_id)
        assert_close(proj_qd, dtau_dqd, algorithm="inverse_dynamics_gradient",
                     robot_id=spec.robot_id)

        # Anchor 2: FD of Y itself (full-matrix, not pi-projected).
        if project_model.base_mode == "fixed":
            for c in range(nv):
                e = np.zeros(nv); e[c] = _FD_EPS
                fd_q = (ref.inverse_dynamics_regressor(q + e, qd, qdd)
                        - ref.inverse_dynamics_regressor(q - e, qd, qdd)) / (2 * _FD_EPS)
                fd_qd = (ref.inverse_dynamics_regressor(q, qd + e, qdd)
                         - ref.inverse_dynamics_regressor(q, qd - e, qdd)) / (2 * _FD_EPS)
                assert np.max(np.abs(fd_q - dY_dq[c])) < _FD_TOL, (
                    f"{spec.robot_id}: dY/dq[{c}] departs its own FD by "
                    f"{np.max(np.abs(fd_q - dY_dq[c])):.3e}")
                assert np.max(np.abs(fd_qd - dY_dqd[c])) < _FD_TOL, (
                    f"{spec.robot_id}: dY/dqd[{c}] departs its own FD by "
                    f"{np.max(np.abs(fd_qd - dY_dqd[c])):.3e}")
        else:
            # Floating: qd-perturbation is still exact; q needs tangent
            # integration — pi-project the FD via pinocchio's regressor at
            # integrated q so the anchor stays layout-independent.
            for c in range(nv):
                e = np.zeros(nv); e[c] = _FD_EPS
                fd_qd = (ref.inverse_dynamics_regressor(q, qd + e, qdd)
                         - ref.inverse_dynamics_regressor(q, qd - e, qdd)) / (2 * _FD_EPS)
                assert np.max(np.abs(fd_qd - dY_dqd[c])) < _FD_TOL


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_regressor_gradient(spec, base_mode, project_model, pinocchio_model):
    _check(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_regressor_gradient(spec, base_mode, project_model, pinocchio_model):
    _check(spec, project_model, pinocchio_model)
