"""Equivalence tests for the f_ext gradient column (section A of the
differentiability extensions plan):

  dtau/dfext      = -J^T          (stacked body-Jacobian transpose, local frame)
  dqdd/dfext      =  M^{-1} J^T   (operational-space inverse-inertia map)
  d(id_du)/dfext  = -dJ^T/dq      (q-derivative of the body Jacobian; q-only)

The project numpy oracle (`RBDReference.f_ext_gradient`) builds these by running
the RNEA backward sweep with unit local wrenches (exactly the forward
`apply_external_forces` convention), and is validated here against the pinocchio
backend oracle, which reads pinocchio's exact RNEA-with-unit-fext response and
`computeMinverse`. The first-order blocks (-J^T, M^-1 J^T) are exact. The mixed
second-order block (-dJ^T/dq) is ANALYTIC (closed form) on the project side for
BOTH fixed and floating base (``RBDReference.f_ext_jacobian_transpose_dq``: the
free-flyer root's 6 motion-subspace columns slot into the same Featherstone
``-crm(S)X`` pushdown as the scalar joints) and central-FD-of-exact on the
pinocchio side. The analytic oracle (fixed AND floating) is additionally
self-checked against a central FD of the exact J^T through ``self.integrate`` in
``test_fixed_base_djt_dq_analytic_matches_fd`` (now covering floating too).
"""

import numpy as np
import pytest

from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.state_sampling import build_dynamics_samples


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_f_ext_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    _check(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_f_ext_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    _check(spec, project_model, pinocchio_model)


def _check(spec, project_model, pinocchio_model):
    for sample in build_dynamics_samples(project_model):
        q = sample.q
        a_dtau, a_dqdd, a_djt = project_model.f_ext_gradient(q)
        e_dtau, e_dqdd, e_djt = pinocchio_model.f_ext_gradient(q)
        # first-order blocks are exact
        assert_close(a_dtau, e_dtau, algorithm="f_ext_gradient", robot_id=spec.robot_id)
        assert_close(a_dqdd, e_dqdd, algorithm="f_ext_gradient", robot_id=spec.robot_id)
        # mixed second-order block (-dJ^T/dq): analytic closed form on the
        # project side (fixed AND floating) vs pinocchio FD-of-exact.
        assert_close(
            a_djt, e_djt, algorithm="f_ext_gradient_so", robot_id=spec.robot_id
        )


@pytest.mark.parametrize(
    ("spec", "base_mode"),
    build_case_params(base_mode="fixed") + build_case_params(base_mode="floating"),
)
def test_fixed_base_djt_dq_analytic_matches_fd(spec, base_mode, project_model):
    """The closed-form ``f_ext_jacobian_transpose_dq`` (dJ^T/dq) must agree with a
    central finite difference of the exact J^T (perturbed through ``self.integrate``,
    so the floating free-flyer root is retracted on SE(3)) to FD-truncation
    accuracy — an independent self-consistency check on the analytic oracle, for
    BOTH fixed and floating base, that does not rely on pinocchio. (The
    ``fixed_base`` in the name is retained for the conftest slow-gating substring;
    the test now also covers floating-base cases.)"""
    ref = project_model.reference
    nb = ref.robot.get_num_bodies()
    nv = ref.robot.get_num_vel()
    h = 1e-6
    for sample in build_dynamics_samples(project_model):
        q = sample.q
        analytic = ref.f_ext_jacobian_transpose_dq(q)
        fd = np.zeros((nv, 6 * nb, nv), dtype=np.float64)
        for i in range(nv):
            dv = np.zeros(nv, dtype=np.float64)
            dv[i] = h
            jt_p = ref.f_ext_jacobian_transpose(ref.integrate(q, dv))
            jt_m = ref.f_ext_jacobian_transpose(ref.integrate(q, -dv))
            fd[:, :, i] = (jt_p - jt_m) / (2.0 * h)
        err = float(np.max(np.abs(analytic - fd)))
        assert err < 1e-5, (
            f"{spec.robot_id}: analytic dJ^T/dq vs central FD maxerr={err:.3e}"
        )
