"""Equivalence tests for the f_ext gradient column (section A of the
differentiability extensions plan):

  dtau/dfext      = -J^T          (stacked body-Jacobian transpose, local frame)
  dqdd/dfext      =  M^{-1} J^T   (operational-space inverse-inertia map)
  d(id_du)/dfext  = -dJ^T/dq      (q-derivative of the body Jacobian; q-only)

The project numpy oracle (`RBDReference.f_ext_gradient`) builds these by running
the RNEA backward sweep with unit local wrenches (exactly the forward
`apply_external_forces` convention), and is validated here against the pinocchio
backend oracle, which reads pinocchio's exact RNEA-with-unit-fext response and
`computeMinverse`. The first-order blocks (-J^T, M^-1 J^T) are exact; the mixed
second-order block (-dJ^T/dq) is finite-differenced on both sides (FD-of-exact).
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
        # mixed second-order block: FD-of-exact on both sides
        assert_close(
            a_djt, e_djt, algorithm="f_ext_gradient_so", robot_id=spec.robot_id
        )
