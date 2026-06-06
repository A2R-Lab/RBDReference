"""Structural identities for the Coriolis matrix numpy reference (PS5 task 1).

`test_energy_equivalence.py` already pins `_EnergyMixin.coriolis_matrix` entrywise
to `pin.computeCoriolisMatrix` (the spatial `Bcrb` recursion) for fixed and
floating base. This file adds the two *factorization-independent* physics
identities that the PS5 ask calls out, which hold for ANY valid Coriolis
factorization and so document the convention without re-deriving pinocchio:

  (1) Drift / nonlinear-effects identity:
          C(q,qd) . qd  +  g(q)  ==  nonlinear_effects(q,qd)   (= RNEA(q,qd,0))

  (2) Energy conservation / skew-symmetry:
          M_dot(q,qd) - 2 C(q,qd)   is skew-symmetric,
      equivalently  M_dot == C + C^T  (the symmetric part of C is 1/2 M_dot).
      `M_dot` is taken by central finite-difference of CRBA along qd, using
      pinocchio's `integrate` for the floating base so the quaternion stays on
      the manifold (guide §6: never reuse a pinocchio view across the FD steps).

Both are exact for the project oracle (residuals at the 1e-9 RNEA / 1e-6 FD
floor), confirming pinocchio's factorization: C splits M_dot into its symmetric
half plus a skew Coriolis part, and C.qd reproduces the velocity-product torque.
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


def _mass_matrix_dot_fd(project_model, pinocchio_model, q, qd, base_mode, step=1e-6):
    """M_dot(q, qd) by central FD of CRBA along the qd direction.

    Floating base advances q on the manifold via pinocchio `integrate` (so the
    quaternion block stays unit-norm); fixed base is a plain q +/- step*qd. Fresh
    arrays per evaluation — no aliased pinocchio view crosses the two steps.
    """
    ref = project_model.reference
    if base_mode == "floating":
        q_plus = pinocchio_model._pin_integrate(q, step * qd)
        q_minus = pinocchio_model._pin_integrate(q, -step * qd)
    else:
        q_plus = np.asarray(q, dtype=np.float64) + step * qd
        q_minus = np.asarray(q, dtype=np.float64) - step * qd
    M_plus = np.asarray(ref.crba(q_plus), dtype=np.float64)
    M_minus = np.asarray(ref.crba(q_minus), dtype=np.float64)
    return (M_plus - M_minus) / (2.0 * step)


def _check_coriolis_identities(spec, project_model, pinocchio_model, base_mode):
    ref = project_model.reference
    for sample in build_dynamics_samples(project_model):
        q, qd = sample.q, sample.qd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            continue
        C = np.asarray(ref.coriolis_matrix(q, qd), dtype=np.float64)
        g = np.asarray(ref.generalized_gravity(q), dtype=np.float64)
        nle = np.asarray(ref.nonlinear_effects(q, qd), dtype=np.float64)

        # (1) drift identity: exact (both sides are RNEA compositions).
        assert_close(C @ qd + g, nle, algorithm="inverse_dynamics", robot_id=spec.robot_id)

        # (2) skew-symmetry of M_dot - 2C  <=>  M_dot == C + C^T (FD-limited, so
        # use the standard "FD of a verified first-order quantity" bucket).
        M_dot = _mass_matrix_dot_fd(project_model, pinocchio_model, q, qd, base_mode)
        skew = M_dot - 2.0 * C
        assert_close(skew, -skew.T, algorithm="idsva_so_body_frame", robot_id=spec.robot_id)
        assert_close(M_dot, C + C.T, algorithm="idsva_so_body_frame", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_coriolis_identities(spec, base_mode, project_model, pinocchio_model):
    _check_coriolis_identities(spec, project_model, pinocchio_model, base_mode)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_coriolis_identities(spec, base_mode, project_model, pinocchio_model):
    _check_coriolis_identities(spec, project_model, pinocchio_model, base_mode)
