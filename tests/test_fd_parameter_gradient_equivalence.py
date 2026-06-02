"""Pinocchio equivalence for the forward-dynamics inertial-parameter gradient.

Cross-checks `_RegressorMixin.fd_parameter_gradient` (the numpy reference
`dqdd/dpi`, the inertial-parameter gradient of forward dynamics) against an
independent Pinocchio oracle. This closes the C2 gap: the existing
`test_regressor_equivalence.py` only validated the regressor `Y`, never the
`-Minv . Y` compose that `fd_parameter_gradient` returns.

Oracle (derived):
  Forward dynamics solves  M(pi) qddot + c(q, qdot, pi) = u  for fixed input
  torque u. Inverse dynamics  ID(q, qdot, qddot, pi) = M(pi) qddot + c(q, qdot, pi)
  is affine in pi with Jacobian  Y(q, qdot, qddot) = d ID / d pi  (the
  joint-torque regressor) evaluated at the actual acceleration. Differentiating
  the dynamics in pi at fixed u:
      d ID/d pi |_{qddot}  +  M . d qddot/d pi  =  0
  =>  d qddot/d pi  =  - M^{-1} . Y(q, qdot, qddot_actual)
  with  qddot_actual = forward_dynamics(q, qdot, u) = aba(q, qdot, u).

The oracle is assembled from the Pinocchio backend's `minv` (pin.crba inverse /
computeMinverse), `aba` (pin.aba, mimic-reduced), and `joint_torque_regressor`
(pin.computeJointTorqueRegressor, remapped to the project body order + GRiD
param basis) -- each already validated independently by the minv / aba /
regressor equivalence suites -- so this test validates the COMPOSE, not the
pieces. The input torque is grounded as  u = rnea(q, qdot, qddot_sample)  so the
gradient is taken at a physical acceleration; the SAME u feeds both sides.
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


def _pin_fd_parameter_gradient_oracle(pinocchio_model, body_joint_names, q, qd, u):
    """Independent -Minv . Y(q,qd,qdd_actual) oracle from the Pinocchio backend.

    qdd_actual = aba(q,qd,u); Y is remapped to the project body order + GRiD
    param basis; Minv is the reduced project-layout inverse mass matrix.
    """
    qdd_actual = np.asarray(pinocchio_model.aba(q, qd, u), dtype=np.float64)
    Y_pin = pinocchio_model.joint_torque_regressor(
        q, qd, qdd_actual, project_body_joint_names=body_joint_names
    )
    Minv_pin = np.asarray(pinocchio_model.minv(q), dtype=np.float64)
    return -(Minv_pin @ Y_pin)


def _check_fd_parameter_gradient(spec, project_model, pinocchio_model):
    body_joint_names = project_model.body_joint_names
    for sample in build_dynamics_samples(project_model):
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            # Degenerate / zero-inertia model (e.g. rizon4's broken URDF) — the
            # Pinocchio Minv / regressor oracle is non-physical here.
            continue
        # Ground the input torque at a physical acceleration so the gradient is
        # evaluated at qdd_actual = forward_dynamics(q, qd, u) ≈ qdd_sample.
        u = project_model.rnea(q, qd, qdd)

        actual = project_model.fd_parameter_gradient(q, qd, u)
        oracle = _pin_fd_parameter_gradient_oracle(pinocchio_model, body_joint_names, q, qd, u)
        assert_close(actual, oracle, algorithm="fd_param_grad", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_fd_parameter_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    _check_fd_parameter_gradient(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_fd_parameter_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    _check_fd_parameter_gradient(spec, project_model, pinocchio_model)
