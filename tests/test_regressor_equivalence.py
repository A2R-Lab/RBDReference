"""Pinocchio equivalence for the joint-torque regressor numpy reference (sysID).

Cross-checks the additive `_RegressorMixin.inverse_dynamics_regressor` against
`pin.computeJointTorqueRegressor`. Mind the two conventions reconciled here:
  - the per-link 10-param block ORDER (project body id <-> pin joint id; the
    floating root maps to pin's root_joint), and
  - the inertia-entry BASIS permutation inside each 10-block (GRiD/URDF
    [Ixx,Ixy,Ixz,Iyy,Iyz,Izz] vs pin [Ixx,Ixy,Iyy,Ixz,Iyz,Izz]).
Both are handled by `PinocchioModelAdapter.inverse_dynamics_regressor` when given
the project's body-joint-name ordering, so the two matrices line up column for
column. Also verifies the structural identity `Y @ pi == inverse_dynamics(q,qd,qdd)`.
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


def _project_pi(project_model):
    """Stack each project body's 10 standard inertial params in the GRiD/URDF
    basis [m, h(3), Ixx, Ixy, Ixz, Iyy, Iyz, Izz]."""
    robot = project_model.robot
    nb = robot.get_num_bodies()
    pi = np.zeros(10 * nb, dtype=np.float64)
    for b in range(nb):
        Ib = np.asarray(robot.get_Imat_by_id(b), dtype=np.float64)
        m = Ib[5, 5]
        mc_skew = Ib[:3, 3:6]
        h = np.array([mc_skew[2, 1], mc_skew[0, 2], mc_skew[1, 0]], dtype=np.float64)
        I_O = Ib[:3, :3]
        pi[10 * b:10 * b + 10] = [
            m, h[0], h[1], h[2],
            I_O[0, 0], I_O[0, 1], I_O[0, 2], I_O[1, 1], I_O[1, 2], I_O[2, 2],
        ]
    return pi


def _check_regressor(spec, project_model, pinocchio_model):
    pi = _project_pi(project_model)
    body_joint_names = project_model.body_joint_names
    for sample in build_dynamics_samples(project_model):
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            # Degenerate / zero-inertia model (e.g. rizon4's broken URDF) — the
            # Pinocchio regressor / RNEA oracle is non-physical here.
            continue
        Y_ref = project_model.inverse_dynamics_regressor(q, qd, qdd)
        # structural identity: Y @ pi == inverse_dynamics
        tau = project_model.inverse_dynamics(q, qd, qdd)
        assert_close(Y_ref @ pi, tau, algorithm="inverse_dynamics", robot_id=spec.robot_id)
        # vs pinocchio (remapped to the project body order + GRiD param basis)
        Y_pin = pinocchio_model.inverse_dynamics_regressor(
            q, qd, qdd, project_body_joint_names=body_joint_names
        )
        assert_close(Y_ref, Y_pin, algorithm="inverse_dynamics_regressor", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_regressor_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_regressor(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_regressor_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_regressor(spec, project_model, pinocchio_model)
