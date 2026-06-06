"""Energy regressors + body-regressor self-consistency (PS5 task 3 gap fill).

The regressor family already shipped `inverse_dynamics_regressor` (Y: tau=Y.pi)
and `forward_dynamics_parameter_gradient` (-Minv.Y), each validated vs pinocchio
elsewhere. This file fills the two obvious remaining regressor gaps that
pinocchio also exposes:

  * `kinetic_energy_regressor`   vs `pin.computeKineticEnergyRegressor`
  * `potential_energy_regressor` vs `pin.computePotentialEnergyRegressor`

Both are 1 x 10*NB rows (KE / PE are each affine in the stacked standard inertial
parameters pi, GRiD basis [m, h(3), I_O(6)]). We check the structural identity
(y . pi == kinetic_energy / potential_energy, exact) and the pinocchio
cross-check (remapped to the project body order + GRiD param basis, same machinery
as the joint-torque regressor). We also add a direct unit check of the per-link
`body_regressor` (f = Y_body . pi == I a + v x* (I v)), which previously was only
exercised transitively through the joint-torque regressor.

Convention notes:
  * KE regressor uses the RNEA-forward-pass per-link spatial velocity v_i (local
    frame); 1/2 v_i^T I_i v_i is frame-invariant, so it matches pinocchio's value.
  * PE regressor uses g = [0, 0, GRAVITY] and PE = -sum_i g . (m_i p_i + R_i h_i);
    only the mass + first-moment columns are nonzero (PE is independent of I).
    The floating-base PE itself carries ~1e-7 cross-library round-off (same as
    the energy bucket), so the pinocchio cross-check uses the `energy` bucket
    while the exact y.pi structural identity uses the tight regressor bucket.
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
    """Stack each project body's 10 standard inertial params, GRiD/URDF basis
    [m, h(3), Ixx, Ixy, Ixz, Iyy, Iyz, Izz]."""
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


def _check_body_regressor(spec, project_model):
    """Per-link f = Y_body . pi == I a + v x* (I v), independent of pinocchio."""
    ref = project_model.reference
    robot = project_model.robot
    rng = np.random.default_rng(abs(hash(spec.robot_id)) % (2**32))
    for b in range(robot.get_num_bodies()):
        I = np.asarray(robot.get_Imat_by_id(b), dtype=np.float64)
        v = rng.standard_normal(6)
        a = rng.standard_normal(6)
        pi_b = np.zeros(10, dtype=np.float64)
        m = I[5, 5]
        mc = I[:3, 3:6]
        h = np.array([mc[2, 1], mc[0, 2], mc[1, 0]], dtype=np.float64)
        I_O = I[:3, :3]
        pi_b[:] = [m, h[0], h[1], h[2], I_O[0, 0], I_O[0, 1], I_O[0, 2], I_O[1, 1], I_O[1, 2], I_O[2, 2]]
        Y_body = ref.body_regressor(v, a)
        f_expected = I @ a + ref.dual_cross_operator(v) @ (I @ v)
        assert_close(Y_body @ pi_b, f_expected, algorithm="inverse_dynamics", robot_id=spec.robot_id)


def _check_energy_regressors(spec, project_model, pinocchio_model):
    ref = project_model.reference
    pi = _project_pi(project_model)
    body_joint_names = project_model.body_joint_names
    for sample in build_dynamics_samples(project_model):
        q, qd = sample.q, sample.qd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            continue

        y_ke = np.asarray(ref.kinetic_energy_regressor(q, qd), dtype=np.float64)
        y_pe = np.asarray(ref.potential_energy_regressor(q), dtype=np.float64)

        # structural identity (exact): y . pi == energy
        assert_close(float(y_ke @ pi), project_model.kinetic_energy(q, qd),
                     algorithm="inverse_dynamics_regressor", robot_id=spec.robot_id)
        assert_close(float(y_pe @ pi), project_model.potential_energy(q),
                     algorithm="inverse_dynamics_regressor", robot_id=spec.robot_id)

        # vs pinocchio (remapped to project body order + GRiD param basis)
        y_ke_pin = pinocchio_model.kinetic_energy_regressor(
            q, qd, project_body_joint_names=body_joint_names
        )
        y_pe_pin = pinocchio_model.potential_energy_regressor(
            q, project_body_joint_names=body_joint_names
        )
        assert_close(y_ke, y_ke_pin, algorithm="inverse_dynamics_regressor", robot_id=spec.robot_id)
        # PE carries the floating-base energy round-off (~1e-7); use energy bucket.
        assert_close(y_pe, y_pe_pin, algorithm="energy", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_energy_regressors_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_body_regressor(spec, project_model)
    _check_energy_regressors(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_energy_regressors_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_body_regressor(spec, project_model)
    _check_energy_regressors(spec, project_model, pinocchio_model)
