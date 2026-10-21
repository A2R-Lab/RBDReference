"""External forces as an input of the value and first-order surfaces, against
pinocchio: inverse dynamics, forward dynamics, ABA and the inverse-dynamics
gradients with a nonzero local wrench on every body, mimic-driven bodies included.

The dedicated f_ext gradient column (dtau/dfext, dqdd/dfext, -dJ^T/dq) is covered
by test_f_ext_gradient_equivalence.py; this module checks the f_ext argument
threaded through the existing algorithms. The torque that drives the forward
dynamics is the project's own inverse dynamics of the sample (u = ID(q, qd, qdd)
without the wrench), so the step is float-meaningful on every robot.

Before 2026-10-20 the pinocchio oracle placed the wrenches through the non-mimic
joint-name list, so on a robot whose mimic joints sit mid-chain (h1_2) every body
after the first mimic joint received its neighbour's wrench and the mimic bodies
received none; GRiD's CUDA three-way check therefore left the pinocchio leg out
for mimic robots. The oracle now maps every project body by name, mimic joints
included, the same way its f_ext gradient column does.
"""
import numpy as np
import pytest

from RBDReference.tests.comparators import assert_close
from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.state_sampling import build_dynamics_samples

_WRENCH_SEED = 20260530  # the draw GRiD's CUDA f_ext module uses


def _wrenches(project_model):
    nb = project_model.reference.robot.get_num_bodies()
    rng = np.random.default_rng(_WRENCH_SEED)
    return [rng.uniform(-3.0, 3.0, size=6) for _ in range(nb)]


def _check(spec, project_model, pinocchio_model):
    f_ext = _wrenches(project_model)
    robot_id = spec.robot_id
    for sample in build_dynamics_samples(project_model):
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            pytest.skip(
                f"{robot_id} mass matrix is singular (degenerate/zero-inertia URDF), "
                "so the forward dynamics with external forces is not well-defined."
            )
        assert_close(
            project_model.inverse_dynamics(q, qd, qdd, f_ext=f_ext),
            pinocchio_model.inverse_dynamics(q, qd, qdd, f_ext=f_ext),
            algorithm="inverse_dynamics", robot_id=robot_id,
        )
        u = np.asarray(project_model.inverse_dynamics(q, qd, qdd), dtype=np.float64)
        expected_qdd = pinocchio_model.aba(q, qd, u, f_ext=f_ext)
        assert_close(
            project_model.forward_dynamics(q, qd, u, f_ext=f_ext), expected_qdd,
            algorithm="forward_dynamics", robot_id=robot_id,
        )
        assert_close(
            project_model.aba(q, qd, u, f_ext=f_ext), expected_qdd,
            algorithm="aba", robot_id=robot_id,
        )
        a_dq, a_dqd = project_model.inverse_dynamics_gradient(q, qd, qdd, f_ext=f_ext)
        e_dq, e_dqd = pinocchio_model.inverse_dynamics_gradient(q, qd, qdd, f_ext=f_ext)
        assert_close(a_dq, e_dq, algorithm="inverse_dynamics_gradient", robot_id=robot_id)
        assert_close(a_dqd, e_dqd, algorithm="inverse_dynamics_gradient", robot_id=robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_external_forces_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_external_forces_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check(spec, project_model, pinocchio_model)
