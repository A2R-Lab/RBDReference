import pytest

from RBDReference.tests.comparators import assert_close
from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.state_sampling import build_dynamics_samples


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_inverse_dynamics_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.inverse_dynamics(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.inverse_dynamics(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="inverse_dynamics", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_inverse_dynamics_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.inverse_dynamics(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.inverse_dynamics(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="inverse_dynamics", robot_id=spec.robot_id)
