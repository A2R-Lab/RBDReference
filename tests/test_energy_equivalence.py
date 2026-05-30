"""Pinocchio equivalence for the energy / gravity / Coriolis numpy reference (R1).

Cross-checks the additive `_EnergyMixin` methods (`generalized_gravity`,
`nonlinear_effects`, `kinetic_energy`, `potential_energy`, `mechanical_energy`,
`coriolis_matrix`) against the corresponding Pinocchio calls
(`computeGeneralizedGravity`, `nonLinearEffects`, `computeKineticEnergy`,
`computePotentialEnergy`, `computeMechanicalEnergy`, `computeCoriolisMatrix`).
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


def _check_energy(spec, project_model, pinocchio_model, coriolis: bool):
    for sample in build_dynamics_samples(project_model):
        q, qd = sample.q, sample.qd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            # Degenerate / zero-inertia model (e.g. rizon4's broken URDF) — the
            # Pinocchio oracle returns non-physical (NaN) energy/gravity, so the
            # equivalence is not well-defined for this configuration.
            continue
        assert_close(
            project_model.generalized_gravity(q),
            pinocchio_model.generalized_gravity(q),
            algorithm="energy", robot_id=spec.robot_id,
        )
        assert_close(
            project_model.nonlinear_effects(q, qd),
            pinocchio_model.nonlinear_effects(q, qd),
            algorithm="energy", robot_id=spec.robot_id,
        )
        assert_close(
            project_model.kinetic_energy(q, qd),
            pinocchio_model.kinetic_energy(q, qd),
            algorithm="energy", robot_id=spec.robot_id,
        )
        assert_close(
            project_model.potential_energy(q),
            pinocchio_model.potential_energy(q),
            algorithm="energy", robot_id=spec.robot_id,
        )
        assert_close(
            project_model.mechanical_energy(q, qd),
            pinocchio_model.mechanical_energy(q, qd),
            algorithm="energy", robot_id=spec.robot_id,
        )
        if coriolis:
            assert_close(
                project_model.coriolis_matrix(q, qd),
                pinocchio_model.coriolis_matrix(q, qd),
                algorithm="energy", robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_energy_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    # Coriolis matrix reference is implemented for fixed-base.
    _check_energy(spec, project_model, pinocchio_model, coriolis=True)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_energy_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    # Skip the Coriolis matrix on floating-base (reference is fixed-base only).
    _check_energy(spec, project_model, pinocchio_model, coriolis=False)
