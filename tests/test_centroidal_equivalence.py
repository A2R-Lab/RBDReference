"""Pinocchio equivalence for the CoM / centroidal numpy reference (R2/R3).

Cross-checks the additive `_CentroidalMixin` methods (`com`, `jacobian_com`,
`ccrba`, `centroidal_momentum`) against Pinocchio (`centerOfMass`,
`jacobianCenterOfMass`, `ccrba` -> data.Ag/data.hg, `computeCentroidalMomentum`).
The centroidal momentum is expressed at the CoM in a world-aligned frame,
ordered [linear; angular], matching Pinocchio's data.Ag / data.hg.
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


def _check_centroidal(spec, project_model, pinocchio_model):
    for sample in build_dynamics_samples(project_model):
        q, qd = sample.q, sample.qd
        if not pinocchio_model.has_invertible_mass_matrix(q):
            # Degenerate / zero-inertia model (e.g. rizon4's broken URDF) — the
            # Pinocchio oracle returns NaN CoM/centroidal quantities here, so the
            # equivalence is not well-defined.
            continue
        assert_close(
            project_model.com(q),
            pinocchio_model.com(q),
            algorithm="centroidal", robot_id=spec.robot_id,
        )
        assert_close(
            project_model.jacobian_com(q),
            pinocchio_model.jacobian_com(q),
            algorithm="centroidal", robot_id=spec.robot_id,
        )
        A_ref, h_ref = project_model.ccrba(q, qd)
        A_pin, h_pin = pinocchio_model.ccrba(q, qd)
        assert_close(A_ref, A_pin, algorithm="centroidal", robot_id=spec.robot_id)
        assert_close(h_ref, h_pin, algorithm="centroidal", robot_id=spec.robot_id)
        assert_close(
            project_model.centroidal_momentum(q, qd),
            pinocchio_model.centroidal_momentum(q, qd),
            algorithm="centroidal", robot_id=spec.robot_id,
        )
        # consistency: h == A @ qd
        assert_close(h_ref, A_ref @ np.asarray(qd, dtype=np.float64),
                     algorithm="centroidal", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_centroidal_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_centroidal(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_centroidal_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_centroidal(spec, project_model, pinocchio_model)
