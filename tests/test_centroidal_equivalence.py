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


# The C2 centroidal-derivative blocks are central finite differences of the
# exact value layer (3*nv perturbations, each re-running the full ccrba + bias
# composition), so they are O(nv^2) per sample — too costly to run on all 7
# energetic samples for the big humanoids. Cover them on the first few samples
# (zero + conservative + one high-energy) which already span the low/high-energy
# regimes; the value layer (A/h/com/Jcom) still runs on every sample.
_DERIV_SAMPLE_LIMIT = 3


def _check_centroidal(spec, project_model, pinocchio_model):
    for sample_idx, sample in enumerate(build_dynamics_samples(project_model)):
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

        # ---- centroidal rate hdot = A qdd + Adot qd (vs pin) ----
        qdd = sample.qdd
        assert_close(
            project_model.centroidal_momentum_time_variation(q, qd, qdd),
            pinocchio_model.centroidal_momentum_time_variation(q, qd, qdd),
            algorithm="centroidal", robot_id=spec.robot_id,
        )

        if sample_idx >= _DERIV_SAMPLE_LIMIT:
            continue

        # ---- centroidal dynamics derivatives (C2) ----
        # (dh_dq, dhdot_dq, dhdot_dv, dhdot_da) vs
        # pin.computeCentroidalDynamicsDerivatives. dh_dq = d(A qd)/dq is the C2
        # deliverable; dhdot_da == A is exact. The first three blocks are
        # FD-sourced (the 'centroidal_grad' bucket); dhdot_da is exact (the
        # tight 'centroidal' bucket).
        dh_dq, dhdot_dq, dhdot_dv, dhdot_da = project_model.centroidal_dynamics_derivatives(q, qd, qdd)
        p_dh_dq, p_dhdot_dq, p_dhdot_dv, p_dhdot_da = pinocchio_model.centroidal_dynamics_derivatives(q, qd, qdd)
        assert_close(dh_dq, p_dh_dq, algorithm="centroidal_grad", robot_id=spec.robot_id)
        assert_close(dhdot_dq, p_dhdot_dq, algorithm="centroidal_grad", robot_id=spec.robot_id)
        assert_close(dhdot_dv, p_dhdot_dv, algorithm="centroidal_grad", robot_id=spec.robot_id)
        assert_close(dhdot_da, p_dhdot_da, algorithm="centroidal", robot_id=spec.robot_id)
        # dhdot_da is exactly the CMM A.
        assert_close(dhdot_da, A_ref, algorithm="centroidal", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
def test_fixed_base_centroidal_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_centroidal(spec, project_model, pinocchio_model)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
def test_floating_base_centroidal_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _check_centroidal(spec, project_model, pinocchio_model)
