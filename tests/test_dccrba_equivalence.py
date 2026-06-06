"""Pinocchio / FD equivalence for the dCCRBA layer of `_centroidal.py`.

Targets the two derivative-of-the-centroidal-map accessors:

  * `cmm_time_variation(q, qd)` -> `Adot = dAg/dt`   (6 x nv), and
  * `dccrba(q)`                 -> `dA_dq[:, k, i]`   (6 x nv x nv),

both in the Pinocchio centroidal convention ([linear; angular] at the CoM,
world-aligned). Both are now computed ANALYTICALLY (`_dccrba_analytic`: one
world-frame sweep + spatial cross-product / inertia-derivative operators,
mimic-aware, valid for fixed- and floating-base). They are validated against:

  * `pin.dccrba(model, data, q, v)` for `Adot` (exact analytic oracle), and
  * the 4th-order central FD cross-checks `dccrba_fd` / `cmm_time_variation_fd`
    of the validated value-layer `ccrba` (float64, `copy=True` on every sampled
    `A` to dodge the pin-view aliasing trap, guide §6).

Cross-checks the two contractions of the analytic tensor:
    Adot  == sum_i dA_dq[:, :, i] qd[i]   (vs pin.dccrba)
    dh_dq == sum_k dA_dq[:, k, :] qd[k]   (vs pin.computeCentroidalDynamicsDerivatives)

Covered on iiwa14 (fixed) + go2/g1 (floating) + fr3/h1_2 (mimic). For mimic
robots the pinocchio oracle's v-axis is folded into the project layout via
`_reduce_pin_matrix_to_project` (the analytic side is already mimic-aware).
"""

import numpy as np
import pytest

from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.state_sampling import build_dynamics_samples


# (robot_id, base_mode): iiwa14 fixed FIRST, then floating quadruped/humanoid,
# then the two mimic robots (fr3 fixed, h1_2 fixed). The analytic dCCRBA is
# mimic-aware, so the mimic robots are no longer skipped — pin's wider-nv
# oracle is folded to the project layout for the column comparison.
_DCCRBA_CASES = [
    ("iiwa14", "fixed"),
    ("go2", "floating"),
    ("g1", "floating"),
    ("fr3", "fixed"),
    ("h1_2", "fixed"),
]

# A few samples (zero / conservative / one high-energy) span the regimes; the
# analytic tensor is one world sweep + cross-products per DOF, so keep bounded.
_SAMPLE_LIMIT = 3


def _build_pair(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        if spec.robot_id != robot_id:
            continue
        resolved = resolve_robot_spec(spec)
        pin_model = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
        proj_model = build_project_adapter(spec, resolved, base_mode=base_mode)
        return spec, pin_model, proj_model
    pytest.skip(f"robot {robot_id} ({base_mode}) not in manifest for this base mode")


def _pin_dh_dq(pin_model, q, qd, qdd):
    """pin's dh_dq = d(A qd)/dq, folded to the project v-layout for mimic."""
    import pinocchio as pin

    q_pin = pin_model._to_pin_q(q)
    v_pin = pin_model._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
    a_pin = pin_model._expand_project_v_to_pin(np.asarray(qdd, dtype=np.float64))
    dh_dq, *_ = pin.computeCentroidalDynamicsDerivatives(
        pin_model.model, pin_model.data, q_pin, v_pin, a_pin
    )
    dh_dq = np.asarray(dh_dq, dtype=np.float64)
    if pin_model.mimic_info is not None and not pin_model.mimic_info.is_empty():
        dh_dq = pin_model._reduce_pin_matrix_to_project(dh_dq, axes_to_reduce=[(1, "v")])
    return np.asarray(dh_dq, dtype=np.float64)


@pytest.mark.pinocchio_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    ("robot_id", "base_mode"),
    _DCCRBA_CASES,
    ids=[f"{r}-{b}" for r, b in _DCCRBA_CASES],
)
def test_dccrba_matches_pinocchio_and_fd(robot_id, base_mode):
    spec, pin_model, proj_model = _build_pair(robot_id, base_mode)
    ref = proj_model.reference
    nv = ref.robot.get_num_vel()

    for idx, sample in enumerate(build_dynamics_samples(proj_model)):
        if idx >= _SAMPLE_LIMIT:
            break
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pin_model.has_invertible_mass_matrix(q):
            continue  # degenerate / zero-inertia URDF — oracle is NaN

        qd_arr = np.asarray(qd, dtype=np.float64)

        # ---- Adot = analytic cmm_time_variation vs pin.dccrba (exact oracle) ----
        Adot = ref.cmm_time_variation(q, qd)
        Adot_pin = pin_model.dccrba(q, qd)
        assert Adot.shape == (6, nv)
        assert_close(Adot, Adot_pin, algorithm="dccrba", robot_id=robot_id)

        # ---- analytic Adot vs its own FD cross-check (value layer is exact) ----
        assert_close(Adot, ref.cmm_time_variation_fd(q, qd),
                     algorithm="dccrba", robot_id=robot_id)

        # ---- analytic dA_dq tensor vs FD-of-ccrba (tight) ----
        dA = ref.dccrba(q)
        assert dA.shape == (6, nv, nv)
        assert_close(dA, ref.dccrba_fd(q),
                     algorithm="dccrba", robot_id=robot_id)

        # ---- contraction consistency: tensor -> Adot and tensor -> dh_dq ----
        Adot_from_tensor = np.einsum("abi,i->ab", dA, qd_arr)
        assert_close(Adot_from_tensor, Adot_pin,
                     algorithm="dccrba", robot_id=robot_id)

        dh_dq = np.einsum("abi,b->ai", dA, qd_arr)
        assert_close(dh_dq, _pin_dh_dq(pin_model, q, qd, qdd),
                     algorithm="dccrba", robot_id=robot_id)

        # ---- Adot @ qd == the centroidal bias (the Adot-qd term of hdot) ----
        bias = ref._centroidal_bias(q, qd)
        assert_close(Adot @ qd_arr, bias,
                     algorithm="dccrba", robot_id=robot_id)
