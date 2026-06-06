"""Pinocchio / FD equivalence for the dCCRBA layer of `_centroidal.py`.

Targets the two derivative-of-the-centroidal-map accessors:

  * `cmm_time_variation(q, qd)` -> `Adot = dAg/dt`   (6 x nv), and
  * `dccrba(q)`                 -> `dA_dq[:, k, i]`   (6 x nv x nv),

both in the Pinocchio centroidal convention ([linear; angular] at the CoM,
world-aligned). These are validated against:

  * `pin.dccrba(model, data, q, v)` for `Adot` (exact analytic oracle), and
  * a 4th-order central finite difference of the validated value-layer `ccrba`
    along the Lie-group tangent for both `Adot` and the full `dA_dq` tensor
    (float64, `copy=True` on every sampled `A` to dodge the pin-view aliasing
    trap, guide §6).

Cross-checks the two contractions of the tensor:
    Adot  == sum_i dA_dq[:, :, i] qd[i]
    dh_dq == sum_k dA_dq[:, k, :] qd[k]   (vs pin.computeCentroidalDynamicsDerivatives)

Covered on iiwa14 (fixed) + go2/g1 (floating). Mimic robots (fr3, h1_2) are
skipped: pinocchio's reduced model has a different nv than the project surface,
so `pin.dccrba`'s column layout would need the mimic column-folding the value
layer already exercises elsewhere; the dCCRBA tensor here is validated on the
non-mimic robots only and noted as a follow-up.
"""

import numpy as np
import pytest

from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.state_sampling import build_dynamics_samples


# (robot_id, base_mode): the non-mimic iiwa14 + a floating quadruped/humanoid.
_DCCRBA_CASES = [
    ("iiwa14", "fixed"),
    ("go2", "floating"),
    ("g1", "floating"),
]

# A few samples (zero / conservative / one high-energy) span the regimes; the
# tensor build is O(nv) value-layer evals per DOF, so keep it bounded.
_SAMPLE_LIMIT = 3
_FD_STEP = 1e-5


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


def _pin_dccrba(pin_model, q, qd):
    import pinocchio as pin

    q_pin = pin_model._to_pin_q(q)
    v_pin = pin_model._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
    return np.asarray(pin.dccrba(pin_model.model, pin_model.data, q_pin, v_pin),
                      dtype=np.float64)


def _pin_dh_dq(pin_model, q, qd, qdd):
    import pinocchio as pin

    q_pin = pin_model._to_pin_q(q)
    v_pin = pin_model._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
    a_pin = pin_model._expand_project_v_to_pin(np.asarray(qdd, dtype=np.float64))
    dh_dq, *_ = pin.computeCentroidalDynamicsDerivatives(
        pin_model.model, pin_model.data, q_pin, v_pin, a_pin
    )
    return np.asarray(dh_dq, dtype=np.float64)


def _fd_dA_dq(ref, q, fd_step=_FD_STEP):
    """4th-order central FD of the value-layer ccrba: dA_dq[:, k, i]."""
    nv = ref.robot.get_num_vel()
    zero_v = np.zeros(nv, dtype=np.float64)
    dA = np.zeros((6, nv, nv), dtype=np.float64)
    for i in range(nv):
        e = np.zeros(nv, dtype=np.float64)
        e[i] = fd_step

        def _A(scale):
            A, _h = ref.ccrba(ref.integrate(q, scale * e), zero_v)
            return np.asarray(A, dtype=np.float64).copy()

        dA[:, :, i] = (-_A(2.0) + 8.0 * _A(1.0) - 8.0 * _A(-1.0) + _A(-2.0)) / (
            12.0 * fd_step
        )
    return dA


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

    # Mimic guard: pin's reduced nv must match the project nv for a direct
    # column-by-column comparison (no mimic-folding needed for these cases).
    mi = getattr(pin_model, "mimic_info", None)
    if mi is not None and not mi.is_empty():
        pytest.skip(
            f"{robot_id}: mimic robot — pin.dccrba column layout needs mimic "
            "folding; dCCRBA tensor validated on non-mimic robots only (noted)."
        )

    for idx, sample in enumerate(build_dynamics_samples(proj_model)):
        if idx >= _SAMPLE_LIMIT:
            break
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        if not pin_model.has_invertible_mass_matrix(q):
            continue  # degenerate / zero-inertia URDF — oracle is NaN

        qd_arr = np.asarray(qd, dtype=np.float64)

        # ---- Adot = cmm_time_variation vs pin.dccrba (exact analytic oracle) ----
        Adot = ref.cmm_time_variation(q, qd)
        Adot_pin = _pin_dccrba(pin_model, q, qd)
        assert Adot.shape == (6, nv)
        assert_close(Adot, Adot_pin, algorithm="centroidal_grad", robot_id=robot_id)

        # ---- dA_dq tensor vs FD-of-ccrba (tight) ----
        dA = ref.dccrba(q)
        assert dA.shape == (6, nv, nv)
        dA_fd = _fd_dA_dq(ref, q)
        assert_close(dA, dA_fd, algorithm="centroidal_grad", robot_id=robot_id)

        # ---- contraction consistency: tensor -> Adot and tensor -> dh_dq ----
        Adot_from_tensor = np.einsum("abi,i->ab", dA, qd_arr)
        assert_close(Adot_from_tensor, Adot_pin,
                     algorithm="centroidal_grad", robot_id=robot_id)

        dh_dq = np.einsum("abi,b->ai", dA, qd_arr)
        assert_close(dh_dq, _pin_dh_dq(pin_model, q, qd, qdd),
                     algorithm="centroidal_grad", robot_id=robot_id)

        # ---- Adot @ qd == the centroidal bias (the Adot-qd term of hdot) ----
        bias = ref._centroidal_bias(q, qd)
        assert_close(Adot @ qd_arr, bias,
                     algorithm="centroidal_grad", robot_id=robot_id)
