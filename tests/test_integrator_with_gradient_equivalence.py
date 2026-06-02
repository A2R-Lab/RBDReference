"""Standalone numpy equivalence test for the COMBINED integrator-with-gradient
surface.

The GRiD CUDA codegen exposes ``integrator_with_gradient`` — a single device
pass that computes BOTH the next state ``x_{k+1}`` AND the integrator Jacobian
``[A | B]`` (see ``_integrator_gradient.py`` ``compute_x_kp1=True``). Its numpy
oracle is the pairing ``(RBDReference.integrator, RBDReference.integrator_grad)``:
the CUDA equivalence harness diffs the combined kernel's ``x_kp1`` block against
``integrator(...)`` and its ``dAB`` block against ``integrator_gradient(...)``
(see ``test/cuda_equivalents/test_cuda_integrator_equivalence.py``).

Until now that pairing was only exercised TRANSITIVELY (through the CUDA
harness). This adds a standalone, GPU-free numpy check that the combined
surface is self-consistent:

1. the combined ``x_{k+1}`` equals the plain ``integrator`` reference, and
2. the combined gradient block equals the ``integrator_gradient`` reference,

for the same ``(q, qd, u, dt, integrator_type)``. To make this more than a
tautology (both blocks ultimately come from the same numpy module), we also
finite-difference the plain ``integrator`` and assert the combined gradient
block IS the true Jacobian of the SAME step that produced the combined
``x_{k+1}`` (fixed-base, where the tangent space is trivial so a plain
component-wise FD is exact). This ties the two outputs together: a kernel that
emitted a correct ``x_{k+1}`` but a stale/mismatched ``dAB`` (or vice-versa)
would be caught.

Robots: iiwa14 (small fixed), go2 (floating), fr3 (mimic: NB=9 > NV=8 fixed,
14-DoF floating). Methods: euler + semi_implicit_euler + a multi-stage RK
(rk4). Tolerances reuse the per-robot ``rnea`` bucket, matching the existing
integrator equivalence + FD-sanity suites.
"""

from __future__ import annotations

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


# euler + semi-implicit euler + a multi-stage RK (rk4). Both single-stage
# variants (with distinct q/v ordering) and the chained multi-stage path are
# covered; the CUDA combined kernel templates over IntegratorType for all of
# these.
_INTEGRATORS = ("euler", "semi_implicit_euler", "rk4")

# (robot_id, base_mode). fr3-fixed is the mimic case (NB > NV).
_CASES = (
    ("iiwa14", "fixed"),
    ("go2", "floating"),
    ("fr3", "floating"),
    ("fr3", "fixed"),
)

_DT = 0.01


def _build_adapter(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        if spec.robot_id != robot_id:
            continue
        try:
            resolved = resolve_robot_spec(spec)
        except RuntimeError as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"could not resolve {robot_id}-{base_mode}: {exc}")
        return build_project_adapter(spec, resolved, base_mode=base_mode), spec
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _combined_integrator_with_gradient(pm, q, qd, u, dt, integrator_type):
    """The numpy stand-in for the CUDA ``integrator_with_gradient`` kernel:
    compute BOTH outputs from the canonical references the CUDA harness diffs
    against. Returns ``(x_kp1, dAB)``."""
    x_kp1 = pm.integrator(q, qd, u, dt, integrator_type=integrator_type)
    dAB = pm.integrator_gradient(q, qd, u, dt, integrator_type=integrator_type)
    return np.asarray(x_kp1, dtype=np.float64), np.asarray(dAB, dtype=np.float64)


def _samples(pm, seed=11):
    """Zero + a couple of bounded-random states. Floating-base configs carry a
    unit quaternion in q[3:7]."""
    rng = np.random.default_rng(seed)
    nq, nv = pm.nq, pm.nv
    out = []

    q0 = np.zeros(nq)
    if pm.base_mode == "floating":
        q0[6] = 1.0  # quaternion w
    out.append((q0, np.zeros(nv), np.zeros(nv), "zero"))

    for k in range(2):
        q = np.zeros(nq)
        if pm.base_mode == "floating":
            q[0:3] = rng.uniform(-0.25, 0.25, size=3)
            quat = rng.uniform(-1.0, 1.0, size=4)
            quat /= np.linalg.norm(quat)
            q[3:7] = quat
            q[7:] = rng.uniform(-0.4, 0.4, size=nq - 7)
        else:
            q[:] = rng.uniform(-0.4, 0.4, size=nq)
        qd = rng.uniform(-1.0, 1.0, size=nv)
        u = rng.uniform(-0.5, 0.5, size=nv)
        out.append((q, qd, u, f"random_{k}"))
    return out


@pytest.mark.developer_only
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
@pytest.mark.parametrize(
    ("robot_id", "base_mode"), _CASES, ids=lambda v: v if isinstance(v, str) else None
)
def test_combined_matches_separate_references(robot_id, base_mode, integrator_type):
    """The combined ``integrator_with_gradient`` surface (x_{k+1} + dAB) must
    agree, block-for-block, with the separate ``integrator`` and
    ``integrator_gradient`` references."""
    pm, spec = _build_adapter(robot_id, base_mode)
    nq, nv = pm.nq, pm.nv

    max_x_err = 0.0
    max_g_err = 0.0
    for q, qd, u, name in _samples(pm):
        x_kp1, dAB = _combined_integrator_with_gradient(
            pm, q, qd, u, _DT, integrator_type
        )
        assert x_kp1.shape == (nq + nv,), (
            f"{robot_id}-{base_mode} {integrator_type} x_kp1 shape {x_kp1.shape} "
            f"(expected {(nq + nv,)})"
        )
        assert dAB.shape == (2 * nv, 3 * nv), (
            f"{robot_id}-{base_mode} {integrator_type} dAB shape {dAB.shape} "
            f"(expected {(2 * nv, 3 * nv)})"
        )

        # 1) combined x_{k+1} == plain integrator reference
        expected_x = np.asarray(
            pm.integrator(q, qd, u, _DT, integrator_type=integrator_type),
            dtype=np.float64,
        )
        assert_close(x_kp1, expected_x, algorithm="rnea", robot_id=robot_id)

        # 2) combined gradient block == integrator_gradient reference
        expected_g = np.asarray(
            pm.integrator_gradient(q, qd, u, _DT, integrator_type=integrator_type),
            dtype=np.float64,
        )
        assert_close(dAB, expected_g, algorithm="rnea", robot_id=robot_id)

        denom_x = max(np.max(np.abs(expected_x)), 1e-12)
        denom_g = max(np.max(np.abs(expected_g)), 1e-12)
        max_x_err = max(max_x_err, float(np.max(np.abs(x_kp1 - expected_x)) / denom_x))
        max_g_err = max(max_g_err, float(np.max(np.abs(dAB - expected_g)) / denom_g))

    print(
        f"[combined-consistency] {robot_id}-{base_mode} {integrator_type}: "
        f"max_rel_x={max_x_err:.3e} max_rel_grad={max_g_err:.3e}"
    )


@pytest.mark.developer_only
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
@pytest.mark.parametrize(
    ("robot_id", "base_mode"),
    tuple(c for c in _CASES if c[1] == "fixed"),
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_combined_gradient_is_jacobian_of_combined_state(
    robot_id, base_mode, integrator_type
):
    """Tie the two combined outputs together (fixed-base): the combined
    gradient block must be the true Jacobian of the SAME ``integrator`` step
    that produces the combined ``x_{k+1}``. Caught by a centered finite
    difference of the plain integrator — independent of the analytical
    gradient formulas. (Fixed-base only: the tangent space is trivial so a
    component-wise FD on q/qd/u is exact; floating-base FD needs an SE(3) log
    and is covered by ``test_integrator_pinocchio_equivalence.py``.)"""
    pm, spec = _build_adapter(robot_id, base_mode)
    nq, nv = pm.nq, pm.nv
    assert nq == nv, f"{robot_id} fixed-base expected nq==nv, got nq={nq} nv={nv}"
    eps = 1e-6

    max_fd_err = 0.0
    for q, qd, u, name in _samples(pm):
        _x_kp1, dAB = _combined_integrator_with_gradient(
            pm, q, qd, u, _DT, integrator_type
        )
        inputs = np.concatenate([q, qd, u])
        fd = np.zeros((2 * nv, 3 * nv))
        for j in range(3 * nv):
            ip = inputs.copy(); ip[j] += eps
            im = inputs.copy(); im[j] -= eps
            xp = pm.integrator(ip[:nv], ip[nv:2 * nv], ip[2 * nv:], _DT,
                               integrator_type=integrator_type)
            xm = pm.integrator(im[:nv], im[nv:2 * nv], im[2 * nv:], _DT,
                               integrator_type=integrator_type)
            fd[:, j] = (np.asarray(xp) - np.asarray(xm)) / (2 * eps)
        np.testing.assert_allclose(
            dAB, fd, rtol=1e-4, atol=1e-5,
            err_msg=(
                f"{robot_id}-{base_mode} {integrator_type} combined gradient vs "
                f"FD of combined x_kp1 @ {name}"
            ),
        )
        max_fd_err = max(max_fd_err, float(np.max(np.abs(dAB - fd))))

    print(
        f"[combined-fd] {robot_id}-{base_mode} {integrator_type}: "
        f"max_abs_grad_vs_fd={max_fd_err:.3e}"
    )
