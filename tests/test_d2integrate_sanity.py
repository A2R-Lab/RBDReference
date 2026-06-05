"""Sanity checks for the second-order Lie-group retract `d2Integrate`.

`RBDReference.d2Integrate(q, v_dt, arg1, arg2)` returns the tangent-space
derivative of the `dIntegrate` Jacobian:

    H[i, j, k] = d/dxi_k ( dIntegrate(q, v_dt, arg1)[i, j] )

with `xi_k` a unit tangent perturbation in the `arg2` coordinate. It is the
building block for the floating-base position rows of a second-order
integrator (plant_step_hessian); see
docs/open-tasks/f1_plant_step_hessian_plan.md.

Two independent checks (the FD implementation alone would only be
self-consistent, so the closed-form anchor is what makes this meaningful):

1. **Closed-form anchor at v_dt = 0.** The leading-order expansions of the
   SE(3) right-Jacobian and adjoint give exact, hand-derived values for the
   free-flyer 6x6 block of every nonzero tensor. This pins the *structure*
   (which blocks are nonzero, signs, the -1/2 vs -1 factor between the
   J_r-side and the Ad-side) independently of pinocchio and of our own FD.
2. **Pinocchio FD cross-check at random v_dt.** Compares against a 4th-order
   central difference of `pin.dIntegrate` (an independent FD path from the
   one inside `RBDReference.d2Integrate`, which differences our own
   `dIntegrate`). The derivative is far more sensitive to a wrong
   v-dependence in the SE(3) Q-block than the Jacobian value itself.

Fixed-base d2Integrate is identically zero (affine retract) and is asserted
as such.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _robot_ids():
    # Floating, non-mimic robots: their project nv matches pinocchio nv so the
    # FD cross-check needs no reduced-layout bookkeeping.
    raw = os.environ.get("GRID_D2INT_ROBOTS", "go2,g1,h1_2")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _build(robot_id, base_mode):
    spec = _spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode=base_mode)
    pin = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
    q = build_dynamics_samples(pm)[0].q  # any valid config (result is q-independent)
    return spec, pm.reference, pin, q


# --------------------------------------------------------------------------
# 1. Closed-form anchor at v_dt = 0 (free-flyer 6x6 block).
# --------------------------------------------------------------------------
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-anchor")
def test_d2integrate_zero_increment_closed_form(robot_id):
    _spec_, ref, _pin, q = _build(robot_id, "floating")
    nv = ref.robot.get_num_vel()
    v0 = np.zeros(nv)
    e = [np.eye(3)[i] for i in range(3)]

    # H(arg1='v', arg2='v'): d/dv of the SE(3) right-Jacobian at 0.
    #   rho-direction k:   only top-right block [0:3,3:6] = -1/2 [e_k]_x
    #   phi-direction m:   diag blocks [0:3,0:3]=[3:6,3:6] = -1/2 [e_m]_x
    Hvv = ref.d2Integrate(q, v0, "v", "v")
    for k in range(3):
        exp = np.zeros((nv, nv))
        exp[0:3, 3:6] = -0.5 * _skew(e[k])
        np.testing.assert_allclose(
            Hvv[:, :, k], exp, atol=1e-6,
            err_msg=f"{robot_id} d2Int(v,v) rho-dir {k} @ v=0")
    for m in range(3):
        exp = np.zeros((nv, nv))
        exp[0:3, 0:3] = -0.5 * _skew(e[m])
        exp[3:6, 3:6] = -0.5 * _skew(e[m])
        np.testing.assert_allclose(
            Hvv[:, :, 3 + m], exp, atol=1e-6,
            err_msg=f"{robot_id} d2Int(v,v) phi-dir {m} @ v=0")

    # H(arg1='q', arg2='v'): d/dv of Ad(exp(-v)) at 0 (same structure, factor -1).
    #   rho-direction k:   top-right block [0:3,3:6] = -[e_k]_x
    #   phi-direction m:   diag blocks [0:3,0:3]=[3:6,3:6] = -[e_m]_x
    Hqv = ref.d2Integrate(q, v0, "q", "v")
    for k in range(3):
        exp = np.zeros((nv, nv))
        exp[0:3, 3:6] = -_skew(e[k])
        np.testing.assert_allclose(
            Hqv[:, :, k], exp, atol=1e-6,
            err_msg=f"{robot_id} d2Int(q,v) rho-dir {k} @ v=0")
    for m in range(3):
        exp = np.zeros((nv, nv))
        exp[0:3, 0:3] = -_skew(e[m])
        exp[3:6, 3:6] = -_skew(e[m])
        np.testing.assert_allclose(
            Hqv[:, :, 3 + m], exp, atol=1e-6,
            err_msg=f"{robot_id} d2Int(q,v) phi-dir {m} @ v=0")

    # arg2 == 'q' is exactly zero (free-flyer dIntegrate is q-independent).
    for arg1 in ("q", "v"):
        Hq = ref.d2Integrate(q, v0, arg1, "q")
        np.testing.assert_array_equal(Hq, np.zeros((nv, nv, nv)))


# --------------------------------------------------------------------------
# 2. Pinocchio FD cross-check at random v_dt.
# --------------------------------------------------------------------------
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.pinocchio_equivalence
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-vs-pin")
@pytest.mark.parametrize("arg1", ("q", "v"))
def test_d2integrate_matches_pinocchio_fd(robot_id, arg1):
    spec, ref, pin, q = _build(robot_id, "floating")
    nv = ref.robot.get_num_vel()
    rng = np.random.default_rng(11)
    for trial in range(3):
        # Keep the increment moderate: SE(3) closed forms are smooth but the
        # Q-block conditioning degrades near |phi| ~ pi.
        v_dt = rng.uniform(-0.4, 0.4, size=nv)
        ours = ref.d2Integrate(q, v_dt, arg1, "v")
        theirs = pin.d2Integrate(q, v_dt, arg1, "v")
        np.testing.assert_allclose(
            ours, theirs, rtol=1e-5, atol=1e-7,
            err_msg=f"{robot_id} d2Int({arg1},v) vs pin FD, trial {trial}")


# --------------------------------------------------------------------------
# 3. Fixed-base is identically zero.
# --------------------------------------------------------------------------
@pytest.mark.developer_only
@pytest.mark.parametrize("robot_id", ("iiwa14", "go2"), ids=lambda r: f"{r}-fixed-zero")
def test_d2integrate_fixed_base_is_zero(robot_id):
    _spec_, ref, _pin, q = _build(robot_id, "fixed")
    nv = ref.robot.get_num_vel()
    v_dt = np.linspace(-0.3, 0.3, nv)
    for arg1 in ("q", "v"):
        for arg2 in ("q", "v"):
            H = ref.d2Integrate(q, v_dt, arg1, arg2)
            np.testing.assert_array_equal(H, np.zeros((nv, nv, nv)))
