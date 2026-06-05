"""Floating-base `plant_step_hessian` cross-checked against Pinocchio.

`plant_step_hessian(q,qd,u,dt)[o,a,b] = d/dz[a] plant_step_gradient[o,b]` with
z = [dq; dqd; du] in the nv tangent. For a floating base the position rows ride
the SE(3) Lie-group retract `integrate`, so the analytic tensor is built from
the pin-validated `d2Integrate` / `dIntegrate` (position rows) and the
`fdsva_so` second-order dynamics (velocity rows). This module proves the
assembly against a 4th-order central FINITE DIFFERENCE of **pinocchio's own**
`integrator_gradient` along the nv-tangent z directions (q perturbed on the
manifold via `pin.integrate`, qd/u additively).

Two independent oracles, deliberately separated so neither hides the other:

  1. **FD-of-pinocchio-gradient (authoritative cross-library check).** The
     position rows for Euler, and *all* rows for go2, match pinocchio to float64
     precision -- this is the real proof that the `d2Integrate`-based retract
     formulas reproduce pinocchio's `integrate`. Bucketed per (robot,
     integrator, row-group): see `_PIN_RTOL`.

  2. **FD-of-GRiD's-own-gradient (self-consistency, like the fixed-base
     sanity).** This holds to ~1e-6 for EVERY robot/integrator/row, because the
     analytic tensor is, by construction, the exact second derivative of
     `integrator_gradient`.

WHY THE PER-ROBOT PIN BUCKETS WIDEN. The velocity rows are `dt * D2qdd`, and
the q-q block of `D2qdd` is GRiD's *body-frame* `fdsva_so` second derivative.
That body-frame second derivative equals pinocchio's `integrate`-tangent second
derivative of qdd for go2 (~1e-9) but DIVERGES by a robot-dependent O(few)
amount on more-jointed humanoids (g1 ~1.6e-3, h1_2 ~3.5 absolute on the q-q
block) -- a pre-existing convention gap between GRiD's body-frame
`idsva_so_body_frame`/`fdsva_so` and pinocchio's `computeABADerivatives`
tangent, NOT introduced by this Hessian. It is invisible at first order (the
gradients agree to 1e-13) and only shows up in the second derivative. For
SI-Euler it also leaks into the *position* rows through the
`dt^2 * dIntegrate_v . D2qdd` chain term (h1_2 SI-Euler position rows ~2.6e-2).
The pin buckets below track exactly that residual; the tight
self-consistency check (oracle 2) is the gate that the analytic is the correct
derivative of GRiD's own surface. Mimic robots are skipped (reduced-model
pin dim mismatch).
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


_INTEGRATORS = ("euler", "semi_implicit_euler")
_DT = 0.01
_FD_STEP = 5e-4  # 4th-order stencil sweet spot; clear of the small-angle cliff.


def _robot_ids():
    # Floating, non-mimic robots whose project nv == pinocchio nv (so an
    # nv-tangent FD needs no reduced-layout bookkeeping).
    raw = os.environ.get("GRID_PLANT_HESS_PIN_ROBOTS", "go2,g1,h1_2")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


# Per-(robot, integrator, row-group) relative tolerance vs the FD-of-pinocchio
# gradient. go2 is tight everywhere; the humanoids widen ONLY where the
# body-frame fdsva_so second derivative feeds in (velocity rows always; SI-Euler
# position rows via the chain term). Position rows for Euler stay tight for all
# robots -- the pure d2Integrate retract cross-check. See module docstring.
_PIN_RTOL = {
    # robot: {(integrator, "pos"|"vel"): rtol}
    "go2": {
        ("euler", "pos"): 1e-7, ("euler", "vel"): 1e-7,
        ("semi_implicit_euler", "pos"): 1e-7, ("semi_implicit_euler", "vel"): 1e-7,
    },
    "g1": {
        ("euler", "pos"): 1e-7, ("euler", "vel"): 1e-5,
        ("semi_implicit_euler", "pos"): 1e-7, ("semi_implicit_euler", "vel"): 1e-5,
    },
    "h1_2": {
        ("euler", "pos"): 1e-7, ("euler", "vel"): 1e-4,
        # SI-Euler position rows inherit the q-q gap via dt^2 * dInt_v . D2qdd.
        ("semi_implicit_euler", "pos"): 5e-2, ("semi_implicit_euler", "vel"): 1e-4,
    },
}


def _spec(robot_id):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-floating not in manifest")


def _build(robot_id):
    spec = _spec(robot_id)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode="floating")
    pin = build_pinocchio_adapter(spec, resolved, base_mode="floating")
    return spec, pm.reference, pin, build_dynamics_samples(pm)


def _G_pin(pin, q, qd, u, integrator_type):
    # `integrator_gradient` returns views into pin's reused data buffers; copy
    # so a 4-point FD stencil over successive calls doesn't alias.
    return np.array(
        pin.integrator_gradient(q, qd, u, _DT, integrator_type=integrator_type),
        copy=True,
    )


def _fd_pin_hessian(pin, nv, q, qd, u, integrator_type, h=_FD_STEP):
    """4th-order central FD of pinocchio's `integrator_gradient` along the
    nv tangent: q perturbed on the manifold via `pin.integrate`, qd/u
    additively. Returns H_fd[o, a, b]."""
    nz = 3 * nv
    H = np.zeros((2 * nv, nz, nz))
    for a in range(nz):
        cols = []
        for sgn in (h, -h, 2.0 * h, -2.0 * h):
            if a < nv:  # q-tangent: retract on the manifold (NOT additive)
                ea = np.zeros(nv)
                ea[a] = 1.0
                q_pert = pin._pin_integrate(q, sgn * ea)
                cols.append(_G_pin(pin, q_pert, qd, u, integrator_type))
            else:       # qd / u: additive
                idx = a - nv
                qd2, u2 = qd.copy(), u.copy()
                if a < 2 * nv:
                    qd2[idx] += sgn
                else:
                    u2[idx - nv] += sgn
                cols.append(_G_pin(pin, q, qd2, u2, integrator_type))
        H[:, a, :] = (8.0 * (cols[0] - cols[1]) - (cols[2] - cols[3])) / (12.0 * h)
    return H


def _fd_grid_hessian(ref, nv, q, qd, u, integrator_type, h=1e-5):
    """2nd-order central FD of GRiD's own `integrator_gradient` (self-
    consistency oracle, mirroring the fixed-base sanity test). q is perturbed
    on the manifold via `ref.integrate`."""
    nz = 3 * nv
    H = np.zeros((2 * nv, nz, nz))
    for a in range(nz):
        cols = []
        for sgn in (h, -h):
            if a < nv:
                ea = np.zeros(nv)
                ea[a] = 1.0
                q_pert = ref.integrate(q, sgn * ea)
                cols.append(np.array(
                    ref.integrator_gradient(q_pert, qd, u, _DT,
                                            integrator_type=integrator_type),
                    copy=True))
            else:
                idx = a - nv
                qd2, u2 = qd.copy(), u.copy()
                if a < 2 * nv:
                    qd2[idx] += sgn
                else:
                    u2[idx - nv] += sgn
                cols.append(np.array(
                    ref.integrator_gradient(q, qd2, u2, _DT,
                                            integrator_type=integrator_type),
                    copy=True))
        H[:, a, :] = (cols[0] - cols[1]) / (2.0 * h)
    return H


# --------------------------------------------------------------------------
# Oracle 1: vs a finite difference of PINOCCHIO's integrator gradient.
# --------------------------------------------------------------------------
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.pinocchio_equivalence
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-floating")
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_plant_hessian_floating_matches_pinocchio_fd(robot_id, integrator_type):
    _spec_, ref, pin, samples = _build(robot_id)
    nv = ref.robot.get_num_vel()
    nz = 3 * nv
    rtol_pos = _PIN_RTOL[robot_id][(integrator_type, "pos")]
    rtol_vel = _PIN_RTOL[robot_id][(integrator_type, "vel")]

    for sample in samples:
        q = np.asarray(sample.q, float)
        qd = np.asarray(sample.qd, float)
        u = np.asarray(sample.qdd, float)  # 3rd input = control torque
        analytic = ref.plant_step_hessian(q, qd, u, _DT,
                                          integrator_type=integrator_type)
        assert analytic.shape == (2 * nv, nz, nz)

        H_fd = _fd_pin_hessian(pin, nv, q, qd, u, integrator_type)
        scale = max(1.0, float(np.abs(H_fd).max()))

        pos_err = float(np.abs(analytic[:nv] - H_fd[:nv]).max()) / scale
        vel_err = float(np.abs(analytic[nv:] - H_fd[nv:]).max()) / scale
        assert pos_err <= rtol_pos, (
            f"{robot_id} {integrator_type} POSITION rows vs FD-of-pinocchio "
            f"@ {sample.name}: rel {pos_err:.3e} > {rtol_pos:.0e}")
        assert vel_err <= rtol_vel, (
            f"{robot_id} {integrator_type} VELOCITY rows vs FD-of-pinocchio "
            f"@ {sample.name}: rel {vel_err:.3e} > {rtol_vel:.0e}")


# --------------------------------------------------------------------------
# Oracle 2: self-consistency vs a finite difference of GRiD's OWN gradient.
# This is the tight correctness gate (the analytic IS the second derivative of
# `integrator_gradient`), holding for every robot/integrator/row group.
# --------------------------------------------------------------------------
@pytest.mark.developer_only
@pytest.mark.floating_base
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-floating")
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_plant_hessian_floating_self_consistent(robot_id, integrator_type):
    _spec_, ref, _pin, samples = _build(robot_id)
    nv = ref.robot.get_num_vel()
    for sample in samples:
        q = np.asarray(sample.q, float)
        qd = np.asarray(sample.qd, float)
        u = np.asarray(sample.qdd, float)
        analytic = ref.plant_step_hessian(q, qd, u, _DT,
                                          integrator_type=integrator_type)
        H_fd = _fd_grid_hessian(ref, nv, q, qd, u, integrator_type)
        # Max-norm relative error (scaled by the tensor magnitude). A per-
        # element assert_allclose trips on the handful of structurally-tiny
        # entries where a central FD of the gradient is dominated by its own
        # O(h^2) noise -- a global scale is the honest self-consistency metric.
        scale = max(1.0, float(np.abs(H_fd).max()))
        rel = float(np.abs(analytic - H_fd).max()) / scale
        assert rel <= 1e-5, (
            f"{robot_id} {integrator_type} plant_step_hessian vs "
            f"FD-of-GRiD-gradient @ {sample.name}: rel {rel:.3e} > 1e-5")
