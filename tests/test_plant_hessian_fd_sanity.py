"""Finite-difference sanity check for the analytical `plant_step_hessian`
(fixed-base Euler + semi-implicit Euler).

`plant_step_hessian(q,qd,u,dt)[o,a,b] = d^2 x_{k+1}[o] / dz[a] dz[b]` is the
derivative of `plant_step_gradient(q,qd,u,dt)[o,b]` w.r.t. z[a], with
z = [q; qd; u]. We central-difference the (already pinocchio-validated)
gradient along each of the 3*nv input directions and compare. This composes
only trusted pieces (the gradient) and is the primary oracle cross-check from
docs/open-tasks/f1_plant_step_hessian_plan.md (cross-check #1).

Fixed-base, pure-revolute robots only (nq == nv), so an additive input
perturbation IS the tangent perturbation. Floating-base / RK are deferred and
asserted to raise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.equivalents.reference_backend import build_project_adapter


_INTEGRATORS = ("euler", "semi_implicit_euler")


def _robot_ids():
    raw = os.environ.get("GRID_PLANT_HESS_FD_ROBOTS", "iiwa14,go2")
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
    return spec, pm.reference, build_dynamics_samples(pm)


@pytest.mark.developer_only
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-fixed")
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_plant_hessian_matches_finite_difference(robot_id, integrator_type):
    _spec_, ref, samples = _build(robot_id, "fixed")
    nv = ref.robot.get_num_vel()
    nz = 3 * nv
    dt = 0.01
    eps = 1e-6

    for sample in samples:
        q, qd, u = sample.q, sample.qd, sample.qdd  # 3rd input = control torque
        analytic = ref.plant_step_hessian(q, qd, u, dt, integrator_type=integrator_type)
        assert analytic.shape == (2 * nv, nz, nz)

        z = np.concatenate([np.asarray(q, float), np.asarray(qd, float), np.asarray(u, float)])
        H_fd = np.zeros((2 * nv, nz, nz))
        for a in range(nz):
            zp = z.copy(); zp[a] += eps
            zm = z.copy(); zm[a] -= eps
            gp = ref.plant_step_gradient(zp[:nv], zp[nv:2 * nv], zp[2 * nv:], dt,
                                         integrator_type=integrator_type)
            gm = ref.plant_step_gradient(zm[:nv], zm[nv:2 * nv], zm[2 * nv:], dt,
                                         integrator_type=integrator_type)
            H_fd[:, a, :] = (np.asarray(gp) - np.asarray(gm)) / (2 * eps)

        np.testing.assert_allclose(
            analytic, H_fd, rtol=1e-4, atol=1e-5,
            err_msg=f"{robot_id} {integrator_type} plant_step_hessian vs FD-of-gradient @ {sample.name}")


@pytest.mark.developer_only
def test_plant_hessian_defers_rk_and_floating():
    # RK is deferred (multi-stage 2nd-order chain rule).
    _spec_, ref, samples = _build("iiwa14", "fixed")
    s = samples[0]
    with pytest.raises(NotImplementedError):
        ref.plant_step_hessian(s.q, s.qd, s.qdd, 0.01, integrator_type="rk4")

    # Floating-base is deferred (SE(3) retract connection term).
    _spec2, ref_fl, samples_fl = _build("go2", "floating")
    sf = samples_fl[0]
    with pytest.raises(NotImplementedError):
        ref_fl.plant_step_hessian(sf.q, sf.qd, sf.qdd, 0.01, integrator_type="euler")
