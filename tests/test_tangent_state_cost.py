"""Oracle gates for `dDifference` + `quadratic_state_cost_tangent` (GATO ASK 3).

Layers, each independent:

1. **Pinocchio anchor** for `dDifference` (ARG0/ARG1) on go2-floating and
   iiwa14-fixed — the external ground truth for the exact J_diff.
2. **FD-through-integrate gate** for `dDifference` using only our own code:
   perturb the 'to' / 'from' argument along tangent basis vectors via
   `integrate` and central-difference `difference`.
3. **Value-FD gates** for the tangent cost: gradient vs central differences of
   the VALUE under tangent perturbations, and the full-Newton hessian vs
   second-order central differences of the VALUE — all differences taken in
   the SAME chart (delta at the evaluation point), which is exactly the
   quantity the oracle defines. Gauss-Newton is checked structurally
   (== J^T diag(Q) J) and == Newton at zero error.
4. **Fixed-base reduction**: equals `quadratic_state_cost` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


def _case(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in the robot manifest")


def _project(robot_id, base_mode):
    spec = _case(robot_id, base_mode)
    resolved = resolve_robot_spec(spec)
    return build_project_adapter(spec, resolved, base_mode=base_mode)


def _rand_q(rng, nq, floating):
    q = rng.uniform(-1.0, 1.0, nq)
    if floating:
        q[3:7] /= np.linalg.norm(q[3:7])
    return q


@pytest.mark.pinocchio_equivalence
@pytest.mark.parametrize("robot_id,base_mode", [("go2", "floating"), ("iiwa14", "fixed")])
def test_ddifference_pinocchio(robot_id, base_mode):
    spec = _case(robot_id, base_mode)
    resolved = resolve_robot_spec(spec)
    project = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref, robot = project.reference, project.robot
    nq = robot.get_num_pos()
    floating = base_mode == "floating"

    from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
    pin_adapter = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
    import pinocchio as pin
    model = pin_adapter.model

    rng = np.random.default_rng(7)
    for _ in range(15):
        q1, q2 = _rand_q(rng, nq, floating), _rand_q(rng, nq, floating)
        assert np.max(np.abs(
            ref.dDifference(q1, q2, "to") - pin.dDifference(model, q1, q2, pin.ARG1))) < 1e-9
        assert np.max(np.abs(
            ref.dDifference(q1, q2, "from") - pin.dDifference(model, q1, q2, pin.ARG0))) < 1e-9


def test_ddifference_fd_through_integrate():
    project = _project("go2", "floating")
    ref, robot = project.reference, project.robot
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(11)
    h = 1e-6
    for _ in range(5):
        q1, q2 = _rand_q(rng, nq, True), _rand_q(rng, nq, True)
        J_to = ref.dDifference(q1, q2, "to")
        J_from = ref.dDifference(q1, q2, "from")
        for i in range(nv):
            e_i = np.zeros(nv)
            e_i[i] = h
            fd_to = (ref.difference(q1, ref.integrate(q2, e_i))
                     - ref.difference(q1, ref.integrate(q2, -e_i))) / (2 * h)
            fd_from = (ref.difference(ref.integrate(q1, e_i), q2)
                       - ref.difference(ref.integrate(q1, -e_i), q2)) / (2 * h)
            assert np.max(np.abs(J_to[:, i] - fd_to)) < 1e-7
            assert np.max(np.abs(J_from[:, i] - fd_from)) < 1e-7


def _perturb_x(ref, q, qd, delta, nv):
    return ref.integrate(q, delta[:nv]), qd + delta[nv:]


def _value_at(ref, q, qd, x_des, Q, delta, nv):
    qp, qdp = _perturb_x(ref, q, qd, delta, nv)
    value, _, _ = ref.quadratic_state_cost_tangent(
        np.concatenate([qp, qdp]), x_des, Q)
    return value


def test_tangent_cost_gradient_and_newton_hessian_fd():
    project = _project("go2", "floating")
    ref, robot = project.reference, project.robot
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(23)
    q, qd = _rand_q(rng, nq, True), rng.uniform(-1.0, 1.0, nv)
    q_des, qd_des = _rand_q(rng, nq, True), rng.uniform(-1.0, 1.0, nv)
    x = np.concatenate([q, qd])
    x_des = np.concatenate([q_des, qd_des])
    Q = rng.uniform(0.1, 2.0, 2 * nv)

    value, grad, hess_gn = ref.quadratic_state_cost_tangent(x, x_des, Q)
    _, _, hess_nt = ref.quadratic_state_cost_tangent(x, x_des, Q, gauss_newton=False)
    n = 2 * nv

    # gradient vs central value-FD in the tangent chart
    h = 1e-6
    for i in range(n):
        d = np.zeros(n)
        d[i] = h
        fd = (_value_at(ref, q, qd, x_des, Q, d, nv)
              - _value_at(ref, q, qd, x_des, Q, -d, nv)) / (2 * h)
        assert abs(grad[i] - fd) < 5e-7 * max(1.0, abs(fd))

    # GN structure: J^T diag(Q) J with the exact J_diff
    Jq = ref.dDifference(q_des, q, "to")
    gn_q = Jq.T @ (Q[:nv, None] * Jq)
    assert np.max(np.abs(hess_gn[:nv, :nv] - gn_q)) < 1e-12
    assert np.max(np.abs(hess_gn[nv:, nv:] - np.diag(Q[nv:]))) < 1e-12
    assert np.max(np.abs(hess_gn[:nv, nv:])) == 0.0

    # full-Newton hessian vs second-order central value-FD (same chart)
    h2 = 1e-4
    fd_hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            di, dj = np.zeros(n), np.zeros(n)
            di[i] = h2
            dj[j] = h2
            fd_hess[i, j] = (
                _value_at(ref, q, qd, x_des, Q, di + dj, nv)
                - _value_at(ref, q, qd, x_des, Q, di - dj, nv)
                - _value_at(ref, q, qd, x_des, Q, -di + dj, nv)
                + _value_at(ref, q, qd, x_des, Q, -di - dj, nv)) / (4 * h2 * h2)
    assert np.max(np.abs(hess_nt - fd_hess)) < 5e-5
    # the exact chart hessian is symmetric
    assert np.max(np.abs(hess_nt - hess_nt.T)) < 1e-10

    # at zero error GN == Newton
    v0, g0, hgn0 = ref.quadratic_state_cost_tangent(x, x, Q)
    _, _, hnt0 = ref.quadratic_state_cost_tangent(x, x, Q, gauss_newton=False)
    assert v0 == 0.0 and np.max(np.abs(g0)) < 1e-12
    assert np.max(np.abs(hgn0 - hnt0)) < 1e-12


def test_fixed_base_reduction():
    project = _project("iiwa14", "fixed")
    ref, robot = project.reference, project.robot
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    assert nq == nv
    rng = np.random.default_rng(31)
    x = rng.uniform(-1.0, 1.0, nq + nv)
    x_des = rng.uniform(-1.0, 1.0, nq + nv)
    Q = rng.uniform(0.1, 2.0, 2 * nv)
    v_t, g_t, h_t = ref.quadratic_state_cost_tangent(x, x_des, Q)
    v_r, g_r, h_r = ref.quadratic_state_cost(x, x_des, Q)
    assert abs(v_t - v_r) < 1e-14
    assert np.max(np.abs(g_t - g_r)) < 1e-14
    assert np.max(np.abs(h_t - h_r)) < 1e-14
    # Newton == GN on a vector space
    _, _, h_nt = ref.quadratic_state_cost_tangent(x, x_des, Q, gauss_newton=False)
    assert np.max(np.abs(h_nt - h_t)) < 1e-14
