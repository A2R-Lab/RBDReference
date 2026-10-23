"""Cost menu reference checks: activations, activated terms, linear feature, named frames.

GRiD-defined quantities (the CUDA grid_plant surface), so there is no Pinocchio
oracle; each is checked against a closed form, a central-difference gradient and a
Gauss-Newton Hessian recompute on iiwa14 (fixed) and go2 (floating):
  - the quadratic activation reproduces quadratic_state_cost / quadratic_input_cost /
    ee_pos_cost exactly,
  - the pseudo-Huber activation has the stated value, slope and curvature and the
    weighted-absolute-value and quadratic limits,
  - state_cost / input_cost / linear_feature_cost / frame_position_cost gradients
    match central differences and the Hessians are J^T diag(a'') J.
"""
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference

_ASSETS = Path(__file__).resolve().parents[1] / "robot_assets"
_H = 1e-6


def _ref(name, floating):
    return RBDReference(URDFParser().parse(str(_ASSETS / f"{name}.urdf"), floating_base=floating))


def _central(f, x, h=_H):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for k in range(x.size):
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        g[k] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def test_activation_closed_forms_and_limits():
    W = np.array([2.0, 3.0, 0.5])
    r = np.array([-0.4, 0.0, 0.7])
    v, g, c = RBDReference._activation(0, W, r, 0.0)
    assert np.allclose(v, 0.5 * W * r * r) and np.allclose(g, W * r) and np.allclose(c, W)
    eps = 0.05
    s = np.sqrt(r * r + eps * eps)
    v, g, c = RBDReference._activation(1, W, r, eps)
    assert np.allclose(v, W * (s - eps)) and np.allclose(g, W * r / s) and np.allclose(c, W * eps ** 2 / s ** 3)
    # the slope is the derivative of the value and the curvature the derivative of the slope
    for k in range(3):
        fv = lambda rr: RBDReference._activation(1, W[k:k + 1], rr, eps)[0][0]
        fg = lambda rr: RBDReference._activation(1, W[k:k + 1], rr, eps)[1][0]
        assert abs(_central(fv, r[k:k + 1])[0] - g[k]) < 1e-6
        assert abs(_central(fg, r[k:k + 1])[0] - c[k]) < 1e-5
    # limits: tiny eps is the weighted absolute value; |r| << eps is the quadratic w r^2 / (2 eps)
    v_small, _, _ = RBDReference._activation(1, W, r, 1e-9)
    assert np.allclose(v_small, W * np.abs(r), atol=1e-8)
    v_big, _, c_big = RBDReference._activation(1, W, 1e-4 * r, 1.0)
    assert np.allclose(v_big, 0.5 * W * (1e-4 * r) ** 2, rtol=1e-6) and np.allclose(c_big, W, rtol=1e-6)
    with pytest.raises(ValueError):
        RBDReference._activation(1, W, r, 0.0)
    with pytest.raises(ValueError):
        RBDReference._activation(7, W, r, 0.1)


@pytest.mark.parametrize("name,floating", [("iiwa14", False), ("go2", True)], ids=["iiwa14", "go2"])
def test_activated_terms_against_finite_differences(name, floating):
    ref = _ref(name, floating)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    nx = nq + nv
    rng = np.random.default_rng(5)
    q = rng.standard_normal(nq)
    qd = rng.standard_normal(nv)
    u = rng.standard_normal(nv)
    x = np.concatenate([q, qd])
    x_des = 0.1 * np.arange(nx); Q = 1.0 + 0.5 * np.arange(nx)
    u_des = -0.05 * np.arange(nv); R = 2.0 + 0.1 * np.arange(nv)
    eps = 0.03
    # quadratic activation reproduces the quadratic terms exactly
    for a, b in zip(ref.state_cost(x, x_des, Q, 0, 0.0), ref.quadratic_state_cost(x, x_des, Q)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(ref.input_cost(u, u_des, R, 0, 0.0), ref.quadratic_input_cost(u, u_des, R)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # pseudo-Huber state and input: gradient vs FD of the value, Hessian diagonal of a''
    val, grad, hess = ref.state_cost(x, x_des, Q, 1, eps)
    assert np.allclose(grad, _central(lambda xx: ref.state_cost(xx, x_des, Q, 1, eps)[0], x), atol=1e-6)
    assert np.allclose(np.diag(hess), ref._activation(1, Q, x - x_des, eps)[2]) and np.count_nonzero(hess - np.diag(np.diag(hess))) == 0
    val, grad, hess = ref.input_cost(u, u_des, R, 1, eps)
    assert np.allclose(grad, _central(lambda uu: ref.input_cost(uu, u_des, R, 1, eps)[0], u), atol=1e-6)
    # linear feature on x (2 rows) and on u (1 row)
    A = rng.standard_normal((2, nx)); b = np.array([0.0, 0.3]); t = np.array([0.2, 0.3]); W = np.array([4.0, 5.0])
    val, grad, hess = ref.linear_feature_cost(x, A, b, t, W, 1, eps)
    assert np.allclose(grad, _central(lambda xx: ref.linear_feature_cost(xx, A, b, t, W, 1, eps)[0], x), atol=1e-6)
    curv = ref._activation(1, W, A @ x + b - t, eps)[2]
    assert np.allclose(hess, A.T @ (curv[:, None] * A))
    Au = rng.standard_normal((1, nv))
    val, grad, hess = ref.linear_feature_cost(u, Au, [0.1], [-0.1], [5.0], 1, eps)
    assert np.allclose(grad, _central(lambda uu: ref.linear_feature_cost(uu, Au, [0.1], [-0.1], [5.0], 1, eps)[0], u), atol=1e-6)
    # the quadratic activation on a linear feature is 1/2 (A x + b - t)^T diag(W) (A x + b - t)
    rr = A @ x + b - t
    assert np.isclose(ref.linear_feature_cost(x, A, b, t, W, 0, 0.0)[0], 0.5 * np.sum(W * rr * rr))


@pytest.mark.parametrize("name,floating", [("iiwa14", False), ("go2", True)], ids=["iiwa14", "go2"])
def test_frame_zero_is_the_end_effector_cost(name, floating):
    """frame_position_cost on the end-effector joint under the quadratic activation is
    ee_pos_cost; under pseudo-Huber it carries the Gauss-Newton structure from the
    position Jacobian and zero qd blocks (the floating go2 included)."""
    ref = _ref(name, floating)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    rng = np.random.default_rng(9)
    q = rng.standard_normal(nq)
    frame = ref.robot.get_joint_by_id(ref.robot.get_leaf_nodes()[0]).get_name()
    W = np.array([10.0, 11.0, 12.0]); p_des = np.array([0.1, -0.2, 0.3]); eps = 0.02
    # the frame term is tangent-sized (2 nv), ee_pos_cost is x-sized (nq + nv): equal on the q block,
    # zero elsewhere on both
    fv, fg, fh = ref.frame_position_cost(q, frame, p_des, W, 0, 0.0)
    ev, eg, eh = ref.ee_pos_cost(q, p_des, W)
    assert np.isclose(fv, ev, atol=1e-12) and np.allclose(fg[:nv], eg[:nv], atol=1e-12) and np.allclose(fh[:nv, :nv], eh[:nv, :nv], atol=1e-12)
    assert fg.shape == (2 * nv,) and np.count_nonzero(fg[nv:]) == 0 and np.count_nonzero(eg[nv:]) == 0
    val, grad_x, hess_x = ref.frame_position_cost(q, frame, p_des, W, 1, eps)
    assert np.count_nonzero(grad_x[nv:]) == 0 and np.count_nonzero(hess_x[nv:, :]) == 0
    Jp = np.asarray(ref.frame_jacobian(q, frame))[:3, :]
    _, p, _ = ref._frame_placement(q, frame)
    _, slope, curv = ref._activation(1, W, p - p_des, eps)
    assert np.allclose(grad_x[:nv], Jp.T @ slope) and np.allclose(hess_x[:nv, :nv], Jp.T @ (curv[:, None] * Jp))


# ---- named-frame residuals and barrier activations ----

def _frames(ref):
    robot = ref.robot
    return [robot.get_joint_by_id(robot.get_leaf_nodes()[0]).get_name(), robot.get_joint_by_id(3).get_name()]


@pytest.mark.parametrize("name,floating", [("iiwa14", False), ("fr3", False)], ids=["iiwa14", "fr3"])
def test_frame_position_and_orientation_costs(name, floating):
    ref = _ref(name, floating)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    rng = np.random.default_rng(11)
    q = rng.standard_normal(nq)
    W = np.array([2.0, 3.0, 4.0]); eps = 0.05
    for frame in _frames(ref):
        p_des = np.array([0.1, 0.2, 0.3])
        val, grad, hess = ref.frame_position_cost(q, frame, p_des, W, 1, eps)
        fd = _central(lambda qq: ref.frame_position_cost(qq, frame, p_des, W, 1, eps)[0], q)
        assert np.allclose(grad[:nv], fd, atol=1e-6) and np.count_nonzero(grad[nv:]) == 0
        # orientation: a target rotated from the current frame by phi recovers phi as the residual
        R_f, _, _ = ref._frame_placement(q, frame)
        phi = np.array([0.3, -0.2, 0.5])
        R_des = R_f @ ref._so3_right_jacobian_inv(np.zeros(3))  # identity start
        from scipy.spatial.transform import Rotation
        R_des = R_f @ Rotation.from_rotvec(-phi).as_matrix()
        assert np.allclose(ref._so3_log(R_des.T @ R_f), phi, atol=1e-12)
        val, grad, hess = ref.frame_orientation_cost(q, frame, R_des.flatten(order="F"), W, 1, eps)
        fd = _central(lambda qq: ref.frame_orientation_cost(qq, frame, R_des.flatten(order="F"), W, 1, eps)[0], q)
        assert np.allclose(grad[:nv], fd, atol=1e-6), np.max(np.abs(grad[:nv] - fd))
        assert np.count_nonzero(grad[nv:]) == 0 and np.count_nonzero(hess[nv:, :]) == 0
        # the Gauss-Newton structure J_e^T diag(a'') J_e
        Jw = np.asarray(ref.frame_jacobian(q, frame, "LOCAL"))[3:, :]
        Je = ref._so3_right_jacobian_inv(phi) @ Jw
        curv = ref._activation(1, W, phi, eps)[2]
        assert np.allclose(hess[:nv, :nv], Je.T @ (curv[:, None] * Je))
    # the log map at pi/2 pitch (the Euler singularity) is regular
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    R = Rz(0.3) @ Ry(np.pi / 2)
    phi = ref._so3_log(R)
    from scipy.spatial.transform import Rotation
    assert np.allclose(Rotation.from_rotvec(phi).as_matrix(), R, atol=1e-10)
    # near pi branch
    R_pi = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi - 1e-7])).as_matrix()
    assert np.isclose(np.linalg.norm(ref._so3_log(R_pi)), np.pi - 1e-7, atol=1e-6)


@pytest.mark.parametrize("name", ["iiwa14", "fr3"])
def test_frame_velocity_cost(name):
    ref = _ref(name, False)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    rng = np.random.default_rng(12)
    q = rng.standard_normal(nq); qd = rng.standard_normal(nv)
    x = np.concatenate([q, qd])
    W = np.arange(1, 7, dtype=float); v_des = np.array([0.1, 0.0, -0.1, 0.2, 0.0, 0.1]); eps = 0.05
    for frame in _frames(ref):
        val, grad, hess = ref.frame_velocity_cost(q, qd, frame, v_des, W, 1, eps)
        fd = _central(lambda xx: ref.frame_velocity_cost(xx[:nq], xx[nq:], frame, v_des, W, 1, eps)[0], x)
        assert np.allclose(grad, fd, atol=1e-6), np.max(np.abs(grad - fd))
        # the configuration derivative of J qd against a finite difference of the Jacobian itself
        A = ref._frame_velocity_q_derivative(q, qd, frame)
        F = np.zeros((6, nv))
        for k in range(nq):
            qp = q.copy(); qp[k] += _H; qm = q.copy(); qm[k] -= _H
            F[:, k] = (ref.frame_jacobian(qp, frame) @ qd - ref.frame_jacobian(qm, frame) @ qd) / (2 * _H)
        assert np.allclose(A, F, atol=1e-6)
        J = np.asarray(ref.frame_jacobian(q, frame))
        Jx = np.hstack([A, J]); curv = ref._activation(1, W, J @ qd - v_des, eps)[2]
        assert np.allclose(hess, Jx.T @ (curv[:, None] * Jx))
    # floating base: the configuration derivative is taken along the tangent (base twist in the
    # base frame, then the joint rates); check it against a finite difference through integrate
    go2 = _ref("go2", True)
    nq, nv = go2.robot.get_num_pos(), go2.robot.get_num_vel()
    q = rng.standard_normal(nq); q[3:7] /= np.linalg.norm(q[3:7]); qd = rng.standard_normal(nv)
    for frame in _frames(go2):
        val, grad, hess = go2.frame_velocity_cost(q, qd, frame, v_des, W, 1, eps)
        assert grad.shape == (2 * nv,) and hess.shape == (2 * nv, 2 * nv)
        A = go2._frame_velocity_q_derivative(q, qd, frame)
        F = np.zeros((6, nv))
        for k in range(nv):
            v = np.zeros(nv); v[k] = _H
            F[:, k] = (go2.frame_jacobian(go2.integrate(q, v), frame) @ qd - go2.frame_jacobian(go2.integrate(q, -v), frame) @ qd) / (2 * _H)
        assert np.allclose(A, F, atol=1e-6)
        fd = np.zeros(2 * nv)
        for k in range(2 * nv):
            d = np.zeros(2 * nv); d[k] = _H
            qp, qm = go2.integrate(q, d[:nv]), go2.integrate(q, -d[:nv])
            fd[k] = (go2.frame_velocity_cost(qp, qd + d[nv:], frame, v_des, W, 1, eps)[0]
                     - go2.frame_velocity_cost(qm, qd - d[nv:], frame, v_des, W, 1, eps)[0]) / (2 * _H)
        assert np.allclose(grad, fd, atol=1e-6), np.max(np.abs(grad - fd))


def test_state_cost_tangent():
    """The activated tangent state cost: quadratic activation == quadratic_state_cost_tangent;
    pseudo-Huber and quadratic-barrier gradients against a finite difference through integrate;
    the base-pose bounds are ignored; the GN Hessian is J_x^T diag(a'') J_x."""
    go2 = _ref("go2", True)
    nq, nv = go2.robot.get_num_pos(), go2.robot.get_num_vel()
    rng = np.random.default_rng(31)
    q = rng.standard_normal(nq); q[3:7] /= np.linalg.norm(q[3:7]); qd = rng.standard_normal(nv)
    x = np.concatenate([q, qd])
    delta = 0.05 * np.arange(1, nv + 1) * np.where(np.arange(nv) % 2 == 0, 1.0, -1.0)
    x_des = np.concatenate([go2.integrate(q, delta), qd + 0.03 * np.arange(1, nv + 1)])
    Q = 1.0 + 0.5 * np.arange(2 * nv)
    v0, g0, h0 = go2.state_cost_tangent(x, x_des, Q, 0, 0.0)
    v1, g1, h1 = go2.quadratic_state_cost_tangent(x, x_des, Q, gauss_newton=True)
    assert np.isclose(v0, v1) and np.allclose(g0, g1) and np.allclose(h0, h1)
    lo = np.concatenate([np.full(6, np.nan), q[7:] - 0.02, qd - 0.02]); hi = lo + 0.04
    for kind, e in ((1, 0.05), (3, 0.0), (4, 0.05)):
        val, grad, hess = go2.state_cost_tangent(x, x_des, Q, kind, e, lo, hi)
        fd = np.zeros(2 * nv)
        for k in range(2 * nv):
            d = np.zeros(2 * nv); d[k] = _H
            xp = np.concatenate([go2.integrate(q, d[:nv]), qd + d[nv:]]); xm = np.concatenate([go2.integrate(q, -d[:nv]), qd - d[nv:]])
            fd[k] = (go2.state_cost_tangent(xp, x_des, Q, kind, e, lo, hi)[0] - go2.state_cost_tangent(xm, x_des, Q, kind, e, lo, hi)[0]) / (2 * _H)
        assert np.allclose(grad, fd, atol=1e-5), (kind, np.max(np.abs(grad - fd)))
        Jq = go2.dDifference(x_des[:nq], q, "to"); Jx = np.zeros((2 * nv, 2 * nv)); Jx[:nv, :nv] = Jq; Jx[nv:, nv:] = np.eye(nv)
        r = np.concatenate([go2.difference(x_des[:nq], q), qd - x_des[nq:]])
        v = np.concatenate([np.zeros(6), q[7:], qd]); lo_t = lo.copy(); hi_t = hi.copy(); lo_t[:6] = -np.inf; hi_t[:6] = np.inf
        curv = go2._activation(kind, Q, r, e, v, lo_t, hi_t)[2]
        assert np.allclose(hess, Jx.T @ (curv[:, None] * Jx))
    # a bound on the base pose entries changes nothing
    tight = lo.copy(); tight[:6] = 10.0
    assert np.isclose(go2.state_cost_tangent(x, x_des, Q, 3, 0.0, tight, hi)[0], go2.state_cost_tangent(x, x_des, Q, 3, 0.0, lo, hi)[0])


def test_barrier_activations():
    W = np.array([0.1, 0.2, 0.3]); vals = np.array([0.5, 1.95, -0.3])
    lo = np.array([0.0, 0.0, -np.inf]); hi = np.array([2.0, 2.0, 0.0]); delta = 0.1
    for kind in (2, 3, 4):
        val, sl, cu = RBDReference._activation(kind, W, vals, delta, vals, lo, hi)
        fd = _central(lambda vv: np.sum(RBDReference._activation(kind, W, vv, delta, vv, lo, hi)[0]), vals)
        assert np.allclose(sl, fd, atol=1e-6)
        fd2 = _central(lambda vv: np.sum(RBDReference._activation(kind, W, vv, delta, vv, lo, hi)[1]), vals)
        assert np.allclose(cu, fd2, atol=1e-4)
    # smooth hinge: exactly zero inside the bounds, pseudo-Huber in the violation outside; the
    # curvature is 0 inside and w eps^2 / s^3 outside (w / eps just past a bound)
    inside = np.array([0.5, 1.0, -0.3]); outside = np.array([-0.2, 2.5, 0.4])
    assert np.all(RBDReference._activation(4, W, inside, delta, inside, lo, hi)[0] == 0.0)
    assert np.all(RBDReference._activation(4, W, inside, delta, inside, lo, hi)[2] == 0.0)
    d = np.array([0.2, 0.5, 0.4]); s = np.sqrt(d * d + delta * delta)
    val, sl, cu = RBDReference._activation(4, W, outside, delta, outside, lo, hi)
    assert np.allclose(val, W * (s - delta)) and np.allclose(sl, W * np.array([-1.0, 1.0, 1.0]) * d / s)
    assert np.allclose(cu, W * delta * delta / s ** 3)
    with pytest.raises(ValueError):
        RBDReference._activation(4, W, vals, 0.0, vals, lo, hi)
    # relaxed log barrier: log above delta, finite quadratic extension below and beyond the bound
    v, d1, d2 = RBDReference._relaxed_log_barrier_1d(np.array([0.5, 0.1, 0.05, -1.0]), 0.1, delta)
    assert np.isclose(v[0], -0.1 * np.log(0.5)) and np.isclose(v[1], -0.1 * np.log(0.1))
    assert np.isfinite(v).all() and v[3] > v[2] > v[1]
    # continuity of value and slope at delta
    a = RBDReference._relaxed_log_barrier_1d(np.array([delta + 1e-9]), 0.1, delta)
    b = RBDReference._relaxed_log_barrier_1d(np.array([delta - 1e-9]), 0.1, delta)
    assert abs(a[0][0] - b[0][0]) < 1e-7 and abs(a[1][0] - b[1][0]) < 1e-6
    # quadratic barrier is zero inside and the infinite bound contributes nothing
    val, sl, cu = RBDReference._activation(3, W, vals, 0.0, vals, lo, hi)
    assert val[0] == 0.0 and sl[0] == 0.0 and cu[0] == 0.0 and val[2] == 0.0
    assert np.isclose(val[2], 0.0) and np.isclose(RBDReference._activation(3, W, np.array([-0.3]), 0.0, np.array([-0.3]), np.array([-np.inf]), np.array([-0.5]))[0][0], 0.5 * 0.1 * 0.04)
    with pytest.raises(ValueError):
        RBDReference._activation(2, W, vals, delta)
