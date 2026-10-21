"""Cost menu (phase one) reference checks: activations, activated terms, linear feature.

GRiD-defined quantities (the CUDA grid_plant surface), so there is no Pinocchio
oracle; each is checked against a closed form, a central-difference gradient and a
Gauss-Newton Hessian recompute on iiwa14 (fixed) and go2 (floating):
  - the quadratic activation reproduces quadratic_state_cost / quadratic_input_cost /
    ee_pos_cost exactly,
  - the pseudo-Huber activation has the stated value, slope and curvature and the
    weighted-absolute-value and quadratic limits,
  - state_cost / input_cost / linear_feature_cost / ee_pos_cost_activated gradients
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
    q = ref.robot.normalize_floating_base_q_input(rng.standard_normal(nq)) if floating else rng.standard_normal(nq)
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
def test_activated_ee_position_cost(name, floating):
    ref = _ref(name, floating)
    nq, nv = ref.robot.get_num_pos(), ref.robot.get_num_vel()
    rng = np.random.default_rng(9)
    q = ref.robot.normalize_floating_base_q_input(rng.standard_normal(nq)) if floating else rng.standard_normal(nq)
    W = np.array([10.0, 11.0, 12.0]); p_des = np.array([0.1, -0.2, 0.3]); eps = 0.02
    # quadratic activation reproduces ee_pos_cost
    for a, b in zip(ref.ee_pos_cost_activated(q, p_des, W, 0, 0.0), ref.ee_pos_cost(q, p_des, W)):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-12)
    val, grad_x, hess_x = ref.ee_pos_cost_activated(q, p_des, W, 1, eps)
    assert np.count_nonzero(grad_x[nv:]) == 0 and np.count_nonzero(hess_x[nv:, :]) == 0
    if not floating:
        fd = _central(lambda qq: ref.ee_pos_cost_activated(qq, p_des, W, 1, eps)[0], q)
        assert np.allclose(grad_x[:nv], fd, atol=1e-6), np.max(np.abs(grad_x[:nv] - fd))
    # Gauss-Newton structure from the position Jacobian
    target = ref._ee_target_name(0)
    Jp = np.asarray(ref.end_effector_pose_gradient(q, ee_joint_names=target)[0])[:3, :]
    p = np.asarray(ref.end_effector_pose(q, ee_joint_names=target)[0]).reshape(-1)[:3]
    _, slope, curv = ref._activation(1, W, p - p_des, eps)
    assert np.allclose(grad_x[:nv], Jp.T @ slope) and np.allclose(hess_x[:nv, :nv], Jp.T @ (curv[:, None] * Jp))
