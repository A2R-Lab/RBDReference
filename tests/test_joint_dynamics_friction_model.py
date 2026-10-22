"""Joint-dynamics friction models: sign (default) and tanh (smooth).

RBDReference(use_joint_dynamics=True, friction_model=, friction_width=) applies
tau += damping*qd + friction*s(qd) with s = sign under the default model and
s = tanh(qd/w) under the smooth model. The smooth model has derivatives at and
near qd = 0, which is where the sign model has none: the inverse-dynamics
gradient gains (friction/w) sech^2(qd/w) on the dqd diagonal and the
second-order qd-qd block gains -(2 friction/w^2) tanh(qd/w) sech^2(qd/w) on
its triple diagonal. Every check here is analytic against the reference's own
finite differences, at qd = 0 and at |qd| <= w, on fr3 (damping, friction and
a mimic joint, so the per-slot fold is exercised) and on iiwa14 floating
(damping only, root skipped).
"""
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference

_ASSETS = Path(__file__).resolve().parents[1] / "robot_assets"
_W = 0.03


def _robot(name, floating=False):
    return URDFParser().parse(str(_ASSETS / f"{name}.urdf"), floating_base=floating)


def _samples(rng, nv, width):
    """qd at rest, inside the width, at the width, and well outside it."""
    at_rest = np.zeros(nv)
    inside = rng.uniform(-0.5, 0.5, nv) * width
    edge = np.sign(rng.standard_normal(nv)) * width
    outside = np.sign(rng.standard_normal(nv)) * (0.5 + np.abs(rng.standard_normal(nv)))
    return [at_rest, inside, edge, outside]


def _central(f, x, k, h):
    xp = x.copy(); xp[k] += h
    xm = x.copy(); xm[k] -= h
    return (np.asarray(f(xp)) - np.asarray(f(xm))) / (2.0 * h)


def test_invalid_model_and_width_rejected():
    robot = _robot("fr3")
    with pytest.raises(ValueError):
        RBDReference(robot, use_joint_dynamics=True, friction_model="smooth")
    with pytest.raises(ValueError):
        RBDReference(robot, use_joint_dynamics=True, friction_model="tanh", friction_width=0.0)
    with pytest.raises(ValueError):
        RBDReference(robot, use_joint_dynamics=True, friction_model="tanh",
                     friction_width=np.ones(robot.get_num_vel() + 1))


def test_sign_model_is_the_default_and_unchanged():
    robot = _robot("fr3")
    ref = RBDReference(robot, use_joint_dynamics=True)
    assert ref.friction_model == "sign"
    rng = np.random.default_rng(1)
    nv = robot.get_num_vel()
    q = rng.standard_normal(nv)
    for qd in _samples(rng, nv, _W):
        bias = ref._joint_dynamics_bias(qd)
        expected = np.zeros(nv)
        for jid in range(robot.get_num_bodies()):
            k = robot.get_joint_index_v(jid)
            alpha = ref._mimic_multiplier(jid)
            expected[k] += alpha * (robot.get_damping_by_id(jid) * qd[k]
                                    + robot.get_friction_by_id(jid) * np.sign(qd[k]))
        assert np.allclose(bias, expected, rtol=0, atol=1e-12)
        # the sign model carries no friction slope anywhere and no second derivative
        assert np.allclose(ref._joint_dynamics_derivative_diag(qd, 2), 0.0)
        grad = ref._joint_dynamics_derivative_diag(qd, 1)
        damping_only = np.zeros(nv)
        for jid in range(robot.get_num_bodies()):
            k = robot.get_joint_index_v(jid)
            damping_only[k] += ref._mimic_multiplier(jid) * robot.get_damping_by_id(jid)
        assert np.allclose(grad, damping_only, rtol=0, atol=1e-12)


@pytest.mark.parametrize("width", [_W, "per_slot"], ids=["scalar", "per_slot"])
def test_tanh_bias_and_slopes_match_closed_form(width):
    robot = _robot("fr3")
    nv = robot.get_num_vel()
    rng = np.random.default_rng(2)
    w = np.full(nv, _W) if width == _W else rng.uniform(0.01, 0.1, nv)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model="tanh",
                       friction_width=(_W if width == _W else w))
    for qd in _samples(rng, nv, w):
        bias = np.zeros(nv); g1 = np.zeros(nv); g2 = np.zeros(nv)
        for jid in range(robot.get_num_bodies()):
            k = robot.get_joint_index_v(jid)
            a = ref._mimic_multiplier(jid)
            b = robot.get_damping_by_id(jid)
            c = robot.get_friction_by_id(jid)
            th = np.tanh(qd[k] / w[k]); s2 = 1.0 - th * th
            bias[k] += a * (b * qd[k] + c * th)
            g1[k] += a * (b + c / w[k] * s2)
            g2[k] += a * (-2.0 * c / w[k] ** 2 * th * s2)
        assert np.allclose(ref._joint_dynamics_bias(qd), bias, rtol=0, atol=1e-12)
        assert np.allclose(ref._joint_dynamics_derivative_diag(qd, 1), g1, rtol=0, atol=1e-12)
        assert np.allclose(ref._joint_dynamics_derivative_diag(qd, 2), g2, rtol=0, atol=1e-12)
    # the friction slope is present at rest: this is the point the sign model misses
    assert np.max(ref._joint_dynamics_derivative_diag(np.zeros(nv), 1)) > 1.0


@pytest.mark.parametrize("name,floating", [("fr3", False), ("iiwa14", True)],
                         ids=["fr3-fixed", "iiwa14-floating"])
def test_tanh_gradient_matches_finite_difference(name, floating):
    robot = _robot(name, floating)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model="tanh", friction_width=_W)
    nv = robot.get_num_vel()
    rng = np.random.default_rng(3)
    q = rng.standard_normal(robot.get_num_pos())
    qdd = rng.standard_normal(nv)
    h = 1e-6
    for qd in _samples(rng, nv, _W):
        grad = np.asarray(ref.inverse_dynamics_gradient(q, qd, qdd, GRAVITY=-9.81))
        dqd_analytic = grad[:, nv:]
        fd = np.zeros((nv, nv))
        for k in range(nv):
            fd[:, k] = _central(lambda v: ref.inverse_dynamics(q, v, qdd, GRAVITY=-9.81)[0], qd, k, h)
        err = np.max(np.abs(dqd_analytic - fd))
        assert err < 1e-5, f"{name} qd={qd}: dc_dqd analytic vs FD {err:.2e}"
        # forward-dynamics gradient follows by composition
        u = rng.standard_normal(nv)
        _, qdd_dqd = ref.forward_dynamics_gradient(q, qd, u)
        fd2 = np.zeros((nv, nv))
        for k in range(nv):
            fd2[:, k] = _central(lambda v: ref.forward_dynamics(q, v, u), qd, k, h)
        err2 = np.max(np.abs(qdd_dqd - fd2)) / max(1.0, np.max(np.abs(fd2)))
        assert err2 < 1e-5, f"{name} qd={qd}: qdd_dqd analytic vs FD {err2:.2e}"


@pytest.mark.parametrize("name,floating", [("fr3", False), ("iiwa14", True)],
                         ids=["fr3-fixed", "iiwa14-floating"])
def test_tanh_second_order_matches_finite_difference_of_gradient(name, floating):
    robot = _robot(name, floating)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model="tanh", friction_width=_W)
    nv = robot.get_num_vel()
    rng = np.random.default_rng(4)
    q = rng.standard_normal(robot.get_num_pos())
    qdd = rng.standard_normal(nv)
    h = 1e-5
    for qd in _samples(rng, nv, _W):
        _, d2tau_dqd, _, _ = ref.idsva_so(q, qd, qdd, GRAVITY=-9.81)
        # d2tau[i, j, k] = d/dqd_k of dc_dqd[i, j]
        for k in range(nv):
            fd = _central(lambda v: np.asarray(ref.inverse_dynamics_gradient(q, v, qdd, GRAVITY=-9.81))[:, nv:], qd, k, h)
            err = np.max(np.abs(d2tau_dqd[:, :, k] - fd)) / max(1.0, np.max(np.abs(fd)))
            assert err < 1e-4, f"{name} qd={qd} k={k}: d2tau_dqd vs FD of gradient {err:.2e}"
        if name == "fr3" and np.all(np.abs(qd) <= _W):
            # the friction term is the dominant entry on the triple diagonal at |qd| <= w
            so = ref._joint_dynamics_derivative_diag(qd, 2)
            if np.any(qd != 0.0):
                assert np.max(np.abs(so)) > 1.0
        # fdsva_so follows by composition: its qd-qd block against FD of the FD gradient
        u = rng.standard_normal(nv)
        _, _, daba_dvdv, _ = ref.fdsva_so(q, qd, u, GRAVITY=-9.81)
        for k in range(nv):
            fd = _central(lambda v: ref.forward_dynamics_gradient(q, v, u)[1], qd, k, h)
            err = np.max(np.abs(daba_dvdv[:, :, k] - fd)) / max(1.0, np.max(np.abs(fd)))
            assert err < 1e-4, f"{name} qd={qd} k={k}: daba_dvdv vs FD {err:.2e}"
