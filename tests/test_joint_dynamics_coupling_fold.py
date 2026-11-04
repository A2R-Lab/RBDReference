"""Joint damping and Coulomb friction on coupled joints fold with alpha^2 and
|alpha| (user ruling 2026-11-03).

A follower's own torque b qd_f + f s(qd_f), qd_f = alpha qd_l, projects onto
its leader through alpha: alpha^2 b qd_l + |alpha| f s(qd_l), the congruence
G^T B G of the constraint embedding. The witnesses here are a chain with
ratios 2 and -1.5 (fr3, the only damped mimic robot of the suites, has a
ratio of one, under which the earlier single-alpha fold and this one
coincide), on both coupling kinds:

* the reduced recursions (linear) and the projection (polynomial) match the
  projection of a full spanning tree that carries the damping itself, where
  no reduced-coordinate convention is involved at all;
* the bias matches the closed form, the gradient matches finite differences
  through the chart, and the dissipated power yd^T bias is never negative,
  which the single-alpha fold violated on the negative ratio.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from RBDReference import RBDReference
from RBDReference._embedding import EmbeddedReference
from URDFParser import URDFParser

_INERTIAL = ('<inertial><origin xyz="0.05 0.02 0.1"/><mass value="1.3"/>'
             '<inertia ixx="0.04" ixy="0.001" ixz="0.002" iyy="0.05" iyz="0.003" izz="0.03"/></inertial>')
_DYN = {"j1": (0.1, 0.05), "j2": (0.4, 0.2), "j3": (0.25, 0.0), "j4": (0.3, 0.15)}
_LINEAR = {"j2": ("j1", 2.0, 0.1), "j4": ("j3", -1.5, -0.2)}
_POLY = {"j2": ("j1", "0.1 0.6 0.25 -0.1 0.04", "0.2"), "j4": ("j3", "0 -1.5 0 0.3", None)}


def _joint(name, parent, child, axis, xyz, rpy, mimic_xml):
    b, f = _DYN[name]
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis}"/><origin xyz="{xyz}" rpy="{rpy}"/>'
            f'<dynamics damping="{b}" friction="{f}"/>'
            '<limit lower="-3" upper="3" effort="10" velocity="10"/>' + mimic_xml + '</joint>')


def _mimic(kind, name):
    if kind == "linear":
        tgt, m, o = _LINEAR[name]
        return f'<mimic joint="{tgt}" multiplier="{m}" offset="{o}"/>'
    tgt, coef, q0 = _POLY[name]
    return f'<mimic joint="{tgt}" polycoef="{coef}"' + (f' q0="{q0}"' if q0 else "") + '/>'


def _urdf(tmp_path, kind):
    links = '<link name="base"/>' + "".join(f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, 5))
    joints = (_joint("j1", "base", "l1", "0 0 1", "0 0 0.1", "0 0 0", "")
              + _joint("j2", "l1", "l2", "0 1 0", "0.2 0 0.05", "0.1 0 0", _mimic(kind, "j2"))
              + _joint("j3", "l2", "l3", "1 0 0", "0.15 0.05 0", "0 0.2 0", "")
              + _joint("j4", "l3", "l4", "0 1 0", "0.1 0 0.1", "0 0 0.3", _mimic(kind, "j4")))
    path = tmp_path / f"{kind}_damped_chain.urdf"
    path.write_text(f'<robot name="{kind}_damped_chain">{links}{joints}</robot>')
    return path


def _parse(path, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(path), floating_base=floating)
    assert robot is not None
    return robot


def _state(rng, robot):
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    q = rng.standard_normal(nq)
    if robot.floating_base:
        q[3:7] /= np.linalg.norm(q[3:7])
    yd = rng.uniform(0.3, 1.5, nv) * rng.choice([-1.0, 1.0], nv)   # away from the Coulomb switch
    return q, yd, rng.standard_normal(nv)


def _chart_fd(ref, f, q, nv, h=1e-6):
    cols = []
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        cols.append((f(ref.integrate(q, e)) - f(ref.integrate(q, -e))) / (2 * h))
    return np.array(cols).T


def _plain_fd(f, x, h=1e-6):
    cols = []
    for k in range(len(x)):
        e = np.zeros(len(x)); e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.array(cols).T


def _followers(robot):
    """{follower jid: (leader v-slot, alpha(q) callable, b, f)} of the reduced robot."""
    out = {}
    for j in robot.get_joints_ordered_by_id():
        if getattr(j, "is_mimic", False):
            leader = robot.get_joint_by_id(j.get_mimic_target_id())
            lq = int(robot.get_joint_index_q(leader.get_id()))
            lv = int(robot.get_joint_index_v(leader.get_id()))
            out[j.get_id()] = (lq, lv, j.coupling_alpha, _DYN[j.get_name()])
    return out


def _closed_form_bias(robot, ref, q, yd):
    """alpha^2 b yd_l + |alpha| f s(yd_l) per joint, alpha = 1 on an independent joint."""
    nv = robot.get_num_vel()
    out = np.zeros(nv)
    fol = _followers(robot)
    for j in robot.get_joints_ordered_by_id():
        jid = j.get_id()
        if robot.floating_base and robot.get_parent_id(jid) == -1:
            continue
        b, f = _DYN[j.get_name()]
        if jid in fol:
            lq, k, alpha_fn, _ = fol[jid]
            al = alpha_fn(float(q[lq]))
        else:
            k, al = int(robot.get_joint_index_v(jid)), 1.0
        s = np.tanh(yd[k] / ref.friction_width[k]) if ref.friction_model == "tanh" else np.sign(yd[k])
        out[k] += al * al * b * yd[k] + abs(al) * f * s
    return out


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
@pytest.mark.parametrize("kind", ["linear", "polynomial"])
@pytest.mark.parametrize("model", ["sign", "tanh"])
def test_coupled_joint_dynamics_fold(tmp_path, kind, model, floating):
    robot = _parse(_urdf(tmp_path, kind), floating)
    assert robot.robot_has_joint_damping() and robot.robot_has_joint_friction()
    nv = robot.get_num_vel()
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model=model, friction_width=0.05)
    bare = RBDReference(robot)
    assert (ref._embedding is None) == (kind == "linear")
    fol = _followers(robot)
    assert len(fol) == 2 and any(fn(0.3) < 0 for _, _, fn, _ in fol.values())

    # the projection of a full tree that carries the damping itself (sign model:
    # alpha f sign(alpha qd_l) = |alpha| f sign(qd_l) exactly; the tanh width of
    # the follower's own slot is |alpha| times the reduced slot's so the smooth
    # shapes coincide too, which a constant alpha allows)
    full_robot = robot.full_tree_copy()
    if kind == "linear":
        w_full = np.full(full_robot.get_num_vel(), 0.05)
        for jid, (_, lv, alpha_fn, _) in fol.items():
            fv = int(full_robot.get_joint_index_v(full_robot.get_joint_by_name(robot.get_joint_by_id(jid).get_name()).get_id()))
            w_full[fv] = abs(alpha_fn(0.0)) * 0.05
        physical = EmbeddedReference(robot, RBDReference(full_robot, use_joint_dynamics=True,
                                                         friction_model=model, friction_width=w_full))
    else:
        physical = (EmbeddedReference(robot, RBDReference(full_robot, use_joint_dynamics=True))
                    if model == "sign" else None)

    rng = np.random.default_rng(11)
    for _ in range(4):
        q, yd, ydd = _state(rng, robot)
        c = ref.inverse_dynamics(q, yd, ydd)[0]
        bias = c - bare.inverse_dynamics(q, yd, ydd)[0]
        np.testing.assert_allclose(bias, _closed_form_bias(robot, ref, q, yd), rtol=0, atol=1e-10)
        # dissipation: the joint torques oppose the motion for every sign of alpha
        assert float(yd @ bias) > 0.0
        if physical is not None:
            np.testing.assert_allclose(c, physical.inverse_dynamics(q, yd, ydd)[0], rtol=0, atol=1e-9)
            np.testing.assert_allclose(ref.forward_dynamics(q, yd, ydd), physical.forward_dynamics(q, yd, ydd), rtol=0, atol=1e-8)
            np.testing.assert_allclose(ref.inverse_dynamics_gradient(q, yd, ydd),
                                       physical.inverse_dynamics_gradient(q, yd, ydd), rtol=0, atol=1e-8)
        # the analytic gradient against central differences of the value
        grad = ref.inverse_dynamics_gradient(q, yd, ydd)
        fd_q = _chart_fd(ref, lambda qq: ref.inverse_dynamics(qq, yd, ydd)[0], q, nv)
        fd_qd = _plain_fd(lambda v: ref.inverse_dynamics(q, v, ydd)[0], yd)
        np.testing.assert_allclose(grad[:, :nv], fd_q, rtol=0, atol=1e-6)
        np.testing.assert_allclose(grad[:, nv:], fd_qd, rtol=0, atol=1e-6)
        if kind == "polynomial":
            # the fold's own q-dependence: a nonzero dc_dq diagonal from the damping alone
            jd_dy, jd_dyd = ref._embedding.joint_dynamics_gradient(q, yd)
            assert np.max(np.abs(np.diag(jd_dy))) > 1e-3
            fd_jd = _chart_fd(ref, lambda qq: ref._embedding.joint_dynamics_bias(qq, yd), q, nv)
            np.testing.assert_allclose(jd_dy, fd_jd, rtol=0, atol=1e-6)
        fdg = ref.forward_dynamics_gradient(q, yd, ydd)
        fd_fq = _chart_fd(ref, lambda qq: ref.forward_dynamics(qq, yd, ydd), q, nv)
        np.testing.assert_allclose(fdg[0], fd_fq, rtol=0, atol=1e-5)


def test_single_alpha_fold_is_not_dissipative(tmp_path):
    """The reason for the ruling: on the ratio -1.5 the earlier alpha^1 fold
    produced a damping torque along the motion."""
    robot = _parse(_urdf(tmp_path, "linear"), False)
    j4 = robot.get_joint_by_name("j4")
    alpha, b = j4.get_mimic_multiplier(), _DYN["j4"][0]
    assert alpha < 0
    assert alpha * b < 0 < alpha * alpha * b
