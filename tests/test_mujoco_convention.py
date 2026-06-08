"""Validation for the MuJoCo / mjx output-convention transforms
(:mod:`RBDReference.equivalents.mujoco_convention`).

Two layers:

* **FD self-consistency (no MuJoCo needed)** -- the analytic first-order mjx
  gradients must equal the finite difference of the mjx *value* transforms taken
  along the genuine mjx retract (global-linear base step), with the velocity /
  acceleration / force inputs held fixed in the MJX frame (MuJoCo's native
  derivative convention). Plus value round-trips and the fixed-base no-op.
* **Direct MuJoCo cross-check (skipped if ``mujoco`` absent)** -- on a minimal
  hand-authored floating model that is the SAME system as a matched MJCF, GRiD's
  transformed mass matrix / inverse-dynamics / forward-dynamics must equal
  ``mj_fullM`` / ``mj_inverse`` / ``mj_forward`` to ~1e-12.

The value transforms were confirmed against real MuJoCo to machine precision; the
gradients are then validated as the derivative of those (MuJoCo-matched) values.
"""
import importlib.util

import numpy as np
import pytest

import RBDReference.equivalents.mujoco_convention as mc

pytestmark = [pytest.mark.developer_only]

_HAVE_DEPS = all(
    importlib.util.find_spec(mod) is not None
    for mod in ("robot_descriptions",)
)
_HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _go2(base_mode):
    from RBDReference.tests.model_sources import (
        load_manifest, select_robot_specs, resolve_robot_spec)
    from RBDReference.tests import MANIFEST_PATH
    from RBDReference.equivalents.reference_backend import build_project_adapter
    specs = {s.robot_id: s for s in select_robot_specs(load_manifest(MANIFEST_PATH), tier="smoke")}
    return build_project_adapter(specs["go2"], resolve_robot_spec(specs["go2"]), base_mode)


def _rand_state(rng, nq, nv):
    q = rng.standard_normal(nq)
    if nq > nv:                       # floating: normalise the free-flyer quat
        q[3:7] /= np.linalg.norm(q[3:7])
    return q, rng.standard_normal(nv), rng.standard_normal(nv), rng.standard_normal(nv)


# ---------------------------------------------------------------------------
# Value round-trips + fixed-base no-op (no external deps beyond the model)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_value_round_trip_floating():
    ad = _go2("floating"); nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(0)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    assert np.allclose(mc.q_mjx_to_pin(mc.q_pin_to_mjx(q, L), L), q, atol=1e-14)
    assert np.allclose(mc.v_mjx_to_pin(mc.v_pin_to_mjx(qd, R, L), R, L), qd, atol=1e-13)
    a_mjx = mc.accel_pin_to_mjx(qdd, qd, R, L)
    assert np.allclose(mc.accel_mjx_to_pin(a_mjx, qd, R, L), qdd, atol=1e-13)


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_fixed_base_is_noop():
    ad = _go2("fixed"); nq, nv = ad.nq, ad.nv
    assert nq == nv
    L = mc.FIXED_BASE
    rng = np.random.default_rng(1)
    q, qd, qdd, u = _rand_state(rng, nq, nv)
    R = np.eye(3)
    assert np.array_equal(mc.q_pin_to_mjx(q, L), q)
    assert np.array_equal(mc.v_pin_to_mjx(qd, R, L), qd)
    M = ad.crba(q)
    assert np.array_equal(mc.mass_matrix_pin_to_mjx(M, R, L), M)
    dtdq, dtdqd = ad.inverse_dynamics_gradient(q, qd, qdd)
    g_dq, g_dqd = mc.id_gradient_pin_to_mjx(dtdq, dtdqd, M, ad.inverse_dynamics(q, qd, qdd),
                                            qd, qdd, R, L)
    assert np.array_equal(g_dq, dtdq) and np.array_equal(g_dqd, dtdqd)


# ---------------------------------------------------------------------------
# First-order gradient FD self-consistency along the mjx retract
# ---------------------------------------------------------------------------

def _fd_jac_q(value_fn, q, ref, nv, L, h=1e-6):
    out = None
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        plus = value_fn(mc.mjx_retract(q, e, ref, L))
        minus = value_fn(mc.mjx_retract(q, -e, ref, L))
        if out is None:
            out = np.zeros((plus.shape[0], nv))
        out[:, k] = (plus - minus) / (2 * h)
    return out


def _fd_jac_vec(value_fn, x, nv, h=1e-6):
    out = None
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        plus, minus = value_fn(x + e), value_fn(x - e)
        if out is None:
            out = np.zeros((plus.shape[0], nv))
        out[:, k] = (plus - minus) / (2 * h)
    return out


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_id_gradient_matches_fd_of_value():
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    v_mjx = mc.v_pin_to_mjx(qd, R, L); a_mjx = mc.accel_pin_to_mjx(qdd, qd, R, L)

    def tau_mjx_q(qq):
        RR = mc.base_rotation(qq, L)
        vp = mc.v_mjx_to_pin(v_mjx, RR, L); ap = mc.accel_mjx_to_pin(a_mjx, vp, RR, L)
        return mc.id_tau_pin_to_mjx(np.asarray(ad.inverse_dynamics(qq, vp, ap)), RR, L)

    def tau_mjx_v(vm):
        vp = mc.v_mjx_to_pin(vm, R, L); ap = mc.accel_mjx_to_pin(a_mjx, vp, R, L)
        return mc.id_tau_pin_to_mjx(np.asarray(ad.inverse_dynamics(q, vp, ap)), R, L)

    fd_dq = _fd_jac_q(tau_mjx_q, q, ref, nv, L)
    fd_dqd = _fd_jac_vec(tau_mjx_v, v_mjx, nv)
    M = ad.crba(q); tau_pin = np.asarray(ad.inverse_dynamics(q, qd, qdd))
    dtdq, dtdqd = ad.inverse_dynamics_gradient(q, qd, qdd)
    an_dq, an_dqd = mc.id_gradient_pin_to_mjx(dtdq, dtdqd, M, tau_pin, qd, qdd, R, L)
    assert np.abs(an_dq - fd_dq).max() < 1e-5
    assert np.abs(an_dqd - fd_dqd).max() < 1e-5


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_fd_gradient_matches_fd_of_value():
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, _, u = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    v_mjx = mc.v_pin_to_mjx(qd, R, L); uf_mjx = mc.v_pin_to_mjx(u, R, L)

    def qdd_mjx_q(qq):
        RR = mc.base_rotation(qq, L)
        vp = mc.v_mjx_to_pin(v_mjx, RR, L); up = mc.force_mjx_to_pin(uf_mjx, RR, L)
        return mc.fd_qdd_pin_to_mjx(np.asarray(ad.forward_dynamics(qq, vp, up)), vp, RR, L)

    def qdd_mjx_v(vm):
        vp = mc.v_mjx_to_pin(vm, R, L); up = mc.force_mjx_to_pin(uf_mjx, R, L)
        return mc.fd_qdd_pin_to_mjx(np.asarray(ad.forward_dynamics(q, vp, up)), vp, R, L)

    fd_dq = _fd_jac_q(qdd_mjx_q, q, ref, nv, L)
    fd_dqd = _fd_jac_vec(qdd_mjx_v, v_mjx, nv)
    Minv = ad.minv(q); qdd_val = np.asarray(ad.forward_dynamics(q, qd, u))
    ddq, ddqd = ad.forward_dynamics_gradient(q, qd, u)
    an_dq, an_dqd = mc.fd_gradient_pin_to_mjx(ddq, ddqd, Minv, qdd_val, qd, u, R, L)
    assert np.abs(an_dq - fd_dq).max() < 1e-5
    assert np.abs(an_dqd - fd_dqd).max() < 1e-5


# ---------------------------------------------------------------------------
# Direct MuJoCo cross-check on a minimal matched model (skipped if no mujoco)
# ---------------------------------------------------------------------------

_MINI_MJCF = """
<mujoco><option gravity="0 0 -9.81"/><worldbody>
  <body name="base"><freejoint/><inertial pos="0 0 0" mass="2.0" diaginertia="0.1 0.2 0.3"/>
    <body name="l1" pos="0.3 0 0"><joint name="j1" type="hinge" axis="0 1 0"/>
      <inertial pos="0.1 0 0" mass="1.0" diaginertia="0.05 0.05 0.05"/>
    </body></body>
</worldbody></mujoco>
"""
_MINI_URDF = """
<robot name="mini">
  <link name="base"><inertial><origin xyz="0 0 0"/><mass value="2.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/></inertial></link>
  <link name="l1"><inertial><origin xyz="0.1 0 0"/><mass value="1.0"/>
    <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="100" velocity="100"/></joint>
</robot>
"""


@pytest.mark.skipif(not _HAVE_MUJOCO, reason="needs mujoco")
def test_value_matches_real_mujoco(tmp_path):
    import mujoco
    from URDFParser import URDFParser
    from URDFParser.Joint import Joint
    from RBDReference import RBDReference
    urdf = tmp_path / "mini.urdf"; urdf.write_text(_MINI_URDF)
    Joint.floating_base = True
    robot = URDFParser().parse(str(urdf), floating_base=True)
    ref = RBDReference(robot); nv = robot.get_num_vel()
    m = mujoco.MjModel.from_xml_string(_MINI_MJCF); d = mujoco.MjData(m)
    L = mc.FloatingRootLayout()
    rng = np.random.default_rng(3)
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    q_pin = np.concatenate([rng.standard_normal(3), quat, [0.4]])
    qd = rng.standard_normal(nv); qdd = rng.standard_normal(nv); u = rng.standard_normal(nv)
    R = mc.base_rotation(q_pin, L); q_mjx = mc.q_pin_to_mjx(q_pin, L)

    # mass matrix vs mj_fullM
    d.qpos[:] = q_mjx; mujoco.mj_forward(m, d)
    M_mj = np.zeros((nv, nv)); mujoco.mj_fullM(m, M_mj, d.qM)
    M_grid = mc.mass_matrix_pin_to_mjx(np.asarray(ref.crba(q_pin)), R, L)
    assert np.abs(M_grid - M_mj).max() < 1e-9

    # inverse dynamics vs mj_inverse (mjx-frame inputs)
    v_mjx = mc.v_pin_to_mjx(qd, R, L); a_mjx = mc.accel_pin_to_mjx(qdd, qd, R, L)
    d.qpos[:] = q_mjx; d.qvel[:] = v_mjx; d.qacc[:] = a_mjx; mujoco.mj_inverse(m, d)
    tau_grid = mc.id_tau_pin_to_mjx(np.asarray(ref.inverse_dynamics(q_pin, qd, qdd)[0]), R, L)
    assert np.abs(tau_grid - d.qfrc_inverse).max() < 1e-9

    # forward dynamics vs mj_forward (applied force in mjx frame)
    uf_mjx = mc.v_pin_to_mjx(u, R, L)
    d.qpos[:] = q_mjx; d.qvel[:] = v_mjx; d.qfrc_applied[:] = uf_mjx; d.qacc[:] = 0
    mujoco.mj_forward(m, d); acc_mj = d.qacc.copy(); d.qfrc_applied[:] = 0
    acc_grid = mc.fd_qdd_pin_to_mjx(np.asarray(ref.forward_dynamics(q_pin, qd, u)), qd, R, L)
    assert np.abs(acc_grid - acc_mj).max() < 1e-9
