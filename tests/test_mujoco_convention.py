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


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_value_transforms_match_fd_and_consistency():
    """The row / column / congruence value transforms (jacobian column-reframe,
    coriolis similarity, regressor row-rotate, nonlinear-effects accel-couple)
    each satisfy their defining relationship to machine / FD precision."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    tgt = ad.joint_names[-1]

    # column-reframe: ee_pose_gradient = d(invariant ee_pose)/dxi_mjx (FD along mjx retract)
    g_mjx = mc.jacobian_pin_to_mjx(np.asarray(ad.end_effector_pose_gradient(q, tgt)), R, L)
    fd = _fd_jac_q(lambda qq: np.asarray(ad.end_effector_pose(qq, tgt)), q, ref, nv, L)
    assert np.abs(g_mjx - fd).max() < 1e-5

    # nonlinear-effects: REVISED accel-couple == G . ID(q, qd, accel_mjx_to_pin(0))
    nle = np.asarray(ad.nonlinear_effects(q, qd)); M = np.asarray(ad.crba(q))
    a_pin0 = mc.accel_mjx_to_pin(np.zeros(nv), qd, R, L)
    nle_ref = mc.id_tau_pin_to_mjx(np.asarray(ad.inverse_dynamics(q, qd, a_pin0)), R, L)
    assert np.abs(mc.nonlinear_effects_pin_to_mjx(nle, M, qd, R, L) - nle_ref).max() < 1e-9
    # and the naive covector treatment is grossly wrong (the trap this guards)
    assert np.abs(mc.id_tau_pin_to_mjx(nle, R, L) - nle_ref).max() > 1.0

    # coriolis similarity: C qd is a covector matching the tau transform
    C = np.asarray(ad.coriolis_matrix(q, qd))
    lhs = mc.coriolis_matrix_pin_to_mjx(C, R, L) @ mc.v_pin_to_mjx(qd, R, L)
    assert np.abs(lhs - mc.id_tau_pin_to_mjx(C @ qd, R, L)).max() < 1e-10

    # regressor row-rotate: G . (Y pi) == (G Y) pi  for any params pi
    Y = np.asarray(ad.inverse_dynamics_regressor(q, qd, qdd))
    x = rng.standard_normal(Y.shape[1])
    assert np.abs(mc.id_tau_pin_to_mjx(Y @ x, R, L)
                  - mc.base_rotate_pin_to_mjx(Y, R, L) @ x).max() < 1e-10


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_dccrba_dh_dq_vel_couple_matches_fd():
    """The centroidal dh/dq transform (vel-couple revision) matches FD of the
    (invariant) centroidal momentum wrt q along the mjx retract."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(3)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    A = np.asarray(ad.ccrba(q, qd)[0])
    dh_dq = np.asarray(ref.centroidal_dynamics_derivatives(q, qd, qdd)[0])
    an = mc.dccrba_dh_dq_pin_to_mjx(dh_dq, A, qd, R, L)
    v_mjx = mc.v_pin_to_mjx(qd, R, L)

    def h_q(qq):
        RR = mc.base_rotation(qq, L)
        return np.asarray(ad.centroidal_momentum(qq, mc.v_mjx_to_pin(v_mjx, RR, L)))

    fd = _fd_jac_q(h_q, q, ref, nv, L)
    assert np.abs(an - fd).max() < 1e-5
    # column-reframe alone (no vel-couple) is O(1) wrong
    assert np.abs(mc.jacobian_pin_to_mjx(dh_dq, R, L) - fd).max() > 1.0


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_ee_pose_hessian_symmetrized_matches_coord_hess():
    """The EE-pose Hessian transform (symmetrized frame-correction revision) matches
    the symmetric coordinate Hessian of the invariant pose along the mjx retract, to
    the same FD floor as GRiD's own analytic-vs-coordinate pin Hessian check."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(11)
    q, _, _, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L); tgt = ad.joint_names[-1]
    H_pin = np.asarray(ad.end_effector_pose_hessian(q, tgt))
    dpose = np.asarray(ad.end_effector_pose_gradient(q, tgt))

    def ee(qq):
        return np.asarray(ad.end_effector_pose(qq, tgt))

    def coord_hess(retract, h=1e-4):
        Hd = np.zeros((H_pin.shape[0], nv, nv)); f0 = ee(q)
        for a in range(nv):
            for k in range(a, nv):
                ea = np.zeros(nv); ea[a] = h; ek = np.zeros(nv); ek[k] = h
                val = (ee(retract(q, ea + ek)) - ee(retract(q, ea)) - ee(retract(q, ek))
                       + 2 * f0 - ee(retract(q, -ea)) - ee(retract(q, -ek))
                       + ee(retract(q, -(ea + ek)))) / (2 * h * h)
                Hd[:, a, k] = val; Hd[:, k, a] = val
        return Hd

    floor = np.abs(H_pin - coord_hess(lambda qq, xi: ref.integrate(qq, xi))).max()
    an = mc.ee_pose_hessian_pin_to_mjx(H_pin, dpose, R, L)
    mjx_fd = coord_hess(lambda qq, xi: mc.mjx_retract(qq, xi, ref, L))
    # the transform adds no error beyond the shared FD-truncation floor
    assert np.abs(an - mjx_fd).max() < floor + 1e-6
    # double-reframe alone (no symmetrized frame correction) is materially wrong
    G = mc.g_matrix(R, nv, L); Ginv = G.T
    assert np.abs(np.einsum('ibc,ba,ck->iak', H_pin, Ginv, Ginv) - mjx_fd).max() > 0.1


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
def test_dM_dq_tensor_matches_fd():
    """The mass-matrix gradient tensor transform (one of the idsva_so outputs)
    matches FD of the mjx mass matrix along the mjx retract."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, _, _, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)

    def Mmjx(qq):
        return mc.mass_matrix_pin_to_mjx(np.asarray(ad.crba(qq)), mc.base_rotation(qq, L), L)

    fd = np.zeros((nv, nv, nv)); h = 1e-6
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        fd[:, :, k] = (Mmjx(mc.mjx_retract(q, e, ref, L)) - Mmjx(mc.mjx_retract(q, -e, ref, L))) / (2 * h)
    # pin dM/dq via FD of crba along the PIN retract (stands in for GRiD's dm_dq tensor)
    dM_dq_pin = np.zeros((nv, nv, nv))
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        dM_dq_pin[:, :, k] = (np.asarray(ad.crba(ref.integrate(q, e)))
                              - np.asarray(ad.crba(ref.integrate(q, -e)))) / (2 * h)
    an = mc.dM_dq_pin_to_mjx(dM_dq_pin, np.asarray(ad.crba(q)), R, L)
    assert np.abs(an - fd).max() < 1e-5


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_ee_pos_cost_transform_matches_fd_and_direct():
    """ee_pos_cost (q-block GN cost): value invariant; grad reframes as a covector
    (G·) — checked vs FD of the value along the mjx retract; GN hess reframes by
    congruence (G·Gᵀ) — checked vs an independent mjx-frame recompute JᵀWJ with the
    reframed EE Jacobian."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(31)
    q, _, _, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    W = rng.standard_normal(3) ** 2 + 0.1
    p_des = rng.standard_normal(3)
    _, grad_pin, hess_pin = ref.ee_pos_cost(q, p_des, W)
    g_mjx, H_mjx = mc.quadratic_tracking_cost_pin_to_mjx(grad_pin, hess_pin, R, 0, nv, L)
    # grad vs FD of the invariant value along the mjx retract
    fd = _fd_jac_q(lambda qq: np.array([ref.ee_pos_cost(qq, p_des, W)[0]]), q, ref, nv, L)
    assert np.abs(g_mjx[:nv] - fd[0]).max() < 1e-5
    # GN hess vs independent mjx-frame recompute Jp_mjx^T diag(W) Jp_mjx
    tgt = ref._ee_target_name(0)
    Jfull = np.asarray(ref.end_effector_pose_gradient(q, ee_joint_names=tgt)[0])
    Jp_mjx = mc.jacobian_pin_to_mjx(Jfull, R, L)[:3]
    assert np.abs(H_mjx[:nv, :nv] - Jp_mjx.T @ (W[:, None] * Jp_mjx)).max() < 1e-9


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_momentum_cost_transform_matches_direct():
    """momentum_cost (qd-block GN cost): h invariant ⇒ value invariant; the qd-block
    grad/hess reframe by the CMM column map (A_mjx = A_pin G^{-1}), i.e. covector G·
    on the grad and congruence on the hess. Checked vs an independent mjx-frame
    recompute with the reframed CMM."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(33)
    q, qd, _, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    W = rng.standard_normal(6) ** 2 + 0.1
    h_des = rng.standard_normal(6)
    _, grad_pin, hess_pin = ref.momentum_cost(q, qd, h_des, W)
    g_mjx, H_mjx = mc.quadratic_tracking_cost_pin_to_mjx(grad_pin, hess_pin, R, nq, nv, L)
    # independent mjx recompute: A_mjx = A_pin G^{-1}; r invariant (h, h_des fixed).
    A_pin, h = ref.ccrba(q, qd)
    A_mjx = mc.jacobian_pin_to_mjx(np.asarray(A_pin), R, L)
    r = np.asarray(h).reshape(-1) - h_des
    assert np.abs(g_mjx[nq:nq + nv] - A_mjx.T @ (W * r)).max() < 1e-9
    assert np.abs(H_mjx[nq:nq + nv, nq:nq + nv] - A_mjx.T @ (W[:, None] * A_mjx)).max() < 1e-9


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_quadratic_state_cost_transform_matches_fd_and_direct():
    """quadratic_state_cost (BOTH state blocks live; value convention-DEPENDENT).

    The GRiD kernel evaluates the cost on the PIN-frame state, so under mjx it
    input-converts the velocity block ``qd_pin = G⁻¹ qd_mjx`` before differencing
    against the user's mjx ``x_des``/``Q``. The qd-block of (grad, hess) then
    reframes back by ``G`` (covector / congruence); the q-block is untouched.

    * grad qd-block vs FD of the value w.r.t. the (plain) mjx velocity tangent.
    * hess qd-block vs FD of the value w.r.t. the mjx velocity tangent.
    * q-block grad == ``Q⊙(q-q_des)`` exactly (convention-invariant raw coords).
    * value is convention-dependent (pin-frame qd != mjx-frame qd).
    """
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    nx = nq + nv
    rng = np.random.default_rng(41)
    q, _, _, _ = _rand_state(rng, nq, nv)
    v_mjx = rng.standard_normal(nv)
    R = mc.base_rotation(q, L)
    x_des = rng.standard_normal(nx)
    Q = rng.standard_normal(nx) ** 2 + 0.1

    # the cost the mjx kernel computes: qd input-converted to pin, residual vs x_des.
    def value(qq, vv_mjx):
        RR = mc.base_rotation(qq, L)
        v_pin = mc.v_mjx_to_pin(vv_mjx, RR, L)
        r = np.concatenate([qq, v_pin]) - x_des
        return 0.5 * float(np.sum(Q * r * r))

    v_pin = mc.v_mjx_to_pin(v_mjx, R, L)
    _, grad_pin, hess_pin = ref.quadratic_state_cost(np.concatenate([q, v_pin]), x_des, Q)
    g_mjx, H_mjx = mc.quadratic_state_cost_pin_to_mjx(grad_pin, hess_pin, R, nq, nv, L)

    # qd-block grad vs FD of the value wrt the mjx velocity (a plain tangent: exact).
    h = 1e-6
    fd_qd = np.zeros(nv)
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        fd_qd[k] = (value(q, v_mjx + e) - value(q, v_mjx - e)) / (2 * h)
    assert np.abs(g_mjx[nq:nx] - fd_qd).max() < 1e-5

    # qd-block hess vs FD of the value wrt the mjx velocity.
    fd_H = np.zeros((nv, nv)); hh = 1e-5
    for a in range(nv):
        for b in range(nv):
            ea = np.zeros(nv); ea[a] = hh; eb = np.zeros(nv); eb[b] = hh
            fd_H[a, b] = (value(q, v_mjx + ea + eb) - value(q, v_mjx + ea - eb)
                          - value(q, v_mjx - ea + eb) + value(q, v_mjx - ea - eb)) / (4 * hh * hh)
    assert np.abs(H_mjx[nq:nx, nq:nx] - fd_H).max() < 1e-4

    # q-block is the untouched raw-coordinate gradient; cross/q-q hess blocks unchanged.
    assert np.abs(g_mjx[:nq] - Q[:nq] * (q - x_des[:nq])).max() < 1e-12
    assert np.abs(H_mjx[:nq, :nq] - np.diag(Q[:nq])).max() < 1e-12
    assert np.abs(H_mjx[:nq, nq:nx]).max() < 1e-12 and np.abs(H_mjx[nq:nx, :nq]).max() < 1e-12

    # value is convention-dependent (pin qd != mjx qd) -- a defining property.
    r_pin = np.concatenate([q, v_pin]) - x_des
    r_mjx = np.concatenate([q, v_mjx]) - x_des
    assert abs(0.5 * np.sum(Q * r_pin * r_pin) - 0.5 * np.sum(Q * r_mjx * r_mjx)) > 1e-6

    # fixed-base no-op
    g0, H0 = mc.quadratic_state_cost_pin_to_mjx(grad_pin, hess_pin, R, nq, nv, mc.FIXED_BASE)
    assert np.array_equal(g0, np.asarray(grad_pin)) and np.array_equal(H0, np.asarray(hess_pin))


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_dccrba_dA_dq_tensor_matches_fd():
    """The CMM gradient TENSOR dA/dq transform (GRiD ``dccrba``) matches FD of the
    mjx CMM (A_mjx = A_pin G^{-1}) along the mjx retract."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(9)
    q, qd, _, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)

    def Amjx(qq):
        A = np.asarray(ref.ccrba(qq, qd)[0])
        return mc.jacobian_pin_to_mjx(A, mc.base_rotation(qq, L), L)

    fd = np.zeros((6, nv, nv)); h = 1e-6
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        fd[:, :, k] = (Amjx(mc.mjx_retract(q, e, ref, L)) - Amjx(mc.mjx_retract(q, -e, ref, L))) / (2 * h)
    dA_dq_pin = np.asarray(ref.dccrba(q))                 # [i, l, m] = dA_il/dq_m
    A_pin = np.asarray(ref.ccrba(q, qd)[0])
    an = mc.dccrba_dA_dq_pin_to_mjx(dA_dq_pin, A_pin, R, L)
    assert np.abs(an - fd).max() < 1e-5
    # the double-reframe WITHOUT the base-rotation frame term is O(1) wrong.
    G = mc.g_matrix(R, nv, L); Ginv = G.T
    no_frame = np.einsum('ilm,la,mk->iak', dA_dq_pin, Ginv, Ginv)
    assert np.abs(no_frame - fd).max() > 0.1


def second_order_id_reference(ad, q, qd, qdd, k_dirs, L, h=1e-6):
    """Numerical reference for the mjx d2tau/dq2 tensor: central-difference the
    VALIDATED analytic first-order mjx ``dtau/dq`` along the mjx retract. Correct by
    construction (it differentiates the MuJoCo-matched first-order transform); the
    validation target for any future closed form. Returns ``(nv,nv,len(k_dirs))``."""
    ref = ad.reference; nv = ad.nv
    R0 = mc.base_rotation(q, L)
    v_mjx = mc.v_pin_to_mjx(qd, R0, L); a_mjx = mc.accel_pin_to_mjx(qdd, qd, R0, L)

    def dtau_dq_mjx(qq):
        RR = mc.base_rotation(qq, L)
        vp = mc.v_mjx_to_pin(v_mjx, RR, L); ap = mc.accel_mjx_to_pin(a_mjx, vp, RR, L)
        M = ad.crba(qq); tau = np.asarray(ad.inverse_dynamics(qq, vp, ap))
        dq, dqd = ad.inverse_dynamics_gradient(qq, vp, ap)
        return mc.id_gradient_pin_to_mjx(dq, dqd, M, tau, vp, ap, RR, L)[0]

    out = np.zeros((nv, nv, len(k_dirs)))
    for col, k in enumerate(k_dirs):
        e = np.zeros(nv); e[k] = h
        out[:, :, col] = (dtau_dq_mjx(mc.mjx_retract(q, e, ref, L))
                          - dtau_dq_mjx(mc.mjx_retract(q, -e, ref, L))) / (2 * h)
    return out


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_second_order_id_tensors_match_fd():
    """All four idsva_so tensors transformed pin->mjx match FD of the validated
    analytic first-order mjx gradients along the mjx retract."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    v_mjx = mc.v_pin_to_mjx(qd, R, L); a_mjx = mc.accel_pin_to_mjx(qdd, qd, R, L)

    def grads(qq, vm, am):
        RR = mc.base_rotation(qq, L)
        vp = mc.v_mjx_to_pin(vm, RR, L); ap = mc.accel_mjx_to_pin(am, vp, RR, L)
        M = ad.crba(qq); tau = np.asarray(ad.inverse_dynamics(qq, vp, ap))
        dq, dqd = ad.inverse_dynamics_gradient(qq, vp, ap)
        return mc.id_gradient_pin_to_mjx(dq, dqd, M, tau, vp, ap, RR, L)

    h = 1e-6
    ref_d2q = np.zeros((nv, nv, nv)); ref_cross = np.zeros((nv, nv, nv)); ref_d2qd = np.zeros((nv, nv, nv))
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        gp = grads(mc.mjx_retract(q, e, ref, L), v_mjx, a_mjx)
        gm = grads(mc.mjx_retract(q, -e, ref, L), v_mjx, a_mjx)
        ref_d2q[:, :, k] = (gp[0] - gm[0]) / (2 * h); ref_cross[:, :, k] = (gp[1] - gm[1]) / (2 * h)
        gp = grads(q, v_mjx + e, a_mjx); gm = grads(q, v_mjx - e, a_mjx)
        ref_d2qd[:, :, k] = (gp[1] - gm[1]) / (2 * h)
    so = tuple(np.asarray(t) for t in ref.idsva_so(q, qd, qdd))
    dtdq, dtdqd = ad.inverse_dynamics_gradient(q, qd, qdd)
    d2q, d2qd, cross, dM = mc.second_order_id_pin_to_mjx(
        so, dtdq, dtdqd, ad.crba(q), np.asarray(ad.inverse_dynamics(q, qd, qdd)), qd, qdd, R, L)
    assert np.abs(d2q - ref_d2q).max() < 1e-5
    assert np.abs(d2qd - ref_d2qd).max() < 1e-5
    assert np.abs(cross - ref_cross).max() < 1e-5


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_second_order_fixed_base_is_noop():
    ad = _go2("fixed"); nv = ad.nv; L = mc.FIXED_BASE
    rng = np.random.default_rng(2)
    q, qd, qdd, _ = _rand_state(rng, ad.nq, nv)
    so = tuple(np.asarray(t) for t in ad.reference.idsva_so_body_frame(q, qd, qdd))
    dtdq, dtdqd = ad.inverse_dynamics_gradient(q, qd, qdd)
    out = mc.second_order_id_pin_to_mjx(so, dtdq, dtdqd, ad.crba(q),
                                        np.asarray(ad.inverse_dynamics(q, qd, qdd)), qd, qdd, np.eye(3), L)
    for o, s in zip(out, so):
        assert np.array_equal(o, s)


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_second_order_reference_is_well_defined():
    """The numerical SO reference runs, is finite, and is symmetric in its two
    q-indices on the internal-joint block (a sanity property of d2tau/dq2)."""
    ad = _go2("floating"); nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(1)
    q, qd, qdd, _ = _rand_state(rng, nq, nv)
    internal = list(range(6, nv))
    T = second_order_id_reference(ad, q, qd, qdd, internal, L)
    assert np.isfinite(T).all()
    # d2tau/dq_j dq_k symmetric across the internal block: T[:, j, k] == reference[:, k, j]
    full = second_order_id_reference(ad, q, qd, qdd, list(range(nv)), L)
    for a, j in enumerate(internal):
        for b, k in enumerate(internal):
            assert np.abs(full[:, j, internal[b]] - full[:, k, internal[a]]).max() < 1e-3


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


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_second_order_fd_tensors_match_fd():
    """All four fdsva_so tensors transformed pin->mjx match FD of the validated
    analytic first-order mjx forward-dynamics gradients along the mjx retract
    (q/qvel/applied-force perturbations)."""
    ad = _go2("floating"); ref = ad.reference; nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, _, u = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L)
    v_mjx = mc.v_pin_to_mjx(qd, R, L); uf_mjx = mc.v_pin_to_mjx(u, R, L)

    def grads(qq, vm, um):
        RR = mc.base_rotation(qq, L)
        vp = mc.v_mjx_to_pin(vm, RR, L); up = mc.force_mjx_to_pin(um, RR, L)
        Minv = ad.minv(qq); qdd = np.asarray(ad.forward_dynamics(qq, vp, up))
        dq, dqd = ad.forward_dynamics_gradient(qq, vp, up)
        return mc.fd_gradient_pin_to_mjx(dq, dqd, Minv, qdd, vp, up, RR, L)

    h = 1e-6
    ref_d2q = np.zeros((nv, nv, nv)); ref_cross = np.zeros((nv, nv, nv))
    ref_d2qd = np.zeros((nv, nv, nv)); ref_d2tdq = np.zeros((nv, nv, nv))
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        gp = grads(mc.mjx_retract(q, e, ref, L), v_mjx, uf_mjx)
        gm = grads(mc.mjx_retract(q, -e, ref, L), v_mjx, uf_mjx)
        ref_d2q[:, :, k] = (gp[0] - gm[0]) / (2 * h); ref_cross[:, :, k] = (gp[1] - gm[1]) / (2 * h)
        gp = grads(q, v_mjx + e, uf_mjx); gm = grads(q, v_mjx - e, uf_mjx)
        ref_d2qd[:, :, k] = (gp[1] - gm[1]) / (2 * h)
        gp = grads(q, v_mjx, uf_mjx + e); gm = grads(q, v_mjx, uf_mjx - e)
        ref_d2tdq[:, :, k] = (gp[0] - gm[0]) / (2 * h)
    so = tuple(np.asarray(t) for t in ref.fdsva_so(q, qd, u))
    Minv = ad.minv(q); qdd_val = np.asarray(ad.forward_dynamics(q, qd, u))
    ddq, ddqd = ad.forward_dynamics_gradient(q, qd, u)
    d2q, cross, d2qd, d2tdq = mc.second_order_fd_pin_to_mjx(
        so, ddq, ddqd, Minv, qdd_val, qd, u, R, L)
    assert np.abs(d2q - ref_d2q).max() < 1e-5
    assert np.abs(cross - ref_cross).max() < 1e-5
    assert np.abs(d2qd - ref_d2qd).max() < 1e-5
    assert np.abs(d2tdq - ref_d2tdq).max() < 1e-5


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_second_order_fd_fixed_base_is_noop():
    ad = _go2("fixed"); nv = ad.nv; L = mc.FIXED_BASE
    rng = np.random.default_rng(2)
    q, qd, _, u = _rand_state(rng, ad.nq, nv)
    so = tuple(np.asarray(t) for t in ad.reference.fdsva_so(q, qd, u))
    Minv = ad.minv(q); qdd_val = np.asarray(ad.forward_dynamics(q, qd, u))
    ddq, ddqd = ad.forward_dynamics_gradient(q, qd, u)
    out = mc.second_order_fd_pin_to_mjx(so, ddq, ddqd, Minv, qdd_val, qd, u, np.eye(3), L)
    for o, s in zip(out, so):
        assert np.array_equal(o, s)


# ---------------------------------------------------------------------------
# Integrator gradient (discrete state-transition Jacobian dAB = [A | B])
# ---------------------------------------------------------------------------

def _quat_log_rel(ref, qref_xyzw, qc_xyzw):
    """omega s.t. _spherical_retract(qref, omega) ~= qc; = 2*log(qref^{-1} (x) qc)."""
    qref = np.asarray(qref_xyzw, float); qc = np.asarray(qc_xyzw, float)
    qref_inv = np.array([-qref[0], -qref[1], -qref[2], qref[3]])
    dq = ref._quat_mul_xyzw(qref_inv, qc)
    v = dq[:3]; w = dq[3]; nrm = np.linalg.norm(v)
    if nrm < 1e-14:
        return 2.0 * v
    return (2.0 * np.arctan2(nrm, w)) * v / nrm


def _mjx_integrator_value(ad, q_mjx, v_mjx, u_mjx, dt, it, L):
    """Run the integrator in the mjx convention. Returns (q_kp1_full(nq),
    qd_kp1_mjx(nv)). Mirrors the _integrator.py mjx codegen: mjx inputs -> pin,
    pin integrator, then mjx GLOBAL base-position step + qd output reframe by G."""
    ref = ad.reference; nq = ad.nq
    q_pin = mc.q_mjx_to_pin(q_mjx, L); R = mc.base_rotation(q_pin, L)
    v_pin = mc.v_mjx_to_pin(v_mjx, R, L); u_pin = mc.force_mjx_to_pin(u_mjx, R, L)
    x_kp1 = np.asarray(ref.integrator(q_pin, v_pin, u_pin, dt, integrator_type=it), float)
    q_kp1 = x_kp1[:nq].copy(); qd_kp1_mjx = mc.v_pin_to_mjx(x_kp1[nq:], R, L)
    w_lin = v_mjx[L.lin_slice] if it == "euler" else qd_kp1_mjx[L.lin_slice]
    q_kp1[L.pos_slice] = q_mjx[L.pos_slice] + dt * w_lin   # mjx GLOBAL add
    return q_kp1, qd_kp1_mjx


def _fd_mjx_dAB(ad, q_mjx, v_mjx, u_mjx, dt, it, L, h=1e-6):
    ref = ad.reference; nq, nv = ad.nq, ad.nv
    q_ref = _mjx_integrator_value(ad, q_mjx, v_mjx, u_mjx, dt, it, L)[0][:nq]

    def out_tangent(qc_full, qd_mjx_out):
        xi = np.zeros(nv)
        xi[6:] = qc_full[7:] - q_ref[7:]                              # revolute joints
        xi[L.lin_slice] = qc_full[L.pos_slice] - q_ref[L.pos_slice]   # base pos (global)
        xi[L.ang_slice] = _quat_log_rel(ref, q_ref[3:7], qc_full[3:7])
        return np.concatenate([xi, qd_mjx_out])

    cols = []
    for k in range(nv):                                              # dq_mjx
        e = np.zeros(nv); e[k] = h
        qp = mc.q_pin_to_mjx(mc.mjx_retract(mc.q_mjx_to_pin(q_mjx, L), e, ref, L), L)
        qm = mc.q_pin_to_mjx(mc.mjx_retract(mc.q_mjx_to_pin(q_mjx, L), -e, ref, L), L)
        tp = out_tangent(*_mjx_integrator_value(ad, qp, v_mjx, u_mjx, dt, it, L))
        tm = out_tangent(*_mjx_integrator_value(ad, qm, v_mjx, u_mjx, dt, it, L))
        cols.append((tp - tm) / (2 * h))
    for k in range(nv):                                              # dqd_mjx
        e = np.zeros(nv); e[k] = h
        tp = out_tangent(*_mjx_integrator_value(ad, q_mjx, v_mjx + e, u_mjx, dt, it, L))
        tm = out_tangent(*_mjx_integrator_value(ad, q_mjx, v_mjx - e, u_mjx, dt, it, L))
        cols.append((tp - tm) / (2 * h))
    for k in range(nv):                                              # du_mjx
        e = np.zeros(nv); e[k] = h
        tp = out_tangent(*_mjx_integrator_value(ad, q_mjx, v_mjx, u_mjx + e, dt, it, L))
        tm = out_tangent(*_mjx_integrator_value(ad, q_mjx, v_mjx, u_mjx - e, dt, it, L))
        cols.append((tp - tm) / (2 * h))
    return np.array(cols).T


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
@pytest.mark.parametrize("it", ["euler", "si_euler"])
def test_integrator_gradient_matches_fd_of_value(it):
    """The transformed mjx integrator gradient dAB = [A|B] matches FD of the mjx
    integrator VALUE along the mjx retract (q/qd/u perturbations, output q-tangent
    coordinate-differenced via the mjx retract at the next state)."""
    ad = _go2("floating"); nq, nv = ad.nq, ad.nv; L = mc.FloatingRootLayout()
    rng = np.random.default_rng(7)
    q, qd, _, u = _rand_state(rng, nq, nv)
    R = mc.base_rotation(q, L); dt = 0.01
    q_mjx = mc.q_pin_to_mjx(q, L); v_mjx = mc.v_pin_to_mjx(qd, R, L); u_mjx = mc.v_pin_to_mjx(u, R, L)
    fd = _fd_mjx_dAB(ad, q_mjx, v_mjx, u_mjx, dt, it, L)
    dAB_pin = np.asarray(ad.reference.integrator_gradient(q, qd, u, dt, it), float)
    Minv = ad.minv(q); qdd = np.asarray(ad.forward_dynamics(q, qd, u))
    an = mc.integrator_gradient_pin_to_mjx(dAB_pin, Minv, qdd, qd, u, R, dt, it, L)
    assert np.abs(an - fd).max() < 1e-5


@pytest.mark.skipif(not _HAVE_DEPS, reason="needs robot_descriptions")
def test_integrator_gradient_fixed_base_is_noop():
    ad = _go2("fixed"); nv = ad.nv; L = mc.FIXED_BASE
    rng = np.random.default_rng(2)
    q, qd, _, u = _rand_state(rng, ad.nq, nv); dt = 0.01
    Minv = ad.minv(q); qdd = np.asarray(ad.forward_dynamics(q, qd, u))
    for it in ("euler", "si_euler"):
        dAB = np.asarray(ad.reference.integrator_gradient(q, qd, u, dt, it), float)
        out = mc.integrator_gradient_pin_to_mjx(dAB, Minv, qdd, qd, u, np.eye(3), dt, it, L)
        assert np.array_equal(out, dAB)


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

    # nonlinear effects (bias force) vs mj_forward qfrc_bias (the accel-couple revision)
    d.qpos[:] = q_mjx; d.qvel[:] = v_mjx; d.qacc[:] = 0; mujoco.mj_forward(m, d)
    nle_pin = np.asarray(ref.nonlinear_effects(q_pin, qd))
    if nle_pin.ndim > 1:
        nle_pin = nle_pin[0]
    nle_grid = mc.nonlinear_effects_pin_to_mjx(nle_pin, np.asarray(ref.crba(q_pin)), qd, R, L)
    assert np.abs(nle_grid - d.qfrc_bias).max() < 1e-9

    # frame jacobian (base-linear column reframe) vs mj_jacBody
    jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l1")
    mujoco.mj_jacBody(m, d, jacp, jacr, bid)
    J_grid = mc.jacobian_pin_to_mjx(np.asarray(ref.frame_jacobian(q_pin, "j1")), R, L)
    assert np.abs(J_grid - np.vstack([jacp, jacr])).max() < 1e-9
