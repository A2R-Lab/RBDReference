"""MuJoCo / mjx output-convention transforms for the floating base.

GRiD (and RBDReference / pinocchio) express a free-floating base in the
**pinocchio** convention:

* configuration quaternion ordered **xyzw** (scalar last), and
* the free-joint velocity is the spatial body twist ``[v_lin LOCAL ; omega LOCAL]``.

**MuJoCo / mjx** differ in exactly two ways for the free joint:

1. ``qpos`` quaternion is ordered **wxyz** (scalar first); and
2. ``qvel`` is ``[v_lin GLOBAL ; omega LOCAL]`` — only the *linear* root block is
   expressed in the world frame, the angular block and every internal joint match.

The velocity-frame difference is a single root-block basis change

    G(q) = blockdiag(R, I_3)  on the leading 6 tangent DOF, then I on the rest,

where ``R = R(q)`` is the base orientation read from the free-flyer quaternion.
Because ``R`` is orthogonal, ``G`` is orthogonal, so ``G^{-1} = G^T`` and
``G^{-T} = G`` exactly. Concretely ``G`` is the identity except its top-left 3x3
block (the base-linear tangent DOF), which equals ``R``.

This module is the single source of truth for the pin<->mjx transforms, in pure
numpy, so the GRiD bindings can mirror it and we have a validated reference for
later. Everything here is a **no-op for a fixed base** (no floating root => G=I,
no quaternion to reorder).

Layout assumption (matches RBDReference / GRiD floating robots): the floating
root is joint 0, occupying ``q[0:7] = [pos(3), quat_xyzw(4)]`` and tangent
``v[0:6] = [v_lin(3), omega(3)]``. The default :class:`FloatingRootLayout`
encodes this; pass a custom one only if a model places the root elsewhere.

References: ``docs/open-tasks/mjx_output_convention_flag.md`` (value transforms +
Phase-B gradient design) and ``floating_base_convention_study.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Root layout + small primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FloatingRootLayout:
    """Index layout of the free-flyer root within ``q`` and the tangent ``v``.

    ``floating=False`` marks a fixed base, for which every transform in this
    module is the identity (the mjx flag is a guaranteed no-op).
    """

    floating: bool = True
    pos_start: int = 0          # q index of the base position (3 entries)
    quat_start: int = 3         # q index of the base quaternion (4 entries)
    lin_start: int = 0          # tangent index of base linear velocity (3)
    ang_start: int = 3          # tangent index of base angular velocity (3)

    @property
    def pos_slice(self) -> slice:
        return slice(self.pos_start, self.pos_start + 3)

    @property
    def quat_slice(self) -> slice:
        return slice(self.quat_start, self.quat_start + 4)

    @property
    def lin_slice(self) -> slice:
        return slice(self.lin_start, self.lin_start + 3)

    @property
    def ang_slice(self) -> slice:
        return slice(self.ang_start, self.ang_start + 3)


FIXED_BASE = FloatingRootLayout(floating=False)


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_from_quat_xyzw(quat_xyzw) -> np.ndarray:
    """3x3 rotation from a unit xyzw quaternion (matches
    ``RBDReference._rotation_from_quat_xyzw``)."""
    x, y, z, w = (float(c) for c in quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def base_rotation(q, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Base orientation ``R(q)`` from the (xyzw, pin-layout) configuration."""
    if not layout.floating:
        return np.eye(3)
    return rotation_from_quat_xyzw(np.asarray(q, dtype=np.float64)[layout.quat_slice])


# ---------------------------------------------------------------------------
# G(q) and its tangent derivative
# ---------------------------------------------------------------------------

def g_matrix(R, nv: int, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """``G = blockdiag(R, I_3, I_{nv-6})`` — identity except the base-linear 3x3
    block, which is ``R``. Maps a pin tangent to the mjx tangent: ``v_mjx = G v_pin``.
    """
    G = np.eye(nv)
    if layout.floating:
        G[layout.lin_slice, layout.lin_slice] = R
    return G


def g_dot(R, a: int, nv: int, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """``dot G_a = dG/dxi_{ang_start + a}`` for ``a in {0,1,2}`` — the derivative
    of ``G`` along the ``a``-th base-rotation tangent DOF.

    With the LOCAL (right) retract ``R(q (+) xi) = R exp([phi]_x)``,
    ``dR/dxi_{ang+a} = R [e_a]_x``, so ``dot G_a`` is zero except the base-linear
    block, which equals ``R [e_a]_x``. (Validated by finite difference.)
    """
    Gd = np.zeros((nv, nv))
    if layout.floating:
        e_a = np.zeros(3)
        e_a[a] = 1.0
        Gd[layout.lin_slice, layout.lin_slice] = R @ skew(e_a)
    return Gd


# ---------------------------------------------------------------------------
# Configuration / input reorder (quaternion order + velocity frame)
# ---------------------------------------------------------------------------

def _reorder_quat_in_place(q, layout, to_mjx: bool):
    q = np.asarray(q, dtype=np.float64).copy()
    if not layout.floating:
        return q
    quat = q[layout.quat_slice]
    if to_mjx:                       # xyzw -> wxyz
        q[layout.quat_slice] = np.array([quat[3], quat[0], quat[1], quat[2]])
    else:                            # wxyz -> xyzw
        q[layout.quat_slice] = np.array([quat[1], quat[2], quat[3], quat[0]])
    return q


def q_pin_to_mjx(q, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Configuration relabel pin->mjx: free-flyer quaternion xyzw -> wxyz."""
    return _reorder_quat_in_place(q, layout, to_mjx=True)


def q_mjx_to_pin(q, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Configuration relabel mjx->pin: free-flyer quaternion wxyz -> xyzw."""
    return _reorder_quat_in_place(q, layout, to_mjx=False)


def v_mjx_to_pin(v_mjx, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Velocity (qd) input mjx->pin: ``v_pin = G^{-1} v_mjx`` (rotate base-linear
    block by ``R^T``; everything else unchanged)."""
    v = np.asarray(v_mjx, dtype=np.float64).copy()
    if layout.floating:
        v[layout.lin_slice] = R.T @ v[layout.lin_slice]
    return v


def v_pin_to_mjx(v_pin, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Velocity (qd) pin->mjx: ``v_mjx = G v_pin`` (rotate base-linear block by ``R``)."""
    v = np.asarray(v_pin, dtype=np.float64).copy()
    if layout.floating:
        v[layout.lin_slice] = R @ v[layout.lin_slice]
    return v


def force_mjx_to_pin(u_mjx, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Generalized-force (u / tau) input mjx->pin: a covector transforms inversely
    to a velocity, but with ``G`` orthogonal ``G^{-T}=G`` so ``tau_mjx = G tau_pin``
    => ``tau_pin = G^T tau_mjx`` (rotate base-linear block by ``R^T``)."""
    return v_mjx_to_pin(u_mjx, R, layout)


# Acceleration is NOT a simple frame rotation. The mjx base-linear velocity is the
# GLOBAL position rate ``p_dot = R v_local``, so differentiating,
# ``a_mjx_lin = R (a_pin_lin + omega x v_local)`` -- a velocity-dependent term
# (``Gdot v``, with ``Gdot = blockdiag(R [omega]_x, 0)``). Equivalently the pin
# body-frame acceleration is ``a_pin_lin = R^T a_mjx_lin - omega x v_local``. This
# coupling is what makes the inverse-dynamics CORIOLIS term frame-dependent; the
# design doc (which treated accel like velocity) missed it. Validated vs MuJoCo
# ``mj_inverse`` to ~1e-12.

def accel_pin_to_mjx(a_pin, v_pin, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Acceleration (qdd) output pin->mjx: ``a_mjx = G a_pin + Gdot v_pin``;
    base-linear block ``R (a_pin_lin + omega x v_local)``."""
    a = np.asarray(a_pin, dtype=np.float64).copy()
    if layout.floating:
        v_pin = np.asarray(v_pin, dtype=np.float64)
        omega = v_pin[layout.ang_slice]
        v_lin = v_pin[layout.lin_slice]
        a[layout.lin_slice] = R @ (a[layout.lin_slice] + np.cross(omega, v_lin))
    return a


def accel_mjx_to_pin(a_mjx, v_pin, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Acceleration (qdd) input mjx->pin: ``a_pin_lin = R^T a_mjx_lin - omega x v_local``.
    ``v_pin`` is the pin-frame velocity (use :func:`v_mjx_to_pin` first)."""
    a = np.asarray(a_mjx, dtype=np.float64).copy()
    if layout.floating:
        v_pin = np.asarray(v_pin, dtype=np.float64)
        omega = v_pin[layout.ang_slice]
        v_lin = v_pin[layout.lin_slice]
        a[layout.lin_slice] = R.T @ a[layout.lin_slice] - np.cross(omega, v_lin)
    return a


# ---------------------------------------------------------------------------
# Value-output transforms pin->mjx
# ---------------------------------------------------------------------------

def mass_matrix_pin_to_mjx(M, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """``M_mjx = G^{-T} M_pin G^{-1} = G M_pin G^T`` (since ``G^{-T}=G``)."""
    if not layout.floating:
        return np.asarray(M, dtype=np.float64)
    nv = M.shape[0]
    G = g_matrix(R, nv, layout)
    return G @ np.asarray(M, dtype=np.float64) @ G.T


def minv_pin_to_mjx(Minv, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """``Minv_mjx = G Minv_pin G^T``."""
    if not layout.floating:
        return np.asarray(Minv, dtype=np.float64)
    nv = Minv.shape[0]
    G = g_matrix(R, nv, layout)
    return G @ np.asarray(Minv, dtype=np.float64) @ G.T


def id_tau_pin_to_mjx(tau, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """inverse-dynamics output (covector): ``tau_mjx = G^{-T} tau_pin = G tau_pin``."""
    tau = np.asarray(tau, dtype=np.float64).copy()
    if layout.floating:
        tau[layout.lin_slice] = R @ tau[layout.lin_slice]
    return tau


def fd_qdd_pin_to_mjx(qdd, qd_pin, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """forward-dynamics acceleration output pin->mjx. An acceleration is NOT a
    plain rotation: ``qdd_mjx = G qdd_pin + Gdot qd_pin`` (see :func:`accel_pin_to_mjx`).
    Needs the velocity ``qd_pin`` for the ``omega x v`` term."""
    return accel_pin_to_mjx(qdd, qd_pin, R, layout)


# ---------------------------------------------------------------------------
# First-order gradient transforms pin->mjx (Phase B1)
# ---------------------------------------------------------------------------
#
# A GRiD/pin first-order gradient ``dy_pin/dxi_pin`` is an ``nv x nv`` Jacobian
# whose columns are tangent-space derivatives along the *pin* retract. The
# matching mjx gradient ``dy_mjx/dxi_mjx`` differs in THREE ways:
#
#   1. OUTPUT transport: every column carries the value-map ``T`` of the output
#      (``G^{-T}`` for a covector like tau, ``G`` for a contravector like qdd).
#   2. q-DEPENDENCE correction: ``T = T(q)`` depends on the base orientation, so
#      ``d_k y_mjx = T d_k y_pin + (d_k T) y_pin``. The correction ``(d_k T) y_pin``
#      is nonzero ONLY on the 3 base-rotation columns (k in {ang, ang+1, ang+2})
#      and needs the *value* output.
#   3. COLUMN (input-tangent) factor: the mjx base-LINEAR velocity is GLOBAL while
#      pin's is LOCAL, so the q-/qd-perturbation tangents themselves differ by
#      ``xi_pin = G^{-1} xi_mjx``. Re-expressing columns into the mjx tangent is a
#      right-multiply by ``G^{-1}``. (This only remixes the 3 base-linear columns;
#      it leaves the base-rotation and internal columns -- hence the correction --
#      untouched.) The doc's FD validated along the *pin* retract and so folded
#      this factor out; for parity with a real mjx derivative it is required, and
#      is pinned by the finite-difference-along-the-mjx-retract check.
#
# Fixed base => G=I, every term collapses to the identity (the flag is a no-op).


def _transport_and_reframe(grad, out_map, Ginv, correction=None):
    """``(out_map @ grad + correction) @ Ginv`` -- the common shape of every
    first-order gradient transform (output transport + optional q-correction,
    then the input-tangent column reframe)."""
    transported = out_map @ np.asarray(grad, dtype=np.float64)
    if correction is not None:
        transported = transported + correction
    return transported @ Ginv


#
# IMPORTANT (validated against real MuJoCo, 2026-06-08): MuJoCo's ``d/dqpos``
# derivatives hold ``qvel`` (and ``qacc``/applied force) FIXED IN THE MJX FRAME.
# Because the mjx base-linear velocity is GLOBAL, perturbing the base ORIENTATION
# rotates the *pin-frame* velocity/accel inputs that GRiD's kernels actually
# consumed (``v_pin = G^{-1} v_mjx`` tracks ``R``). So the base-rotation columns
# (3,4,5) of the "wrt q" gradient pick up TWO extra coupling terms beyond the
# doc's frame term, built from the first-order ``d/dqd`` gradient and the mass
# matrix. (Confirmed: holding qvel-mjx vs body-velocity fixed changes ``dtau/dq``
# only on columns 3,4,5, by an O(1) amount.) The "wrt qd" gradient is unaffected
# (perturbing qd does not move ``R``) -- it is a clean transport + column reframe.
#
# These couplings need extra GRiD quantities at the boundary: the ``d/dqd``
# gradient (already returned alongside ``d/dq``), the mass matrix ``M`` / its
# inverse ``Minv``, and the (pin-frame) velocity & accel/force inputs.


# Implementation note: the mjx gradient is assembled MODULARLY as
#
#   dy_mjx/dxi  =  PrefDeriv(y_pin)  +  Tout @ ( dy_pin/dq @ Jq
#                                              + dy_pin/dqd @ Jv
#                                              + dy_pin/d(accel|force) @ Ja )
#
# where ``Tout`` is the output value-map (``G`` for both a covector tau and the
# acceleration-output rotation), ``PrefDeriv`` is the derivative of the output map
# wrt the base rotation, and ``Jq``/``Jv``/``Ja`` are the Jacobians of the
# (validated) mjx->pin INPUT conversions wrt the perturbed quantity. Each Jacobian
# is a small, independently-sensible ``nv x nv`` matrix. This keeps every coupling
# explicit and matches MuJoCo's analytic derivatives (qvel & qacc/force held fixed
# in the mjx frame) -- validated to FD precision against ``mj_inverse``/``mj_forward``.


def _cross_cols(vec, sign, nv, layout):
    """``nv x nv`` matrix whose base-ROTATION columns ``a`` are ``sign * (e_a x vec)``
    in the base-LINEAR rows (everything else zero). Building block for the
    velocity/accel input Jacobians wrt a base-rotation perturbation."""
    out = np.zeros((nv, nv))
    vec = np.asarray(vec, dtype=np.float64)
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        out[layout.lin_slice, layout.ang_start + a] = sign * np.cross(e_a, vec)
    return out


def id_gradient_pin_to_mjx(dtau_dq, dtau_dqd, M, tau_pin, qd_pin, qdd_pin, R,
                           layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform inverse-dynamics first-order gradients pin->mjx for MuJoCo-native
    parity (qvel & qacc held fixed in the mjx frame).

    ``tau`` is a covector (``tau_mjx = G tau_pin``). Returns
    ``(dtau_dq_mjx, dtau_dqd_mjx)`` matching ``mj_inverse``'s ``d/dq`` / ``d/dqvel``.
    Needs ``M``, ``tau_pin`` (value) and the pin-frame inputs ``qd_pin``/``qdd_pin``.
    Fixed base: returned unchanged.
    """
    dtau_dq = np.asarray(dtau_dq, dtype=np.float64)
    dtau_dqd = np.asarray(dtau_dqd, dtype=np.float64)
    if not layout.floating:
        return dtau_dq.copy(), dtau_dqd.copy()
    nv = dtau_dq.shape[0]
    G = g_matrix(R, nv, layout)
    Ginv = G.T                                   # G^{-1}
    M = np.asarray(M, dtype=np.float64)
    tau_pin = np.asarray(tau_pin, dtype=np.float64)
    v_lin = np.asarray(qd_pin, dtype=np.float64)[layout.lin_slice]
    omega = np.asarray(qd_pin, dtype=np.float64)[layout.ang_slice]
    qdd_lin = np.asarray(qdd_pin, dtype=np.float64)[layout.lin_slice]

    # --- output prefactor derivative: column ang+a = Gdot_a tau_pin ---
    pref = np.zeros((nv, nv))
    for a in range(3):
        pref[:, layout.ang_start + a] = g_dot(R, a, nv, layout) @ tau_pin

    # --- wrt q: Jq = G^{-1} (config tangent); Jv_q, Ja_q nonzero on ang cols ---
    Jq = Ginv
    Jv_q = _cross_cols(v_lin, -1.0, nv, layout)                 # d qd_pin/d(theta_a)
    Ja_q = _cross_cols(qdd_lin, -1.0, nv, layout)               # d qdd_pin/d(theta_a) part 1
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        Ja_q[layout.lin_slice, layout.ang_start + a] += (
            -np.cross(e_a, np.cross(omega, v_lin)) + np.cross(omega, np.cross(e_a, v_lin)))
    dtau_dq_mjx = pref + G @ (dtau_dq @ Jq + dtau_dqd @ Jv_q + M @ Ja_q)

    # --- wrt qd: Jvv = G^{-1}; Ja_v from accel's velocity dependence ---
    Jvv = Ginv
    Ja_v = np.zeros((nv, nv))
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        # base-linear qvel col: d qdd_pin/d v_lin = -omega x (R^T e_a)
        Ja_v[layout.lin_slice, layout.lin_start + a] = -np.cross(omega, R.T @ e_a)
        # base-angular qvel col: d qdd_pin/d omega = -(e_a x v_lin)
        Ja_v[layout.lin_slice, layout.ang_start + a] = -np.cross(e_a, v_lin)
    dtau_dqd_mjx = G @ (dtau_dqd @ Jvv + M @ Ja_v)
    return dtau_dq_mjx, dtau_dqd_mjx


def fd_gradient_pin_to_mjx(dqdd_dq, dqdd_dqd, Minv, qdd_pin, qd_pin, u_pin, R,
                           layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform forward-dynamics first-order gradients pin->mjx for MuJoCo-native
    parity (qvel & applied force held fixed in the mjx frame).

    ``qdd`` is an ACCELERATION output (so the output map is the accel transform,
    not a plain rotation -- it carries an ``omega x v`` term and its derivative).
    Returns ``(dqdd_dq_mjx, dqdd_dqd_mjx)`` matching ``mj_forward``'s
    ``d/dq`` / ``d/dqvel``. Needs ``Minv``, ``qdd_pin`` (value) and the pin-frame
    inputs ``qd_pin``/``u_pin``. Fixed base: returned unchanged.
    """
    dqdd_dq = np.asarray(dqdd_dq, dtype=np.float64)
    dqdd_dqd = np.asarray(dqdd_dqd, dtype=np.float64)
    if not layout.floating:
        return dqdd_dq.copy(), dqdd_dqd.copy()
    nv = dqdd_dq.shape[0]
    G = g_matrix(R, nv, layout)
    Ginv = G.T
    Minv = np.asarray(Minv, dtype=np.float64)
    qdd_pin = np.asarray(qdd_pin, dtype=np.float64)
    v_lin = np.asarray(qd_pin, dtype=np.float64)[layout.lin_slice]
    omega = np.asarray(qd_pin, dtype=np.float64)[layout.ang_slice]
    u_lin = np.asarray(u_pin, dtype=np.float64)[layout.lin_slice]
    qdd_lin = qdd_pin[layout.lin_slice]

    # The output map O(a_pin, v_pin, R)_lin = R (a_pin_lin + omega x v_lin); its
    # partials: dO/da_pin = G ; dO/dR|val = R[e_a]x(a_pin_lin + omega x v_lin) ;
    # dO/dv_pin . dv|lin = R(omega x dv_lin + dv_ang x v_lin).

    # inner pin-acceleration gradient wrt the q-perturbation (qvel/force fixed mjx):
    Jv_q = _cross_cols(v_lin, -1.0, nv, layout)        # d qd_pin/d theta_a
    Ju_q = _cross_cols(u_lin, -1.0, nv, layout)        # d u_pin/d theta_a
    dqddpin_dq = dqdd_dq @ Ginv + dqdd_dqd @ Jv_q + Minv @ Ju_q
    out_R = np.zeros((nv, nv)); out_vq = np.zeros((nv, nv))
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        out_R[layout.lin_slice, layout.ang_start + a] = (
            R @ np.cross(e_a, qdd_lin + np.cross(omega, v_lin)))        # dO/dR
        dv_lin = -np.cross(e_a, v_lin)                                  # Jv_q[:,ang+a]
        out_vq[layout.lin_slice, layout.ang_start + a] = R @ np.cross(omega, dv_lin)  # dO/dv
    dqdd_dq_mjx = G @ dqddpin_dq + out_R + out_vq

    # wrt qd: inner pin-accel gradient + output map's explicit v dependence ---
    dqddpin_dqd = dqdd_dqd @ Ginv
    out_vd = np.zeros((nv, nv))
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        # qvel-linear col: dv_pin = Ginv[:,lin_a] = [R^T e_a (lin); 0]
        out_vd[layout.lin_slice, layout.lin_start + a] = R @ np.cross(omega, R.T @ e_a)
        # qvel-angular col: dv_pin = [0; e_a] -> dv_ang x v_lin
        out_vd[layout.lin_slice, layout.ang_start + a] = R @ np.cross(e_a, v_lin)
    dqdd_dqd_mjx = G @ dqddpin_dqd + out_vd
    return dqdd_dq_mjx, dqdd_dqd_mjx


# ---------------------------------------------------------------------------
# Retracts (for finite-difference validation)
# ---------------------------------------------------------------------------

def mjx_retract(q, xi, ref, layout: FloatingRootLayout = FloatingRootLayout()):
    """MuJoCo free-joint retract: base position takes a GLOBAL-frame additive
    step (``pos += xi_lin``), the base quaternion an SO(3) exp, and every other
    joint matches the pin retract. ``ref`` supplies the SO(3)/joint retract
    primitives (an :class:`RBDReference` instance).

    This differs from pin's ``integrate`` only in the base-linear block (pin
    applies the SE(3) ``V(phi) R`` coupling; MuJoCo adds the global step
    directly), which is exactly the ``G`` basis change. Used to finite-difference
    in the mjx tangent so the analytic mjx gradients can be pinned empirically.
    """
    q = np.asarray(q, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    if not layout.floating:
        return ref.integrate(q, xi)
    # pin-integrate the NON-base part by zeroing the base tangent, then overwrite
    # the base block with the mjx (global-linear) update.
    xi_internal = xi.copy()
    xi_internal[layout.lin_slice] = 0.0
    xi_internal[layout.ang_slice] = 0.0
    q_new = ref.integrate(q, xi_internal)
    q_new = np.asarray(q_new, dtype=np.float64).copy()
    q_new[layout.pos_slice] = q[layout.pos_slice] + xi[layout.lin_slice]   # GLOBAL add
    q_new[layout.quat_slice] = ref._spherical_retract(q[layout.quat_slice], xi[layout.ang_slice])
    return q_new

