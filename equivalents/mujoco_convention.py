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


def _real_or_complex(x):
    """np.asarray that PRESERVES a complex dtype (for complex-step differentiation)
    but promotes real input to float64."""
    x = np.asarray(x)
    return x if np.iscomplexobj(x) else x.astype(np.float64)


def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
                    dtype=np.result_type(np.asarray(v), np.float64))


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
    (dtype follows ``R`` so complex-step differentiation propagates through it.)
    """
    R = np.asarray(R)
    G = np.eye(nv, dtype=R.dtype if np.iscomplexobj(R) else np.float64)
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
    R = np.asarray(R)
    Gd = np.zeros((nv, nv), dtype=R.dtype if np.iscomplexobj(R) else np.float64)
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
# Row / column / congruence value transforms (the recurring shapes)
# ---------------------------------------------------------------------------
#
# Three families cover almost every floating output (see the convention map in
# ``docs/open-tasks/mjx_codegen_fusion_master_plan.md`` §2):
#   * base_rotate (covector OUT, row map ``G``)      -- gravity, id_regressor, ...
#   * column_reframe (Jacobian ``J G^{-1}``)         -- frame_jacobian, J_com, ccrba, ...
#   * congruence (``G X G^T``)                        -- mass matrix, Minv, coriolis_matrix
# All are NO-OP on a fixed base.


def base_rotate_pin_to_mjx(y, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Covector/contravector OUTPUT row map ``y_mjx = G y_pin`` -- rotate the
    base-LINEAR rows (0:3) by ``R``. Works on a vector (``generalized_gravity``,
    ``inverse_dynamics``) or a matrix whose ROWS are tangent-indexed
    (``inverse_dynamics_regressor`` ``Y``, ``forward_dynamics_parameter_gradient``):
    ``Y_mjx[0:3] = R Y_pin[0:3]``. (Same base-linear-row rotation as
    :func:`id_tau_pin_to_mjx`, generalised to extra trailing axes.)"""
    y = _real_or_complex(y).copy()
    if layout.floating:
        y[layout.lin_slice] = np.tensordot(R, y[layout.lin_slice], axes=(1, 0))
    return y


def jacobian_pin_to_mjx(J, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Jacobian column reframe ``J_mjx = J_pin G^{-1}`` -- right-multiply the
    base-LINEAR COLUMNS (0:3) by ``R^T``. The output rows are a frame-invariant
    geometric quantity (a spatial velocity / momentum / pose rate); only the
    *input* tangent basis changes (mjx base-linear velocity is global). Covers
    ``frame_jacobian``, ``frame_jacobian_dot``, ``jacobian_com``, the CCRBA matrix
    ``A`` (``h = A qd`` is invariant), ``cmm_time_variation`` and
    ``end_effector_pose_gradient``. Validated vs ``mj_jacBody`` to ~4e-16."""
    J = _real_or_complex(J).copy()
    if layout.floating:
        # columns 0:3 <- J[:, 0:3] @ R^T
        J[:, layout.lin_slice] = J[:, layout.lin_slice] @ R.T
    return J


def coriolis_matrix_pin_to_mjx(C, R, layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Coriolis matrix similarity ``C_mjx = G C_pin G^{-1} = G C_pin G^T`` (``G``
    orthogonal). ``C`` is NOT symmetric (so this is a similarity, not a symmetric
    congruence) but the code is identical to :func:`mass_matrix_pin_to_mjx`: rows
    0:3 rotate by ``R`` (covector output), columns 0:3 reframe by ``R^T`` (the
    ``qd`` it multiplies is reframed). ``C qd`` is then a covector matching
    :func:`id_tau_pin_to_mjx`. The ``qd`` INPUT must already be mjx->pin converted."""
    if not layout.floating:
        return _real_or_complex(C)
    nv = C.shape[0]
    G = g_matrix(R, nv, layout)
    return G @ _real_or_complex(C) @ G.T


def nonlinear_effects_pin_to_mjx(nle_pin, M, qd_pin, R,
                                 layout: FloatingRootLayout = FloatingRootLayout()) -> np.ndarray:
    """Bias force ``nle = ID(q, qd, qacc=0)`` pin->mjx (MuJoCo ``qfrc_bias``).

    REVISED (the ``omega x v`` trap): "qacc = 0" is NOT frame-invariant. MuJoCo's
    bias holds ``qacc_mjx = 0``, which in the pin frame is
    ``a_pin = accel_mjx_to_pin(0, v_pin) = (-omega x v_lin) on the base-linear block``
    (NOT zero). So the correct bias in the pin frame is ``ID(q, qd_pin, a_pin)``,
    i.e. the pin bias ``nle_pin = ID(q,qd,0)`` PLUS ``M . delta_a`` where
    ``delta_a`` carries that base-linear ``-omega x v`` term; then the covector
    base-rotate ``G .``. Naively treating ``nle`` as a plain covector ``G nle_pin``
    is O(1) wrong (1.68 abs error on go2); with the coupling it matches MuJoCo
    ``qfrc_bias`` to ~5e-16. Needs the mass matrix ``M`` and the pin-frame ``qd``."""
    nle = _real_or_complex(nle_pin).copy()
    if not layout.floating:
        return nle
    nv = nle.shape[0]
    qd_pin = _real_or_complex(qd_pin)
    v_lin = qd_pin[layout.lin_slice]
    omega = qd_pin[layout.ang_slice]
    delta_a = np.zeros(nv, dtype=nle.dtype)
    delta_a[layout.lin_slice] = -np.cross(omega, v_lin)
    nle_at_mjx_zero = nle + _real_or_complex(M) @ delta_a
    return id_tau_pin_to_mjx(nle_at_mjx_zero, R, layout)


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
    vec = np.asarray(vec)
    out = np.zeros((nv, nv), dtype=vec.dtype if np.iscomplexobj(vec) else np.float64)
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
    dtau_dq = _real_or_complex(dtau_dq)
    dtau_dqd = _real_or_complex(dtau_dqd)
    if not layout.floating:
        return dtau_dq.copy(), dtau_dqd.copy()
    nv = dtau_dq.shape[0]
    G = g_matrix(R, nv, layout)
    Ginv = G.T                                   # G^{-1}
    M = _real_or_complex(M)
    tau_pin = _real_or_complex(tau_pin)
    qd_pin = _real_or_complex(qd_pin); qdd_pin = _real_or_complex(qdd_pin)
    v_lin = qd_pin[layout.lin_slice]
    omega = qd_pin[layout.ang_slice]
    qdd_lin = qdd_pin[layout.lin_slice]
    cdt = np.result_type(G, M, tau_pin, qd_pin, qdd_pin)   # complex if any input is

    # --- output prefactor derivative: column ang+a = Gdot_a tau_pin ---
    pref = np.zeros((nv, nv), dtype=cdt)
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
    Ja_v = np.zeros((nv, nv), dtype=cdt)
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
# Centroidal-derivative (dccrba) and end-effector Hessian transforms
# ---------------------------------------------------------------------------


def dccrba_dh_dq_pin_to_mjx(dh_dq, A, qd_pin, R,
                            layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform the centroidal-momentum config-gradient ``dh/dq`` pin->mjx
    (the first of ``centroidal_dynamics_derivatives``). The momentum ``h = A qd``
    is INVARIANT (geometric), so holding ``qd`` fixed *in the mjx frame* and
    perturbing the base ORIENTATION rotates the pin-frame ``qd`` the kernel saw:

    ``dh_dq_mjx = dh_dq_pin G^{-1} + A_pin Jv_q``,  ``Jv_q = _cross_cols(v_lin, -1)``.

    REVISED (vel-couple): the column reframe ``dh_dq G^{-1}`` alone is O(1) wrong
    (off by ~24 on go2); the ``A Jv_q`` velocity-coupling term (nonzero only on the
    3 base-rotation columns) brings it to ~2e-8 vs FD of ``h`` along the mjx retract.
    Needs the CCRBA matrix ``A`` and the pin-frame ``qd``. Companion partials:
    ``dh/dqd = A G^{-1}`` (a plain :func:`jacobian_pin_to_mjx` column reframe, ``h``
    invariant in ``qd``); the ``hdot`` (momentum-rate) partials additionally carry
    the ``omega x v`` acceleration coupling (same family as the ID gradient).
    Fixed base: returned unchanged."""
    dh_dq = _real_or_complex(dh_dq).copy()
    if not layout.floating:
        return dh_dq
    nv = dh_dq.shape[1]
    A = _real_or_complex(A)
    qd_pin = _real_or_complex(qd_pin)
    G = g_matrix(R, nv, layout); Ginv = G.T
    Jv_q = _cross_cols(qd_pin[layout.lin_slice], -1.0, nv, layout)
    return dh_dq @ Ginv + A @ Jv_q


def ee_pose_hessian_pin_to_mjx(H_pin, dpose_pin, R,
                               layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform the end-effector pose Hessian ``H_pin[i,a,k] = d2 pose_i / dxi_a dxi_k``
    pin->mjx (the symmetric coordinate Hessian along the retract).

    The pose VALUE is invariant; its first derivative reframes columns by ``G^{-1}``
    (:func:`jacobian_pin_to_mjx`). The second derivative is a DOUBLE column reframe
    plus a frame-correction from the q-dependence of ``G^{-1}``:

        H_mjx[i,a,k] = sum_{b,c} H_pin[i,b,c] G^{-1}[b,a] G^{-1}[c,k]
                       + sym_{a,k}( dpose_pin . d(G^{-1})/dtheta_k )

    where ``d(G^{-1})/dtheta_k`` (k a base-rotation DOF) has base-linear block
    ``-[e_k]_x R^T``. The raw "differentiate the gradient" expression carries the
    correction on the ``k`` index only and is NOT symmetric; GRiD's analytic Hessian
    IS symmetric, so we SYMMETRIZE in the two tangent indices. Validated to the FD
    floor vs the symmetric coordinate Hessian of the (invariant) pose along the mjx
    retract (the column-reframe-only term is off by ~0.5). Needs the value gradient
    ``dpose_pin`` (``end_effector_pose_gradient``). Fixed base: returned unchanged."""
    H = _real_or_complex(H_pin).copy()
    if not layout.floating:
        return H
    nout, nv = H.shape[0], H.shape[1]
    dpose = _real_or_complex(dpose_pin)
    G = g_matrix(R, nv, layout); Ginv = G.T
    out = np.einsum('ibc,ba,ck->iak', H, Ginv, Ginv)
    term = np.zeros((nout, nv, nv), dtype=out.dtype)
    for a in range(3):
        k = layout.ang_start + a
        e_a = np.zeros(3); e_a[a] = 1.0
        dGinv = -skew(e_a) @ R.T                      # d(G^{-1})/dtheta_k, base-linear block
        term[:, layout.lin_slice, k] = dpose[:, layout.lin_slice] @ dGinv
    return out + 0.5 * (term + term.transpose(0, 2, 1))


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


# ---------------------------------------------------------------------------
# Second order (idsva_so / fdsva_so)
# ---------------------------------------------------------------------------
#
# idsva_so returns (d2tau/dq2, d2tau/dqd2, d2tau/dqd-dq, dM/dq). All four are
# transformed to the mjx frame here and validated to FD precision (and, on a
# matched model, vs FD-of-FD of mj_inverse).
#
# KEY SIMPLIFICATION (no retract-curvature term needed): the mjx second derivative
# is the SINGLE derivative of the *already-MuJoCo-correct* analytic FIRST-order mjx
# gradient. Differentiating an exact first-order quantity once more needs only the
# FIRST-order sensitivities of that gradient's inputs (the pin SO tensors contracted
# with the input-conversion Jacobians + the frame derivative) -- the second-order
# behaviour of the retract never enters, because we are not composing two retract
# steps, we are differentiating a function whose value already matches MuJoCo. So
# d2tau/dq2 = d/dxi [ id_gradient_pin_to_mjx(...) ], evaluated by COMPLEX-STEP through
# the validated first-order assembly (exact, and it sidesteps hand-expanding the
# messy acceleration-coupling derivative). dM/dq is a first-order quantity (gradient
# of M) with its own clean closed form. See `second_order_id_pin_to_mjx`.
#
# For the jax / torch surfaces the equivalent (and simplest) path is AUTODIFF THROUGH
# THE VALUE TRANSFORM -- differentiating the validated value map twice gives the mjx
# SO tensors for free; the complex-step here is the numpy-surface analogue.


def dM_dq_pin_to_mjx(dM_dq, M, R, layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform the mass-matrix gradient tensor ``dM/dq`` pin->mjx (one of the four
    ``idsva_so`` outputs; also drives ``fdsva_so``). Index convention
    ``dM_dq[i,l,k] = d M_il / d q_k`` (RBDReference / GRiD order). Needs the VALUE
    ``M``. Fixed base: returned unchanged. Validated to ~1e-8 vs FD of the mjx mass
    matrix along the mjx retract.
    """
    dM_dq = np.asarray(dM_dq, dtype=np.float64)
    if not layout.floating:
        return dM_dq.copy()
    nv = dM_dq.shape[0]
    M = np.asarray(M, dtype=np.float64)
    G = g_matrix(R, nv, layout)
    Ginv = G.T
    # transport: reframe the q-index (config tangent) by G^{-1}, congruence by G.
    tmp = np.einsum('ilm,mk->ilk', dM_dq, Ginv)
    out = np.einsum('ai,ijk,bj->abk', G, tmp, G)
    # frame terms on the base-rotation columns: d(G)/dxi_a M G^T + G M d(G^T)/dxi_a
    for a in range(3):
        k = layout.ang_start + a
        Gd = g_dot(R, a, nv, layout)
        out[:, :, k] += Gd @ M @ G.T + G @ M @ Gd.T
    return out


def dccrba_dA_dq_pin_to_mjx(dA_dq, A, R, layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform the CMM config-gradient TENSOR ``dA/dq`` pin->mjx (GRiD ``dccrba``).

    Index convention ``dA_dq[i,l,m] = d A_il / d q_m`` (RBDReference ``dccrba`` order
    ``dA_dq[:, k, i]``). The CMM ``A`` maps ``qd -> h`` and the centroidal momentum
    ``h`` lives in the world-aligned CoM frame, so the OUTPUT (momentum-row) index is
    INVARIANT -- only the velocity/config tangent reparameterization ``G`` acts.

    The value map is the plain column reframe ``A_mjx = A_pin G^{-1}``
    (:func:`jacobian_pin_to_mjx`, ``h`` invariant). Differentiating along the mjx
    retract gives a DOUBLE ``G^{-1}`` reframe (the qd-column index ``l`` AND the
    q-tangent index ``m``) plus a frame term from the q-dependence of ``G^{-1}`` on the
    3 base-rotation columns::

        out[i,a,k] = sum_{l,m} dA_dq[i,l,m] G^{-1}[l,a] G^{-1}[m,k]
                     + (k = ang_start+c)  A_pin @ (dG/dtheta_c)^T

    Needs the VALUE ``A`` (ccrba). Validated to ~1e-5 vs FD of the mjx CMM along the
    mjx retract; the column-reframe-only term is O(1) wrong on the base-rotation cols.
    Fixed base: returned unchanged."""
    dA_dq = np.asarray(dA_dq, dtype=np.float64)
    if not layout.floating:
        return dA_dq.copy()
    nv = dA_dq.shape[2]
    A = np.asarray(A, dtype=np.float64)
    G = g_matrix(R, nv, layout); Ginv = G.T
    # reframe the q-tangent index m, then the qd-column index l, both by G^{-1}.
    tmp = np.einsum('ilm,mk->ilk', dA_dq, Ginv)
    out = np.einsum('la,ilk->iak', Ginv, tmp)
    # frame term on the base-rotation columns: A d(G^{-1})/dtheta_c = A (dG/dtheta_c)^T.
    for c in range(3):
        k = layout.ang_start + c
        Gd = g_dot(R, c, nv, layout)
        out[:, :, k] += A @ Gd.T
    return out


def quadratic_tracking_cost_pin_to_mjx(grad_x, hess_x, R, block_start, nv,
                                       layout: FloatingRootLayout = FloatingRootLayout()):
    """Transform a Gauss-Newton tracking cost's (grad_x, hess_x) pin->mjx.

    Covers ``ee_pos_cost`` / ``com_cost`` (q-block, ``block_start=0``) and
    ``momentum_cost`` (qd-block, ``block_start=nq``). The tracked quantity (EE/CoM
    position, centroidal momentum) is geometrically INVARIANT, so the cost VALUE is
    unchanged. The single nonzero tangent block of size ``nv`` (the gradient is
    ``Jᵀ(W·r)``, the GN hessian ``JᵀWJ``, with ``J`` the value-Jacobian) reframes by
    the value-Jacobian's column map ``J_mjx = J_pin G^{-1}``:

        grad_mjx[block] = G @ grad_pin[block]            (covector, G^{-T}=G)
        hess_mjx[block,block] = G @ hess_pin[block,block] @ G^T   (congruence)

    The GN hessian DROPS the value-curvature term, so -- unlike the true coordinate
    ``ee_pose_hessian`` -- there is NO frame-correction term here; the congruence is
    exact. ``block_start`` is the offset of the active tangent block in the
    ``nx = nq+nv`` state-gradient layout (0 = q-tangent in [:nv]; nq = qd-tangent).
    Fixed base: returned unchanged."""
    grad_x = np.asarray(grad_x, dtype=np.float64).copy()
    hess_x = np.asarray(hess_x, dtype=np.float64).copy()
    if not layout.floating:
        return grad_x, hess_x
    G = g_matrix(R, nv, layout)
    sl = slice(block_start, block_start + nv)
    grad_x[sl] = G @ grad_x[sl]
    hess_x[sl, sl] = G @ hess_x[sl, sl] @ G.T
    return grad_x, hess_x


def _id_input_xi_jacobians(qd_pin, qdd_pin, R, nv, layout):
    """The base-point first derivatives of the mjx->pin INPUT conversions wrt a
    q-perturbation xi (Jv_q = d qd_pin/dxi, Ja_q = d qdd_pin/dxi). Mirrors the
    couplings inside :func:`id_gradient_pin_to_mjx`."""
    v_lin = np.asarray(qd_pin, dtype=np.float64)[layout.lin_slice]
    omega = np.asarray(qd_pin, dtype=np.float64)[layout.ang_slice]
    qdd_lin = np.asarray(qdd_pin, dtype=np.float64)[layout.lin_slice]
    Jv_q = _cross_cols(v_lin, -1.0, nv, layout)
    Ja_q = _cross_cols(qdd_lin, -1.0, nv, layout)
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        Ja_q[layout.lin_slice, layout.ang_start + a] += (
            -np.cross(e_a, np.cross(omega, v_lin)) + np.cross(omega, np.cross(e_a, v_lin)))
    return Jv_q, Ja_q


def d2tau_dq2_pin_to_mjx(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq,
                         dtau_dq, dtau_dqd, M, tau_pin, qd_pin, qdd_pin, R,
                         layout: FloatingRootLayout = FloatingRootLayout(), h=1e-30):
    """MuJoCo-parity second derivative ``d2tau/dq2`` (the headline idsva_so tensor),
    holding qvel/qacc fixed in the mjx frame. Index convention
    ``d2tau_dq[i,j,k] = d2 tau_i / dq_j dq_k`` matching ``RBDReference.idsva_so``
    ``(di2_dq, di2_dqd, di2_dvdq, dm_dq)`` (so pass ``d2tau_cross = di2_dvdq`` with
    ``[i, qd, q]`` order). Needs the first-order gradients + ``M`` + the value
    ``tau_pin`` + the pin-frame inputs.

    Implemented by COMPLEX-STEP differentiating the validated analytic first-order
    transform (:func:`id_gradient_pin_to_mjx`) along the mjx perturbation: the SO
    tensor is a single derivative of the (MuJoCo-matched) first-order gradient, so it
    needs only the first-order sensitivities of that gradient's inputs (the pin SO
    tensors contracted with the input-conversion Jacobians + the frame derivative) --
    NOT a retract-curvature term. Validated to FD precision vs the mjx d/dq gradient.
    Fixed base: returned unchanged.
    """
    d2tau_dq = np.asarray(d2tau_dq, dtype=np.float64)
    if not layout.floating:
        return d2tau_dq.copy()
    nv = d2tau_dq.shape[0]
    d2tau_dqd = np.asarray(d2tau_dqd, dtype=np.float64)
    d2tau_cross = np.asarray(d2tau_cross, dtype=np.float64)
    dM_dq = np.asarray(dM_dq, dtype=np.float64)
    dtau_dq = np.asarray(dtau_dq, dtype=np.float64)
    dtau_dqd = np.asarray(dtau_dqd, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    tau_pin = np.asarray(tau_pin, dtype=np.float64)
    qd_pin = np.asarray(qd_pin, dtype=np.float64)
    qdd_pin = np.asarray(qdd_pin, dtype=np.float64)
    G = g_matrix(R, nv, layout); Jq = G.T
    Jv_q, Ja_q = _id_input_xi_jacobians(qd_pin, qdd_pin, R, nv, layout)

    out = np.zeros((nv, nv, nv))
    for k in range(nv):
        jqk, jvk, jak = Jq[:, k], Jv_q[:, k], Ja_q[:, k]
        # first-order sensitivities of id_gradient's inputs along xi_k:
        d_dtau_dq = (np.einsum('ijm,m->ij', d2tau_dq, jqk)
                     + np.einsum('inj,n->ij', d2tau_cross, jvk)
                     + np.einsum('ilj,l->ij', dM_dq, jak))
        d_dtau_dqd = (np.einsum('ijm,m->ij', d2tau_cross, jqk)
                      + np.einsum('ijn,n->ij', d2tau_dqd, jvk))
        d_M = np.einsum('ilm,m->il', dM_dq, jqk)
        d_tau = dtau_dq @ jqk + dtau_dqd @ jvk + M @ jak
        d_R = np.zeros((3, 3))
        if layout.lin_start <= 0 and layout.ang_start <= k < layout.ang_start + 3:
            e_a = np.zeros(3); e_a[k - layout.ang_start] = 1.0
            d_R = R @ skew(e_a)
        # complex-step: imag(first_order(inputs + i h dinputs))/h = d(first_order)/dxi_k
        g = id_gradient_pin_to_mjx(
            dtau_dq + 1j * h * d_dtau_dq, dtau_dqd + 1j * h * d_dtau_dqd,
            M + 1j * h * d_M, tau_pin + 1j * h * d_tau,
            qd_pin + 1j * h * jvk, qdd_pin + 1j * h * jak,
            R + 1j * h * d_R, layout)[0]
        out[:, :, k] = np.imag(g) / h
    return out


def _id_qd_xi_jacobians(qd_pin, R, nv, layout):
    """Base-point first derivatives of the input conversions wrt a qvel (v_mjx)
    perturbation: Jvv = d qd_pin/d v_mjx = G^{-1}; Ja_v = d qdd_pin/d v_mjx (the
    acceleration's velocity dependence). Mirrors :func:`id_gradient_pin_to_mjx`."""
    v_lin = np.asarray(qd_pin, dtype=np.float64)[layout.lin_slice]
    omega = np.asarray(qd_pin, dtype=np.float64)[layout.ang_slice]
    Jvv = g_matrix(R, nv, layout).T
    Ja_v = np.zeros((nv, nv))
    for a in range(3):
        e_a = np.zeros(3); e_a[a] = 1.0
        Ja_v[layout.lin_slice, layout.lin_start + a] = -np.cross(omega, R.T @ e_a)
        Ja_v[layout.lin_slice, layout.ang_start + a] = -np.cross(e_a, v_lin)
    return Jvv, Ja_v


def second_order_id_pin_to_mjx(so_tensors, dtau_dq, dtau_dqd, M, tau_pin,
                               qd_pin, qdd_pin, R,
                               layout: FloatingRootLayout = FloatingRootLayout(), h=1e-30):
    """Transform ALL FOUR idsva_so tensors pin->mjx for MuJoCo-native parity.

    ``so_tensors`` is the RBDReference/GRiD tuple ``(d2tau_dq, d2tau_dqd, d2tau_cross,
    dM_dq)`` with ``d2tau_cross = di2_dvdq`` (``[i, qd, q]``). Returns the same
    4-tuple in the mjx frame. The three genuine second derivatives are computed by
    complex-step differentiating the validated first-order transform (no
    retract-curvature term needed -- it is a single derivative of the MuJoCo-matched
    first-order gradient); ``dM/dq`` uses its clean closed form. Validated to FD
    precision vs FD of the mjx first-order gradients and, on a matched model, vs
    FD-of-FD of ``mj_inverse``. Fixed base: returned unchanged.
    """
    d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq = (np.asarray(t, dtype=np.float64) for t in so_tensors)
    if not layout.floating:
        return d2tau_dq.copy(), d2tau_dqd.copy(), d2tau_cross.copy(), dM_dq.copy()
    nv = d2tau_dq.shape[0]
    dtau_dq = np.asarray(dtau_dq, dtype=np.float64); dtau_dqd = np.asarray(dtau_dqd, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64); tau_pin = np.asarray(tau_pin, dtype=np.float64)
    qd_pin = np.asarray(qd_pin, dtype=np.float64); qdd_pin = np.asarray(qdd_pin, dtype=np.float64)
    G = g_matrix(R, nv, layout); Jq = G.T
    Jv_q, Ja_q = _id_input_xi_jacobians(qd_pin, qdd_pin, R, nv, layout)
    Jvv, Ja_v = _id_qd_xi_jacobians(qd_pin, R, nv, layout)

    d2q = np.zeros((nv, nv, nv)); cross = np.zeros((nv, nv, nv)); d2qd = np.zeros((nv, nv, nv))
    for k in range(nv):
        # --- q-perturbation (for d2tau/dq2 = d(g0)/dq and cross = d(g1)/dq) ---
        jqk, jvk, jak = Jq[:, k], Jv_q[:, k], Ja_q[:, k]
        d_dtau_dq = (np.einsum('ijm,m->ij', d2tau_dq, jqk)
                     + np.einsum('inj,n->ij', d2tau_cross, jvk)
                     + np.einsum('ilj,l->ij', dM_dq, jak))
        d_dtau_dqd = (np.einsum('ijm,m->ij', d2tau_cross, jqk)
                      + np.einsum('ijn,n->ij', d2tau_dqd, jvk))
        d_M = np.einsum('ilm,m->il', dM_dq, jqk)
        d_tau = dtau_dq @ jqk + dtau_dqd @ jvk + M @ jak
        d_R = np.zeros((3, 3))
        if layout.ang_start <= k < layout.ang_start + 3:
            e_a = np.zeros(3); e_a[k - layout.ang_start] = 1.0; d_R = R @ skew(e_a)
        g0, g1 = id_gradient_pin_to_mjx(
            dtau_dq + 1j * h * d_dtau_dq, dtau_dqd + 1j * h * d_dtau_dqd,
            M + 1j * h * d_M, tau_pin + 1j * h * d_tau,
            qd_pin + 1j * h * jvk, qdd_pin + 1j * h * jak, R + 1j * h * d_R, layout)
        d2q[:, :, k] = np.imag(g0) / h
        cross[:, :, k] = np.imag(g1) / h
        # --- qvel-perturbation (for d2tau/dqd2 = d(g1)/dqd) ---
        jvvk, javk = Jvv[:, k], Ja_v[:, k]
        dv_dtau_dq = (np.einsum('inj,n->ij', d2tau_cross, jvvk)
                      + np.einsum('ilj,l->ij', dM_dq, javk))
        dv_dtau_dqd = np.einsum('ijn,n->ij', d2tau_dqd, jvvk)
        dv_tau = dtau_dqd @ jvvk + M @ javk
        g1v = id_gradient_pin_to_mjx(
            dtau_dq + 1j * h * dv_dtau_dq, dtau_dqd + 1j * h * dv_dtau_dqd,
            M.astype(complex), tau_pin + 1j * h * dv_tau,
            qd_pin + 1j * h * jvvk, qdd_pin + 1j * h * javk, R.astype(complex), layout)[1]
        d2qd[:, :, k] = np.imag(g1v) / h
    dM_mjx = dM_dq_pin_to_mjx(dM_dq, M, R, layout)
    return d2q, d2qd, cross, dM_mjx

