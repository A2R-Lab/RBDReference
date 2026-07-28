import numpy as np
import copy
import sympy as sp

from ._plant import _PlantMixin
from ._energy import _EnergyMixin
from ._centroidal import _CentroidalMixin
from ._regressor import _RegressorMixin

np.set_printoptions(precision=4, suppress=True, linewidth=100)

# Additive reference-oracle mixins (plant/cost/barrier, energy/gravity/Coriolis,
# CoM/centroidal, sysID regressor). Composed onto the existing single-class
# RBDReference; `_HelpersMixin`-style shared state (`self.robot`,
# `_normalize_*`, cross operators, EE pose) lives on the concrete class, so the
# mixins call into it via MRO. This anticipates the D.5 file split without
# performing it (see docs/open-tasks/rbdreference_split_plan.md).
class RBDReference(
    _PlantMixin,
    _EnergyMixin,
    _CentroidalMixin,
    _RegressorMixin,
):
    def __init__(self, robotObj, use_joint_dynamics=False):
        """Initialize RBDReference with a robot object.

        Parameters
        ----------
        robotObj : URDFparser
            An instance of the URDFparser class.
        use_joint_dynamics : bool, optional
            When True, the RNEA/ABA value path applies the joint-local
            ``<dynamics damping>`` / ``<dynamics friction>`` bias
            (tau += damping*qd + friction*sign(qd)). DEFAULT False to stay
            consistent with bare Pinocchio's `pin.rnea`/`pin.aba`, which IGNORE
            `model.damping`/`model.friction` in the value path — making this the
            authoritative-oracle-preserving default. The id/fd GRADIENTS follow
            this same flag: when True they add the damping diagonal
            (d(damping*qd)/dqd) to dc_dqd / qdd_dqd (friction's subgradient is 0).

        Returns
        -------
        None : None
            None
        """
        self.robot = robotObj # instance of Robot Object class created by URDFparser
        self.use_joint_dynamics = use_joint_dynamics
        self._spatial_xmat_derivative_func_cache = {}
        self._spatial_xmat_second_derivative_func_cache = {}

    def _normalize_q_input(self, q):
        return self.robot.normalize_floating_base_q_input(q)

    def _normalize_v_input(self, vec):
        return self.robot.normalize_floating_base_v_input(vec)

    def _denormalize_v_output(self, vec):
        return self.robot.denormalize_floating_base_v_output(vec)

    def _permute_matrix_prefix(self, matrix, permutation, axis):
        if permutation is None:
            return matrix
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        prefix = len(permutation)
        if axis == 0:
            matrix[:prefix, :] = matrix[permutation, :]
        else:
            matrix[:, :prefix] = matrix[:, permutation]
        return matrix

    def _denormalize_qv_matrix_output(self, matrix, row_space=None, col_space=None):
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        if row_space == "v":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_v_permutation_from_internal(),
                axis=0,
            )
        elif row_space == "q":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_q_output_permutation_from_internal(),
                axis=0,
            )
        if col_space == "v":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_v_permutation_from_internal(),
                axis=1,
            )
        elif col_space == "q":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_q_output_permutation_from_internal(),
                axis=1,
            )
        return matrix

    def _denormalize_reduced_q_matrix_output(self, matrix, row_space=None):
        return self._denormalize_qv_matrix_output(matrix, row_space=row_space, col_space=None)

    def _denormalize_inverse_dynamics_gradient_output(self, dc_dq, dc_dqd):
        return np.hstack(
            (
                self._denormalize_reduced_q_matrix_output(dc_dq, row_space="v"),
                self._denormalize_qv_matrix_output(dc_dqd, row_space="v", col_space="v"),
            )
        )

    @staticmethod
    def _normalize_xyzw_quaternion(quat):
        quat = np.asarray(quat, dtype=np.float64).copy()
        quat_norm = np.linalg.norm(quat)
        if quat_norm == 0.0:
            raise ValueError("Floating-base quaternion norm was zero during normalization.")
        return quat / quat_norm

    # ----- SO(3) / SE(3) helpers used by the time-integrator step -----
    # Quaternion convention: xyzw (matches Pinocchio's free-flyer joint).
    # Free-flyer velocity convention: Pinocchio order, v = [v_lin (3); omega (3)],
    # interpreted in the LOCAL/body frame. This matches the user-facing
    # convention everywhere else in the project (see test_forward_dynamics_*).

    @staticmethod
    def _quat_mul_xyzw(a, b):
        """Hamilton-product quaternion multiply: returns a ⊗ b in xyzw order."""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dtype=np.float64)

    @staticmethod
    def _quat_exp_from_half_omega(half_omega):
        """Quaternion exponential map (xyzw) from half-angle vector.

        For an angular velocity * dt = omega_dt, the SO(3) retract is
        `q_new = q ⊗ exp_quat(0.5 * omega_dt)`. This helper takes the
        already-halved 3-vector and returns its quaternion exponential.
        """
        half_omega = np.asarray(half_omega, dtype=np.float64)
        theta = float(np.linalg.norm(half_omega))
        if theta < 1e-12:
            # Small-angle Taylor: sin(theta)/theta ≈ 1 - theta^2/6
            sinc = 1.0 - theta * theta / 6.0
            cos_t = 1.0 - 0.5 * theta * theta
        else:
            sinc = np.sin(theta) / theta
            cos_t = np.cos(theta)
        vec = sinc * half_omega  # (xyz components)
        return np.array([vec[0], vec[1], vec[2], cos_t], dtype=np.float64)

    @staticmethod
    def _rotation_from_quat_xyzw(q):
        """Build a 3x3 rotation matrix from an xyzw quaternion."""
        x, y, z, w = q
        # Standard quaternion-to-rotation formula (assumes unit quaternion).
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
            [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
        ], dtype=np.float64)

    @staticmethod
    def _so3_skew(v):
        x, y, z = v
        return np.array([
            [0.0, -z,   y],
            [z,    0.0, -x],
            [-y,   x,    0.0],
        ], dtype=np.float64)

    @staticmethod
    def _so3_V_matrix(phi):
        """SE(3)-exp 'V' helper: p_delta = V(phi) @ rho where phi = omega*dt
        and rho = v_lin * dt. Same closed-form as Pinocchio's local-frame
        free-flyer exponential.
        """
        phi = np.asarray(phi, dtype=np.float64)
        theta = float(np.linalg.norm(phi))
        skew_phi = RBDReference._so3_skew(phi)
        if theta < 1e-8:
            # Small-angle Taylor:
            #   V ≈ I + 0.5*[phi]_x + (1/6)*[phi]_x^2
            return np.eye(3) + 0.5 * skew_phi + (1.0 / 6.0) * (skew_phi @ skew_phi)
        a = (1.0 - np.cos(theta)) / (theta * theta)
        b = (theta - np.sin(theta)) / (theta ** 3)
        return np.eye(3) + a * skew_phi + b * (skew_phi @ skew_phi)

    @staticmethod
    def _so3_right_jacobian(phi):
        """SO(3) right Jacobian J_r(phi). Used by SE(3) dIntegrate (free-flyer)."""
        phi = np.asarray(phi, dtype=np.float64)
        theta = float(np.linalg.norm(phi))
        skew_phi = RBDReference._so3_skew(phi)
        if theta < 1e-8:
            return np.eye(3) - 0.5 * skew_phi + (1.0 / 6.0) * (skew_phi @ skew_phi)
        a = (1.0 - np.cos(theta)) / (theta * theta)
        b = (theta - np.sin(theta)) / (theta ** 3)
        return np.eye(3) - a * skew_phi + b * (skew_phi @ skew_phi)

    @staticmethod
    def _se3_Q_block(rho, phi):
        """SE(3) right-Jacobian coupling block Q(rho, phi) used by
        `pin.dIntegrate(..., ARG1)` for free-flyer joints with body-frame
        velocity. Closed-form adapted from Sola, Deray & Atchuthan,
        'A micro Lie theory for state estimation in robotics' (2018) eq. (184),
        evaluated at (rho, -phi) to match Pinocchio's sign convention. The
        SO(3) part of phi appears at odd powers in some terms and even
        powers in others — straight substitution gives Pinocchio's form."""
        rho = np.asarray(rho, dtype=np.float64)
        phi_neg = -np.asarray(phi, dtype=np.float64)
        theta = float(np.linalg.norm(phi_neg))
        Px = RBDReference._so3_skew(phi_neg)
        Rx = RBDReference._so3_skew(rho)
        Px2 = Px @ Px
        Rx_Px = Rx @ Px
        Px_Rx = Px @ Rx
        Px_Rx_Px = Px @ Rx @ Px
        Px2_Rx = Px2 @ Rx
        Rx_Px2 = Rx @ Px2
        if theta < 1e-4:
            # Small-angle Taylor: closed-form formula loses precision below
            # this threshold because the c3 numerator becomes catastrophically
            # cancellation-prone (subtracts two near-equal O(theta^3) values).
            sola = (0.5 * Rx
                    + (1.0 / 6.0) * (Px_Rx + Rx_Px + Px_Rx_Px)
                    - (1.0 / 24.0) * (Px2_Rx + Rx_Px2 - 3.0 * Px_Rx_Px))
            return -sola
        c1 = (theta - np.sin(theta)) / (theta ** 3)
        c2 = (1.0 - 0.5 * theta * theta - np.cos(theta)) / (theta ** 4)
        # 4th-order coefficient. The sign here is negative: matching Barfoot's
        # SE(3) Q-block (2θ−3sinθ+θcosθ)/(2θ⁵) requires negating this grouping.
        # Verified against pin.dIntegrate(ARG1) to ~1e-14 across increment
        # magnitudes; the previous (positive) sign matched only for tiny v_dt
        # and diverged as O(|v_dt|) for larger increments.
        c3 = -0.5 * (c2 - 3.0 * (theta - np.sin(theta) - (theta ** 3) / 6.0) / (theta ** 5))
        sola = (0.5 * Rx
                + c1 * (Px_Rx + Rx_Px + Px_Rx_Px)
                - c2 * (Px2_Rx + Rx_Px2 - 3.0 * Px_Rx_Px)
                + c3 * (Px @ Rx_Px2 + Px2 @ Rx_Px))
        return -sola

    @staticmethod
    def _so3_exp(phi):
        """SO(3) exponential: returns the 3x3 rotation matrix R = exp([phi]_x)."""
        phi = np.asarray(phi, dtype=np.float64)
        theta = float(np.linalg.norm(phi))
        Px = RBDReference._so3_skew(phi)
        if theta < 1e-8:
            return np.eye(3) + Px + 0.5 * (Px @ Px)
        sin_t = np.sin(theta) / theta
        one_minus_cos = (1.0 - np.cos(theta)) / (theta * theta)
        return np.eye(3) + sin_t * Px + one_minus_cos * (Px @ Px)

    def _spherical_retract(self, quat_xyzw, omega_dt):
        """SO(3) retract of a spherical joint's unit quaternion:
        q_new = q ⊗ exp(½·omega_dt), renormalized. Reuses the floating-base
        quaternion primitives (`_quat_exp_from_half_omega`, `_quat_mul_xyzw`)
        so there is ONE quaternion-exp implementation. `omega_dt` is the
        body-frame angular increment (Pinocchio JointModelSpherical v ordering).
        """
        delta_quat = self._quat_exp_from_half_omega(0.5 * np.asarray(omega_dt))
        return self._normalize_xyzw_quaternion(
            self._quat_mul_xyzw(np.asarray(quat_xyzw, dtype=np.float64), delta_quat)
        )

    def _joint_retract_specs(self):
        """Yield (jtype, q_slice, v_slice) for each joint that needs a special
        (non-vector-add) retract: the floating ROOT (SE(3), 7q/6v prefix) and
        every SPHERICAL joint (SO(3), 4q/3v block). All other joints retract by
        plain vector add and are handled by the q+v_dt fallback.
        """
        specs = []
        for joint in self.robot.get_joints_ordered_by_id():
            jid = joint.get_id()
            jtype = getattr(joint, "jtype", None)
            if self.robot.floating_base and jid == 0:
                specs.append(("floating",
                              self._as_index_list(self.robot.get_joint_index_q(jid)),
                              self._as_index_list(self.robot.get_joint_index_v(jid))))
            elif jtype == "spherical" and not getattr(joint, "is_mimic", False):
                specs.append(("spherical",
                              self._as_index_list(self.robot.get_joint_index_q(jid)),
                              self._as_index_list(self.robot.get_joint_index_v(jid))))
        return specs

    def integrate(self, q, v_dt):
        """Lie-group retract: q_new = q ⊕ v_dt.

        Per-joint dispatch matching Pinocchio's `pin.integrate(model, q, v_dt)`:
          - vector-space joints (revolute/prismatic/...): q_new = q + v_dt.
          - floating ROOT: the 7q/6v prefix drives an SE(3) exponential update
            of the position + xyzw-quaternion.
          - SPHERICAL joint: its 4q (unit quaternion) / 3v block drives an SO(3)
            quaternion exponential, q_new = q ⊗ exp(½·omega_dt).

        Convention (user-facing, matches Pinocchio):
          q     = [..., pos(3), quat_xyzw(4), ..., joint_q, ...]    size nq
          v_dt  = [..., v_lin*dt(3), omega*dt(3), ..., joint_v*dt, ...]  size nv
          q_new is in the same convention as q. Quaternion convention: xyzw.
        """
        q = np.asarray(q, dtype=np.float64).copy()
        v_dt = np.asarray(v_dt, dtype=np.float64)
        specs = self._joint_retract_specs()
        if not specs:
            return q + v_dt
        # Start from the plain vector add, then OVERWRITE each manifold block.
        q_new = q.copy()
        # Vector-add every v-slot into its q-slot; manifold blocks below replace
        # their own q-block (the floating quat/pos and each spherical quat).
        for joint in self.robot.get_joints_ordered_by_id():
            jid = joint.get_id()
            if self.robot.floating_base and jid == 0:
                continue
            if getattr(joint, "jtype", None) == "spherical" and not getattr(joint, "is_mimic", False):
                continue
            iq = self._as_index_list(self.robot.get_joint_index_q(jid))
            iv = self._as_index_list(self.robot.get_joint_index_v(jid))
            q_new[iq] = q[iq] + v_dt[iv]
        for jtype, iq, iv in specs:
            if jtype == "floating":
                # SE(3) free-flyer prefix (Pinocchio v_dt order [v_lin; omega]).
                rho = v_dt[iv][0:3]
                phi = v_dt[iv][3:6]
                V = self._so3_V_matrix(phi)
                p_delta_local = V @ rho
                R_old = self._rotation_from_quat_xyzw(q[iq][3:7])
                q_new[iq[0:3]] = q[iq][0:3] + R_old @ p_delta_local
                q_new[iq[3:7]] = self._spherical_retract(q[iq][3:7], phi)
            else:  # spherical
                q_new[iq] = self._spherical_retract(q[iq], v_dt[iv])
        return q_new

    def dIntegrate(self, q, v_dt, with_respect_to):
        """Return the (nv, nv) Jacobian of `integrate(q, v_dt)` in tangent
        space. `with_respect_to` is 'q' or 'v' (Pinocchio's ARG0 / ARG1 — ARG1
        is the Jacobian w.r.t. the v_dt argument, not v itself).

        Block-diagonal per joint:
          - vector-space joints: identity.
          - floating ROOT (Pinocchio order [v_lin; omega]):
              ARG_q : Ad(exp(-v_dt))  — SE(3) adjoint of the inverse exponential
              ARG_v : SE(3) right-Jacobian J_r(v_dt) with the Q(rho,phi) coupling
          - SPHERICAL joint (omega-only SO(3) restriction of the free-flyer):
              ARG_q : R_inv = exp(-omega_dt)
              ARG_v : J_r(omega_dt)  (SO(3) right Jacobian)
        """
        del q  # unused for the closed-form retracts
        nv = self.robot.get_num_vel()
        J = np.eye(nv)
        v_dt = np.asarray(v_dt, dtype=np.float64)
        for jtype, iq, iv in self._joint_retract_specs():
            if jtype == "floating":
                rho = v_dt[iv][0:3]
                phi = v_dt[iv][3:6]
                blk = self._se3_dintegrate_block(rho, phi, with_respect_to)
                ix = np.ix_(iv, iv)
                J[ix] = blk
            else:  # spherical: omega-only SO(3) restriction
                phi = v_dt[iv]  # omega * dt
                ix = np.ix_(iv, iv)
                if with_respect_to == "q":
                    J[ix] = self._so3_exp(-phi)
                elif with_respect_to == "v":
                    J[ix] = self._so3_right_jacobian(phi)
                else:
                    raise ValueError("with_respect_to must be 'q' or 'v'")
        return J

    def _se3_dintegrate_block(self, rho, phi, with_respect_to):
        """The 6x6 free-flyer dIntegrate block (Pinocchio order [v_lin; omega])."""
        blk = np.eye(6)
        if with_respect_to == "q":
            R_inv = self._so3_exp(-phi)
            V_neg = self._so3_V_matrix(-phi)
            p_inv = -V_neg @ rho
            P_inv_x = self._so3_skew(p_inv)
            blk[0:3, 0:3] = R_inv
            blk[0:3, 3:6] = P_inv_x @ R_inv
            blk[3:6, 0:3] = 0.0
            blk[3:6, 3:6] = R_inv
            return blk
        if with_respect_to == "v":
            J_r = self._so3_right_jacobian(phi)
            Q = self._se3_Q_block(rho, phi)
            blk[0:3, 0:3] = J_r
            blk[0:3, 3:6] = Q
            blk[3:6, 0:3] = 0.0
            blk[3:6, 3:6] = J_r
            return blk
        raise ValueError("with_respect_to must be 'q' or 'v'")

    def d2Integrate(self, q, v_dt, arg1, arg2, fd_step=1e-3):
        """Second-order Lie-group retract derivative: the tangent-space
        derivative of the `dIntegrate` Jacobian.

        Returns ``H`` of shape ``(nv, nv, nv)`` with

            H[i, j, k] = d/dxi_k ( dIntegrate(q, v_dt, arg1)[i, j] )

        where ``xi_k`` is a unit tangent perturbation in the ``arg2``
        coordinate:

          * ``arg2 == 'v'`` -> additive perturbation of the increment,
            ``v_dt -> v_dt + h * e_k``;
          * ``arg2 == 'q'`` -> right group perturbation of the base,
            ``q -> integrate(q, h * e_k)``.

        ``arg1, arg2`` each in ``{'q', 'v'}`` (Pinocchio ARG0/ARG1
        semantics). This is the building block for the second-order
        sensitivity of the time-integration step (e.g. the floating-base
        position rows of a `plant_step_hessian`).

        Structure for the project manifest (free-flyer + revolute joints,
        optional fixed base):

          * Fixed-base -> identically zero (`integrate` is affine, its
            Jacobian constant).
          * The free-flyer `dIntegrate` blocks depend only on ``v_dt`` and
            never on the base ``q`` (see `dIntegrate`, which discards
            ``q``), so every ``arg2 == 'q'`` tensor is exactly zero and the
            only nonzero entries live in the 6x6 free-flyer block over
            ``arg2 == 'v'``.
          * Revolute rows/cols are linear in the increment -> zero.

        Implementation note: evaluated by 4th-order (Richardson) central
        finite differencing of the pinocchio-validated `dIntegrate`
        (float64, ~1e-8 accurate). The default ``fd_step`` of 1e-3 is
        deliberately *not* tiny: it sits near the roundoff/truncation
        optimum for the 4th-order stencil (h ~ eps**(1/5)) and, crucially,
        keeps the perturbed increments clear of the ill-conditioned
        small-angle regime of `dIntegrate`'s exact (1-cos)/theta^2 and
        SE(3) Q-block formulas (cf. the theta < 1e-4 guard in
        `_se3_Q_block`), which dominate the error as v_dt -> 0. A
        hand-rolled closed form is deferred
        until a GPU/codegen consumer for the floating-base position-row
        Hessian exists; the only current consumers are float64 oracles,
        where exact-to-1e-9 differencing of an already-exact Jacobian is
        indistinguishable from a closed form. See
        docs/open-tasks/f1_plant_step_hessian_plan.md (floating-q rows).
        """
        if arg1 not in ("q", "v") or arg2 not in ("q", "v"):
            raise ValueError("arg1 and arg2 must each be 'q' or 'v'")
        nv = self.robot.get_num_vel()
        H = np.zeros((nv, nv, nv), dtype=np.float64)
        if not self.robot.floating_base:
            return H
        if arg2 == "q":
            # The free-flyer dIntegrate blocks are independent of the base
            # q, so this derivative is exactly zero.
            return H
        # ANALYTIC (2026-07-27): closed-form tangent derivative of dIntegrate,
        # per-joint. Free-flyer 6x6 block over its own 6 v-directions; each
        # SPHERICAL joint's 3x3 SO(3) block over its 3. Structured chain rule
        # (coeff'(theta)·phi_k/theta + coeff·d(skew-products)), small-angle Taylor
        # below theta=0.2. Validated to ~1e-14 vs an mpmath complex-step ground
        # truth across theta in [1e-9, 2]. See docs/open-tasks/plan_phase3_*.
        v_dt = np.asarray(v_dt, dtype=np.float64)
        for jtype, _iq, iv in self._joint_retract_specs():
            if jtype == "floating":
                rho = v_dt[iv][0:3]
                phi = v_dt[iv][3:6]
                for kk in range(6):
                    blk = self._d2_se3_dblock(rho, phi, arg1, kk)   # 6x6
                    for a in range(6):
                        for b in range(6):
                            H[iv[a], iv[b], iv[kk]] = blk[a, b]
            else:  # spherical: omega-only SO(3) restriction (3x3 over 3 dirs)
                phi = v_dt[iv]
                for k in range(3):
                    d3 = (self._d2_se3_dRinv(phi, k) if arg1 == "q"
                          else self._d2_se3_dJr(phi, k))
                    for a in range(3):
                        for b in range(3):
                            H[iv[a], iv[b], iv[k]] = d3[a, b]
        return H

    def _d2Integrate_fd(self, q, v_dt, arg1, arg2, fd_step=1e-3):
        """Legacy 4th-order central-FD d2Integrate, retained as a cross-check for
        the analytic `d2Integrate`. Same signature/semantics."""
        nv = self.robot.get_num_vel()
        H = np.zeros((nv, nv, nv), dtype=np.float64)
        if not self.robot.floating_base or arg2 == "q":
            return H
        v_dt = np.asarray(v_dt, dtype=np.float64)

        def J_at(delta):
            return self.dIntegrate(q, v_dt + delta, arg1)

        h = float(fd_step)
        for k in range(nv):
            e = np.zeros(nv, dtype=np.float64)
            e[k] = 1.0
            d1 = J_at(h * e) - J_at(-h * e)
            d2 = J_at(2.0 * h * e) - J_at(-2.0 * h * e)
            H[:, :, k] = (8.0 * d1 - d2) / (12.0 * h)
        return H

    # ---- Analytic d2Integrate building blocks (SE(3)-exp 2nd derivatives) ----
    # Coefficient value/derivative as functions of theta; exact above _D2_TH,
    # small-angle Taylor below (odd series for derivatives). See plan doc.
    _D2_TH = 0.2
    _D2_VAL_SERIES = {  # even series in theta^2 (coefficient VALUE)
        'a':  [1/2, -1/24, 1/720, -1/40320, 1/3628800],
        'b':  [1/6, -1/120, 1/5040, -1/362880, 1/39916800],
        's':  [1.0, -1/6, 1/120, -1/5040, 1/362880],
        'c2': [-1/24, 1/720, -1/40320, 1/3628800, -1/479001600],
        'c3': [1/120, -1/2520, 1/120960, -1/9979200, 1/1245404160],
    }
    _D2_DER_SERIES = {  # odd series (coefficient DERIVATIVE): sum coeff*theta^(2i+1)
        'a':  [-1/12, 1/180, -1/6720, 1/453600],
        'b':  [-1/60, 1/1260, -1/60480, 1/4989600],
        's':  [-1/3, 1/30, -1/840, 1/45360],
        'c2': [1/360, -1/10080, 1/604800, -1/59875200],
        'c3': [-1/1260, 1/30240, -1/1663200, 1/155675520],
    }

    @staticmethod
    def _d2_coef(name, t):
        """SE(3)-exp coefficient VALUE (a,b/c1,s,c2,c3) at theta=t."""
        if name == 'c1':
            name = 'b'
        if abs(t) >= RBDReference._D2_TH:
            if name == 'a':  return (1.0 - np.cos(t)) / (t * t)
            if name == 'b':  return (t - np.sin(t)) / t**3
            if name == 's':  return np.sin(t) / t
            if name == 'c2': return (1.0 - 0.5 * t * t - np.cos(t)) / t**4
            if name == 'c3': return -0.5 * ((1.0 - 0.5 * t * t - np.cos(t)) / t**4
                                            - 3.0 * (t - np.sin(t) - t**3 / 6.0) / t**5)
        r, p, t2 = 0.0, 1.0, t * t
        for c in RBDReference._D2_VAL_SERIES[name]:
            r += c * p; p *= t2
        return r

    @staticmethod
    def _d2_coef_der(name, t):
        """SE(3)-exp coefficient DERIVATIVE d(coef)/d(theta) at theta=t."""
        if name == 'c1':
            name = 'b'
        if abs(t) >= RBDReference._D2_TH:
            if name == 'a':  return (t * np.sin(t) - 2.0 * (1.0 - np.cos(t))) / t**3
            if name == 'b':  return ((1.0 - np.cos(t)) * t - 3.0 * (t - np.sin(t))) / t**4
            if name == 's':  return (t * np.cos(t) - np.sin(t)) / (t * t)
            if name == 'c2': return (t * np.sin(t) + t * t + 4.0 * np.cos(t) - 4.0) / t**5
            if name == 'c3':
                c2p = (t * np.sin(t) + t * t + 4.0 * np.cos(t) - 4.0) / t**5
                return -0.5 * (c2p - 3.0 * (-4.0 * t - t * np.cos(t)
                                            + 5.0 * np.sin(t) + t**3 / 3.0) / t**6)
        r, p, t2 = 0.0, t, t * t
        for c in RBDReference._D2_DER_SERIES[name]:
            r += c * p; p *= t2
        return r

    @staticmethod
    def _d2_dcoef_dphi(name, phi, k):
        """d(coef(theta))/d(phi_k) = coef'(theta) * (phi_k / theta)."""
        t = float(np.sqrt(phi @ phi))
        if t == 0.0:
            return 0.0
        return RBDReference._d2_coef_der(name, t) * (phi[k] / t)

    @staticmethod
    def _d2_se3_dJr(phi, k):
        """d Jr(phi) / d phi_k  (3x3). Also the spherical ARG_v derivative."""
        S = RBDReference._so3_skew(phi); Ek = RBDReference._so3_skew(np.eye(3)[k])
        t = float(np.sqrt(phi @ phi))
        a = RBDReference._d2_coef('a', t); b = RBDReference._d2_coef('b', t)
        return (-RBDReference._d2_dcoef_dphi('a', phi, k) * S - a * Ek
                + RBDReference._d2_dcoef_dphi('b', phi, k) * (S @ S) + b * (Ek @ S + S @ Ek))

    @staticmethod
    def _d2_se3_dRinv(phi, k):
        """d exp(-phi) / d phi_k  (3x3). Also the spherical ARG_q derivative."""
        S = RBDReference._so3_skew(phi); Ek = RBDReference._so3_skew(np.eye(3)[k])
        t = float(np.sqrt(phi @ phi))
        s = RBDReference._d2_coef('s', t); a = RBDReference._d2_coef('a', t)
        return (-RBDReference._d2_dcoef_dphi('s', phi, k) * S - s * Ek
                + RBDReference._d2_dcoef_dphi('a', phi, k) * (S @ S) + a * (Ek @ S + S @ Ek))

    @staticmethod
    def _d2_se3_dQ(rho, phi, kk):
        """d Q(rho,phi) / d w_kk (kk<3 -> rho, else phi). Q = -sola."""
        Px = RBDReference._so3_skew(-phi); Rx = RBDReference._so3_skew(rho)
        t = float(np.sqrt(phi @ phi)); Px2 = Px @ Px
        c1 = RBDReference._d2_coef('c1', t); c2 = RBDReference._d2_coef('c2', t)
        c3 = RBDReference._d2_coef('c3', t)
        if kk < 3:  # d/d rho_kk: Q linear in rho -> substitute Rx -> skew(e_kk)
            Ek = RBDReference._so3_skew(np.eye(3)[kk])
            A1 = Px @ Ek + Ek @ Px + Px @ Ek @ Px
            A2 = Px2 @ Ek + Ek @ Px2 - 3.0 * (Px @ Ek @ Px)
            A3 = Px @ Ek @ Px2 + Px2 @ Ek @ Px
            return -(0.5 * Ek + c1 * A1 - c2 * A2 + c3 * A3)
        k = kk - 3  # d/d phi_k: Px=-Sphi -> dPx=-E_k, c-coeffs via chain
        dPx = -RBDReference._so3_skew(np.eye(3)[k]); dPx2 = dPx @ Px + Px @ dPx
        A1 = Px @ Rx + Rx @ Px + Px @ Rx @ Px
        A2 = Px2 @ Rx + Rx @ Px2 - 3.0 * (Px @ Rx @ Px)
        A3 = Px @ Rx @ Px2 + Px2 @ Rx @ Px
        dA1 = dPx @ Rx + Rx @ dPx + (dPx @ Rx @ Px + Px @ Rx @ dPx)
        dA2 = dPx2 @ Rx + Rx @ dPx2 - 3.0 * (dPx @ Rx @ Px + Px @ Rx @ dPx)
        dA3 = (dPx @ Rx @ Px2 + Px @ Rx @ dPx2) + (dPx2 @ Rx @ Px + Px2 @ Rx @ dPx)
        dc1 = RBDReference._d2_dcoef_dphi('c1', phi, k)
        dc2 = RBDReference._d2_dcoef_dphi('c2', phi, k)
        dc3 = RBDReference._d2_dcoef_dphi('c3', phi, k)
        return -(dc1 * A1 + c1 * dA1 - dc2 * A2 - c2 * dA2 + dc3 * A3 + c3 * dA3)

    @staticmethod
    def _d2_se3_doff(rho, phi, kk):
        """d off / d w_kk, off = skew(-Jr@rho) @ exp(-phi) (ARG_q coupling)."""
        Jr = np.eye(3) - RBDReference._d2_coef('a', float(np.sqrt(phi @ phi))) * RBDReference._so3_skew(phi) \
            + RBDReference._d2_coef('b', float(np.sqrt(phi @ phi))) * (RBDReference._so3_skew(phi) @ RBDReference._so3_skew(phi))
        R = np.eye(3) - RBDReference._d2_coef('s', float(np.sqrt(phi @ phi))) * RBDReference._so3_skew(phi) \
            + RBDReference._d2_coef('a', float(np.sqrt(phi @ phi))) * (RBDReference._so3_skew(phi) @ RBDReference._so3_skew(phi))
        if kk < 3:  # p_inv=-Jr@rho linear in rho -> dp=-Jr[:,kk], dR=0
            return RBDReference._so3_skew(-Jr[:, kk]) @ R
        k = kk - 3
        p_inv = -Jr @ rho
        dp = -RBDReference._d2_se3_dJr(phi, k) @ rho
        return RBDReference._so3_skew(dp) @ R + RBDReference._so3_skew(p_inv) @ RBDReference._d2_se3_dRinv(phi, k)

    @staticmethod
    def _d2_se3_dblock(rho, phi, arg1, kk):
        """The 6x6 d(dIntegrate block)/d w_kk for the free-flyer, arg1 in {q,v}."""
        blk = np.zeros((6, 6))
        if arg1 == "v":
            dJ = RBDReference._d2_se3_dJr(phi, kk - 3) if kk >= 3 else np.zeros((3, 3))
            blk[0:3, 0:3] = dJ; blk[3:6, 3:6] = dJ
            blk[0:3, 3:6] = RBDReference._d2_se3_dQ(rho, phi, kk)
        else:  # arg1 == "q"
            dR = RBDReference._d2_se3_dRinv(phi, kk - 3) if kk >= 3 else np.zeros((3, 3))
            blk[0:3, 0:3] = dR; blk[3:6, 3:6] = dR
            blk[0:3, 3:6] = RBDReference._d2_se3_doff(rho, phi, kk)
        return blk

    # ----- Multi-integrator one-step time-integration -----

    @staticmethod
    def _integrator_butcher(integrator_type: str):
        """Return (c_list, b_list) for the given integrator.

        c_list[i] is the stage-i offset applied to xdot (i.e. the
        intermediate point for FD evaluation at stage i+1 uses
        p_{i+1}.v = v_orig + c_i * dt * qdd_i).
        b_list[i] is the combination weight for qdd_i in the final v update.
        Lengths: len(c_list) = N-1, len(b_list) = N (N = stage count).
        """
        if integrator_type == "euler":
            return [], [1.0]
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            return [], [1.0]
        if integrator_type == "midpoint":
            return [0.5], [0.0, 1.0]
        if integrator_type == "rk3":
            return [0.5, 0.75], [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]
        if integrator_type == "rk4":
            return [0.5, 0.5, 1.0], [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0]
        raise ValueError(f"Unknown integrator_type: {integrator_type}")

    def integrator(self, q, qd, u, dt, integrator_type: str = "euler", f_ext=None):
        """One time-integration step: x_{k+1} = integrator(x_k, u_k, dt).

        Returns x_kp1 of shape (nq + nv,) — concatenated [q_new, v_new] in the
        user-facing q/v convention. Supports 'euler', 'semi_implicit_euler',
        'trapezoidal', 'midpoint', 'rk3', 'rk4'. For floating-base robots the
        q-update uses `self.integrate` (Lie-group retract); for fixed-base this
        collapses to `q + dt*v`.

        `f_ext` (optional, body-major local-frame [angular; linear], one 6-vector
        per body) is threaded into every forward-dynamics evaluation so the step
        is taken at the f_ext-perturbed operating point (matches GRiD threading
        d_f_ext through the integrator's FD inner). None => no external forces.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        qdd1 = np.asarray(self.forward_dynamics(q, qd, u, f_ext=f_ext)).reshape(-1)
        if integrator_type == "euler":
            q_new = self.integrate(q, dt * qd)
            v_new = qd + dt * qdd1
            return np.concatenate([q_new, v_new])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            v_new = qd + dt * qdd1
            q_new = self.integrate(q, dt * v_new)
            return np.concatenate([q_new, v_new])
        if integrator_type == "trapezoidal":
            # GATO/GRiD trapezoidal (single-stage): v reads qdd Euler-style; q
            # retracts the combined tangent dt*qd + 0.5*dt^2*qdd in ONE step. For
            # fixed-base this is q + dt*qd + 0.5*dt^2*qdd; for floating-base
            # self.integrate applies the SE(3) Lie retract of the combined tangent.
            v_new = qd + dt * qdd1
            q_new = self.integrate(q, dt * qd + 0.5 * dt * dt * qdd1)
            return np.concatenate([q_new, v_new])
        # Multi-stage RK family — TrajoptPlant convention: each stage uses the
        # ORIGINAL v for its xdot.v term (only qdd is refined across stages).
        c_list, b_list = self._integrator_butcher(integrator_type)
        N = len(b_list)
        qdd_list = [qdd1]
        prev_qdd = qdd1
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            p_q  = self.integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            stage_qdd = np.asarray(self.forward_dynamics(p_q, p_qd, u, f_ext=f_ext)).reshape(-1)
            qdd_list.append(stage_qdd)
            prev_qdd = stage_qdd
        accel = sum(b * qdd for b, qdd in zip(b_list, qdd_list))
        q_new = self.integrate(q, dt * qd)   # q update is Euler-style for every RK variant
        v_new = qd + dt * accel
        return np.concatenate([q_new, v_new])

    def integrator_gradient(self, q, qd, u, dt, integrator_type: str = "euler", f_ext=None):
        """Return [A | B] of shape (2*nv, 3*nv) — the Jacobian of the integrator
        step in tangent space. Column order is [d/dq | d/dqd | d/du] where
        d/dq is the nv-tangent perturbation of q (NOT the nq scalar
        perturbation). For fixed-base this matches the historical layout.

        `f_ext` (optional, body-major local-frame) is threaded into every FD value
        and FD-gradient evaluation, so the gradient is linearized at the
        f_ext-perturbed operating point (matches GRiD threading d_f_ext into the
        integrator gradient's vaf/ID linearization). None => no external forces.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        nv = self.robot.get_num_vel()
        I_n = np.eye(nv)
        Z_n = np.zeros((nv, nv))

        def fd_grad_at(pq, pqd):
            J_qq, J_qv = self.forward_dynamics_gradient(pq, pqd, u, f_ext=f_ext)
            return (np.asarray(J_qq, dtype=np.float64),
                    np.asarray(J_qv, dtype=np.float64),
                    np.asarray(self.minv(pq), dtype=np.float64))

        # dq_block / dv_block: (nv x nv) Jacobians of q_new w.r.t. q and v
        # respectively, evaluated at v_dt. For fixed-base both reduce to
        # identity (q_block) and dt*I (v_block); for floating-base they
        # involve the SO(3) right-Jacobian on the free-flyer block.
        def q_top_blocks(v_dt_arg):
            dInt_q = self.dIntegrate(q, v_dt_arg, "q")
            dInt_v = self.dIntegrate(q, v_dt_arg, "v")
            return dInt_q, dInt_v

        if integrator_type == "euler":
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            dInt_q, dInt_v = q_top_blocks(dt * qd)
            top = np.hstack([dInt_q, dt * dInt_v, Z_n])
            bottom = np.hstack([dt * J_qq, I_n + dt * J_qv, dt * Minv])
            return np.vstack([top, bottom])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            # v_new = qd + dt*qdd(q, qd, u);  q_new = integrate(q, dt*v_new)
            # ∂v_new/∂q  = dt*J_qq
            # ∂v_new/∂qd = I + dt*J_qv
            # ∂v_new/∂u  = dt*Minv
            # ∂q_new/∂q  = dInt_q(dt*v_new) + dt*dInt_v(dt*v_new) @ ∂v_new/∂q
            # ∂q_new/∂qd = dt*dInt_v(dt*v_new) @ ∂v_new/∂qd
            # ∂q_new/∂u  = dt*dInt_v(dt*v_new) @ ∂v_new/∂u
            qdd_si = np.asarray(self.forward_dynamics(q, qd, u, f_ext=f_ext)).reshape(-1)
            v_new = qd + dt * qdd_si
            dInt_q, dInt_v = q_top_blocks(dt * v_new)
            dvdq = dt * J_qq
            dvdv = I_n + dt * J_qv
            dvdu = dt * Minv
            dt_dInt_v = dt * dInt_v
            top = np.hstack([dInt_q + dt_dInt_v @ dvdq,
                              dt_dInt_v @ dvdv,
                              dt_dInt_v @ dvdu])
            bottom = np.hstack([dvdq, dvdv, dvdu])
            return np.vstack([top, bottom])
        if integrator_type == "trapezoidal":
            # v_new = qd + dt*qdd(q,qd,u);  q_new = integrate(q, dt*qd + dt2h*qdd),
            # dt2h = 0.5*dt*dt. Bottom (v) rows match Euler/SI (dt*); the top (q)
            # rows weight the FD gradient by dt2h (the +0.5*dt^2*qdd accel term).
            # General for both bases: dInt_q/dInt_v collapse to I/dt*I fixed-base,
            # and carry the SE(3) free-flyer blocks (at tangent w) floating-base.
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            qdd_t = np.asarray(self.forward_dynamics(q, qd, u, f_ext=f_ext)).reshape(-1)
            dt2h = 0.5 * dt * dt
            w = dt * qd + dt2h * qdd_t
            dInt_q, dInt_v = q_top_blocks(w)
            # q_new = integrate(q, w):  dw/dq = dt2h*J_qq, dw/dqd = dt*I + dt2h*J_qv,
            # dw/du = dt2h*Minv. top = dInt_q + dInt_v @ dw/d*.
            dwdq = dt2h * J_qq
            dwdv = dt * I_n + dt2h * J_qv
            dwdu = dt2h * Minv
            top = np.hstack([dInt_q + dInt_v @ dwdq,
                             dInt_v @ dwdv,
                             dInt_v @ dwdu])
            bottom = np.hstack([dt * J_qq, I_n + dt * J_qv, dt * Minv])
            return np.vstack([top, bottom])
        # ----- Multi-stage chain rule (Midpoint / RK3 / RK4) -----
        c_list, b_list = self._integrator_butcher(integrator_type)
        N = len(b_list)
        qdd_list = []
        D_qdd_list = []
        # Stage 1: FD at the original (q, qd).
        qdd_list.append(np.asarray(self.forward_dynamics(q, qd, u, f_ext=f_ext)).reshape(-1))
        J_qq, J_qv, Minv = fd_grad_at(q, qd)
        D_qdd_list.append(np.hstack([J_qq, J_qv, Minv]))  # (nv, 3*nv)
        # Subsequent stages: chain rule through self.integrate at the
        # intermediate point (q_orig perturbed by c_{i-1}*dt*v_orig).
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            prev_qdd = qdd_list[-1]
            p_q = self.integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            stage_qdd = np.asarray(self.forward_dynamics(p_q, p_qd, u, f_ext=f_ext)).reshape(-1)
            qdd_list.append(stage_qdd)
            J_qq_i, J_qv_i, Minv_i = fd_grad_at(p_q, p_qd)
            # ∂p_i.q / ∂(q, v, u) — block structure (each (nv, nv)):
            #   [dInt_q(q, c_prev*dt*v) | c_prev*dt*dInt_v(q, c_prev*dt*v) | 0]
            v_dt_stage = c_prev * dt * qd
            dInt_q_stage, dInt_v_stage = q_top_blocks(v_dt_stage)
            dp_q_block = np.hstack([dInt_q_stage, c_prev * dt * dInt_v_stage, Z_n])
            # ∂p_i.qd / ∂(q, v, u) = [0, I, 0] + c_prev*dt * D_qdd_{i-1}
            dp_qd_block = np.hstack([Z_n, I_n, Z_n]) + c_prev * dt * D_qdd_list[-1]
            # Compose: D_qdd_i = J_qq_i @ dp_q_block + J_qv_i @ dp_qd_block + Minv_i @ [0|0|I]
            d_u_block = np.hstack([Z_n, Z_n, Minv_i])
            D_qdd_list.append(J_qq_i @ dp_q_block + J_qv_i @ dp_qd_block + d_u_block)

        sum_b_D = sum(b * D for b, D in zip(b_list, D_qdd_list))
        # Final assembly. q_new = integrate(q, dt*qd) — same as Euler.
        dInt_q_final, dInt_v_final = q_top_blocks(dt * qd)
        top = np.hstack([dInt_q_final, dt * dInt_v_final, Z_n])
        bottom = np.hstack([Z_n, I_n, Z_n]) + dt * sum_b_D
        return np.vstack([top, bottom])

    @staticmethod
    def _quat_xyzw_from_rotation_matrix(rot, reference_quat=None):
        rot = np.asarray(rot, dtype=np.float64)
        trace = np.trace(rot)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
        else:
            diag = np.diag(rot)
            if diag[0] > diag[1] and diag[0] > diag[2]:
                s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
                w = (rot[2, 1] - rot[1, 2]) / s
                x = 0.25 * s
                y = (rot[0, 1] + rot[1, 0]) / s
                z = (rot[0, 2] + rot[2, 0]) / s
            elif diag[1] > diag[2]:
                s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
                w = (rot[0, 2] - rot[2, 0]) / s
                x = (rot[0, 1] + rot[1, 0]) / s
                y = 0.25 * s
                z = (rot[1, 2] + rot[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
                w = (rot[1, 0] - rot[0, 1]) / s
                x = (rot[0, 2] + rot[2, 0]) / s
                y = (rot[1, 2] + rot[2, 1]) / s
                z = 0.25 * s
        quat = RBDReference._normalize_xyzw_quaternion(np.array([x, y, z, w], dtype=np.float64))
        if reference_quat is not None:
            reference_quat = RBDReference._normalize_xyzw_quaternion(reference_quat)
            if np.dot(quat, reference_quat) < 0.0:
                quat = -quat
        return quat


    def cross_operator(self, v):
        """Compute the 6x6 spatial cross product matrix for a velocity vector.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_cross : numpy.ndarray
            6x6 spatial cross product matrix.
        """
        # for any vector v, computes the operator v x 
        # vec x = [wx   0]
        #         [vox wx]
        #(crm in spatial_v2_extended)
        v_cross = np.array([0, -v[2], v[1], 0, 0, 0,
                            v[2], 0, -v[0], 0, 0, 0,
                            -v[1], v[0], 0, 0, 0, 0,
                            0, -v[5], v[4], 0, -v[2], v[1], 
                            v[5], 0, -v[3], v[2], 0, -v[0],
                            -v[4], v[3], 0, -v[1], v[0], 0]
                          ).reshape(6,6)
        return(v_cross)
    
    def dual_cross_operator(self, v):
        """Compute the 6x6 spatial dual cross product matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_dual_cross : numpy.ndarray
            6x6 spatial dual cross product matrix.
        """
        #(crf in in spatial_v2_extended)
        return(-1 * self.cross_operator(v).T)
    
    def dot_matrix(self, I, v):
        """Compute the time derivative of the spatial inertia matrix.

        Parameters
        ----------
        I : numpy.ndarray
            6x6 spatial inertia matrix.
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        I_dot : numpy.ndarray
            6x6 time derivative of spatial inertia.
        """
        return self.dual_cross_operator(v) @ I - I @ self.cross_operator(v)
    
    def icrf(self, v):
        """Compute the inverse of the force (dual) cross operator.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        res : numpy.ndarray
            6x6 inverse force cross operator matrix.
        """
        #helper function defined in spatial_v2_extended library, called by idsva() and inverse_dynamics_gradient()
        # inverse of the force(dual) cross operator
        # v crf f = f icrf v
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asarray(res, dtype=np.float64)
    
    def factor_functions(self, I, v, number=3):
        """Helper functions for factorization in IDSVA and RNEA gradient.

        Parameters
        ----------
        I : numpy.ndarray
            6x6 spatial inertia matrix.
        v : numpy.ndarray
            6D spatial velocity vector.
        number : int
            Type of factorization to perform.

        Returns
        -------
        B : numpy.ndarray
            The resulting factorized matrix.
        """
        # helper function defined in spatial_v2_extended library, called by idsva() and inverse_dynamics_gradient()
        if number == 1:
            B = self.dual_cross_operator(v) * I
        elif number == 2:
            B = self.icrf(np.matmul(I,v)) - I * self.cross_operator(v)
        else:
            B = 1/2 * (np.matmul(self.dual_cross_operator(v),I) + self.icrf(np.matmul(I,v)) - np.matmul(I, self.cross_operator(v)))

        return B

    def _mxS(self, S, vec, alpha=1.0):
        """Compute the product of a cross operator matrix and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        vS : numpy.ndarray
            The 6xN matrix product.
        """
        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        return np.squeeze(np.array((alpha * np.dot(self.cross_operator(vec), S)))) # added np.squeeze and np.array

    def mxS(self, S, vec):
        """Compute the cross product of a spatial vector and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        vS : numpy.ndarray
            The 6xN matrix product.
        """
        result = np.zeros((6))
        # Flatten S to 1-D so each S[k] is a true scalar: a 6x1 subspace column
        # arrives as shape (6,1), so without this reshape S[k] would be a
        # 1-element array. Passing that as the scalar `alpha` into mx1-mx6's
        # per-element writes would raise NumPy's "Conversion of an array with
        # ndim > 0 to a scalar is deprecated" DeprecationWarning (will error in
        # a future NumPy). The reshape extracts genuine scalars and silences it.
        S = np.asarray(S).reshape(-1)
        if not S[0] == 0:
            result += self.mx1(vec, S[0])
        if not S[1] == 0:
            result += self.mx2(vec, S[1])
        if not S[2] == 0:
            result += self.mx3(vec, S[2])
        if not S[3] == 0:
            result += self.mx4(vec, S[3])
        if not S[4] == 0:
            result += self.mx5(vec, S[4])
        if not S[5] == 0:
            result += self.mx6(vec, S[5])
        return result

    def mx1(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 1).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[1] = vec[2] * alpha
        vecX[2] = -vec[1] * alpha
        vecX[4] = vec[5] * alpha
        vecX[5] = -vec[4] * alpha
        return vecX

    def mx2(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 2).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[0] = -vec[2] * alpha
        vecX[2] = vec[0] * alpha
        vecX[3] = -vec[5] * alpha
        vecX[5] = vec[3] * alpha
        return vecX

    def mx3(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 3).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[0] = vec[1] * alpha
        vecX[1] = -vec[0] * alpha
        vecX[3] = vec[4] * alpha
        vecX[4] = -vec[3] * alpha
        return vecX

    def mx4(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 4).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[4] = vec[2] * alpha
        vecX[5] = -vec[1] * alpha
        return vecX

    def mx5(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 5).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[3] = -vec[2] * alpha
        vecX[5] = vec[0] * alpha
        return vecX

    def mx6(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 6).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        vec = np.asarray(vec).reshape(-1)
        vecX[3] = vec[1] * alpha
        vecX[4] = -vec[0] * alpha
        return vecX

    def fxv(self, fxVec, timesVec):
        """Compute the spatial cross product between two spatial vectors.

        Parameters
        ----------
        v1 : numpy.ndarray
            First 6D spatial vector.
        v2 : numpy.ndarray
            Second 6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            6D spatial vector cross product.
        """
        # Fx(fxVec)*timesVec
        #   0  -v(2)  v(1)    0  -v(5)  v(4)
        # v(2)    0  -v(0)  v(5)    0  -v(3)
        # -v(1)  v(0)    0  -v(4)  v(3)    0
        #   0     0     0     0  -v(2)  v(1)
        #   0     0     0   v(2)    0  -v(0)
        #   0     0     0  -v(1)  v(0)    0
        result = np.zeros((6))
        result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
        result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
        result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
        result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
        result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
        result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
        return result

    def fxS(self, S, vec, alpha=1.0):
        """Compute the spatial cross product between a vector and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        res : numpy.ndarray
            6xN spatial cross product matrix.
        """
        # force spatial cross product with motion subspace
        return np.squeeze(
            np.array(alpha * np.matmul(self.dual_cross_operator(S), vec))
        )

    def vxIv(self, vec, Imat):
        """Compute the spatial force vector v x (I * v).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        I : numpy.ndarray
            6x6 spatial inertia matrix.

        Returns
        -------
        res : numpy.ndarray
            6D spatial force vector.
        """
        # necessary component in differentiating Iv (product rule).
        # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
        # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
        temp = np.matmul(Imat, vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec
    

    """
    End Effector Joint Selector

    Helper function to select specific end-effector joints for the end-effector position and gradient functions. If no joints specified then defaults to all leaf joints.
    """
    def select_end_effector_joints(self, ee_joint_names):
        """Identify the joint indices along the chain to the specified end effector.

        Parameters
        ----------
        ee_id : int
            Index of the end effector.

        Returns
        -------
        q_inds : list
            List of joint indices from root to end effector.

        Raises
        ------
        ValueError if ee_id is not a valid end-effector index.
        """
        # deterimine the target joints for the kinematic calcs
        ee_jids = []
        fixed_jids = []
        # if no joints specified then do all leaf joints
        if ee_joint_names is None:
            ee_jids = self.robot.get_leaf_nodes()
        # else search for specific end-effector joints
        else:
            if isinstance(ee_joint_names, str):
                ee_joint_names = [ee_joint_names]
            ee_jids = []
            fixed_jids = []
            for name in ee_joint_names:
                joint = self.robot.get_joint_by_name(name)
                if joint is not None:
                    ee_jids.append(joint.get_id())
                else:
                    fjoint = self.robot.get_fixed_joint_by_name(name)
                    if fjoint is None:
                        raise ValueError("Could not find joint or fixed joint named: " + name)
                    fixed_jids.append(fjoint.get_id())
        return ee_jids, fixed_jids

    # End-effector positions. see docs/open-tasks/notes.md (RBDReference.py:936)

    def _normalize_ee_offsets(self, offsets=None):
        if offsets is None:
            offsets = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)]
        return [
            np.asarray(offset, dtype=np.float64).reshape(4, 1)
            for offset in offsets
        ]

    def _normalize_ee_offset_transforms(self, offsets=None):
        """Normalize ee offsets to a list of 4x4 SE(3) tool transforms X_tool.

        Each offset may be given either as a POINT (``[x,y,z]`` or homogeneous
        ``[x,y,z,1]``) -- a pure translation, ``R_tool = I`` -- or as a full 4x4
        SE(3) ``X_tool`` (rotation + translation of the tool/tip frame in the
        target joint frame). This is the runtime "welded tool" offset: the tip
        frame is ``X_frame = X_target @ X_tool``.

        A point offset yields output BYTE-IDENTICAL to the legacy point path:
        ``X @ [[I,p],[0,1]]`` has the same position column (``X @ [x,y,z,1]``) and
        the same rotation block (``R @ I == R``) as ``X`` itself, so the pose rpy
        and the Jacobian lever arm are unchanged.
        """
        if offsets is None:
            return [np.eye(4, dtype=np.float64)]
        out = []
        for off in offsets:
            A = np.asarray(off, dtype=np.float64)
            if A.shape == (4, 4):
                out.append(A.copy())
            else:
                X = np.eye(4, dtype=np.float64)
                X[:3, 3] = A.reshape(-1)[:3]
                out.append(X)
        return out

    def end_effector_pose(self, q, ee_joint_names = None, ee_offsets = None):
        """Compute the 4x4 homogeneous transformation matrix of the end effector.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        T : numpy.ndarray
            4x4 homogeneous transformation matrix.
        """
        # chain up the transforms (version 1 for starting from the root)
        def forwardChain(self, jid, q):
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            # then chain them up
            Xmat_hom = np.eye(4)
            for ind in jidChain:
                inds_q = self.robot.get_joint_index_q(ind)
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[inds_q])
                Xmat_hom = np.matmul(Xmat_hom,currX)
            return Xmat_hom

        # chain up the transforms (version 2 for starting from the leaf)
        def backwardChain(self, jid, q, finalXmat_hom = np.eye(4)):
            currId = jid
            Xmat_hom = finalXmat_hom
            while(currId != -1):
                inds_q = self.robot.get_joint_index_q(currId)
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[inds_q])
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)
            return Xmat_hom

        # Extract the end-effector pose at the tool frame X_frame = X_target @ X_tool.
        # X_tool is a 4x4 SE(3) offset (point offset => R_tool = I; see
        # _normalize_ee_offset_transforms). see docs/open-tasks/notes.md (RBDReference.py:987)
        def eePos_from_Xmat_hom(Xmat_hom, ee_offset_Xtools):
            # tool/tip frame: post-multiply the target world transform by X_tool
            X_frame = np.matmul(np.asarray(Xmat_hom, dtype=np.float64), ee_offset_Xtools[0])

            # xyz position is the translation column of the tool frame
            eePos_xyz = X_frame[:3, 3:4]

            # roll pitch yaw from the TOOL-frame rotation block (= target rotation
            # when R_tool = I, so a point offset is byte-identical)
            eePos_roll = np.arctan2(X_frame[2,1],X_frame[2,2])
            pitch_temp = np.sqrt(X_frame[2,2]*X_frame[2,2] + X_frame[2,1]*X_frame[2,1])
            eePos_pitch = np.arctan2(-X_frame[2,0],pitch_temp)
            eePos_yaw = np.arctan2(X_frame[1,0],X_frame[0,0])
            eePos_rpy = np.array([[eePos_roll], [eePos_pitch], [eePos_yaw]], dtype=np.float64)

            # then stack it up!
            eePos = np.vstack((eePos_xyz,eePos_rpy))
            return eePos

        # do the actual computations
        ee_offset_Xtools = self._normalize_ee_offset_transforms(ee_offsets)
        eePos_arr = []
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        for jid in ee_jids:
            # Xmat_hom = forwardChain(self, jid, q)
            Xmat_hom = backwardChain(self, jid, q)
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offset_Xtools)
            eePos_arr.append(eePos)
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            if fj.parent_name == -1:
                Xmat_hom = fj.get_transformation_matrix_hom()
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                Xmat_hom = backwardChain(self, parent.get_id(), q, fj.get_transformation_matrix_hom())
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offset_Xtools)
            eePos_arr.append(eePos)
        return eePos_arr

    """
    End Effectors Pose Gradients
    """
    def equals_or_hstack(self, obj, col):
        """Concatenate or assign vectors depending on initialization state.

        Parameters
        ----------
        target : numpy.ndarray or None
            Target array to concatenate to.
        source : numpy.ndarray
            Array to append.

        Returns
        -------
        res : numpy.ndarray
            The resulting concatenated array.
        """
        if obj is None:
            obj = col
        else:
            obj = np.hstack((obj,col))
        return obj

    def _normalize_kinematics_q(self, q):
        """Ensure the joint position vector corresponds to the robot degrees of freedom.

        Parameters
        ----------
        q : numpy.ndarray
            Vector of joint positions.

        Returns
        -------
        q : numpy.ndarray
            Normalized joint position vector.

        Raises
        ------
        ValueError if q length does not match robot model degrees of freedom.
        """
         # keep the user-facing floating-base convention in one place:
        # q = [x, y, z, qx, qy, qz, qw, ...]
        #
        # for the analytic kinematics helpers below we assume the quaternion
        # block is already normalized, so make that explicit here rather than
        # scattering the normalization logic through each helper.
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.floating_base and self.robot.using_quaternion:
            quat = q[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm == 0.0:
                raise ValueError("Floating-base quaternion norm was zero during kinematics normalization.")
            q[3:7] = quat / quat_norm
        return q

    def end_effector_pose_gradient(self, q, ee_joint_names = None, ee_offsets = None):
        """Analytic gradient of the end-effector pose w.r.t. generalized velocity v.

        Convention: **d/dv (tangent space)**, matching pinocchio. Output is 6 x nv per
        end-effector (NOT 6 x nq). For fixed-base robots nq == nv so the shape is
        unchanged. For floating-base robots nv = 6 + n_joints (spatial twist of the
        base + joint velocities), versus nq = 7 + n_joints (xyz + quaternion + joint
        positions). The base block is now standard spatial (omega; v_world) rather
        than the older non-standard quaternion-component derivatives. Pose is
        [xyz; rpy] with R = Rz(yaw)Ry(pitch)Rx(roll).

        Algorithm: shared-chain geometric (spatial) Jacobian. One forward-kinematics
        pass caches every joint's world transform, then each velocity column is
        filled in O(1) -- O(nv + depth) total -- replacing the per-column FK
        re-chain (O(nq * depth)) used previously.

        Parameters
        ----------
        q : numpy.ndarray
            Generalized position. Floating base uses [xyz, quat(xyzw), joints].
        ee_joint_names : list of str, optional
            Joint names to use as end-effectors; defaults to the robot leaves.
        ee_offsets : list, optional
            Per-end-effector point offsets in the EE joint frame, as homogeneous
            points [x, y, z, 1]. Currently the first offset is applied to every EE.

        Returns
        -------
        list of numpy.ndarray
            Per end-effector 6 x nv matrix of d(pose)/dv.
        """
        q = self._normalize_kinematics_q(q)
        ee_offset_Xtools = self._normalize_ee_offset_transforms(ee_offsets)
        nv = self.robot.get_num_vel()
        n_joints = self.robot.get_num_joints()

        def vinds_for(jid):
            try:
                inds = self.robot.get_joint_index_v(jid)
            except Exception:
                inds = self.robot.get_joint_index_q(jid)
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def q_arg(jid):
            # Use the mimic-aware helper so a mimic joint's transform sees
            # `multiplier * q[mimicked] + offset`, exactly as the URDF prescribes.
            return self.robot.q_for_joint(jid, q)

        def mimic_scale(jid):
            joint = self.robot.get_joint_by_id(jid)
            return joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0

        # one forward-kinematics pass: world transform of every joint
        Xw = [None] * n_joints
        for j in range(n_joints):
            X_local = np.asarray(self.robot.get_Xmat_hom_Func_by_id(j)(q_arg(j)),
                                 dtype=np.float64)
            par = self.robot.get_parent_id(j)
            Xw[j] = X_local if par == -1 else (Xw[par] @ X_local)

        # E(rpy) maps rpy-rates -> WORLD angular velocity for R = Rz(yaw)Ry(pitch)Rx(roll):
        #   omega_world = E * [roll_dot; pitch_dot; yaw_dot]
        # so d(rpy)/dv = E^{-1} * J_omega_world
        def E_world(R_ee):
            roll = np.arctan2(R_ee[2, 1], R_ee[2, 2])
            pitch = np.arctan2(-R_ee[2, 0],
                               np.sqrt(R_ee[2, 2] * R_ee[2, 2] + R_ee[2, 1] * R_ee[2, 1]))
            yaw = np.arctan2(R_ee[1, 0], R_ee[0, 0])
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            return np.array([[cy * cp, -sy, 0.0],
                             [sy * cp,  cy, 0.0],
                             [-sp,     0.0, 1.0]], dtype=np.float64)

        def jacobian_for_chain(chain_jids, X_ee):
            # tool/tip frame X_frame = X_ee @ X_tool; the lever arm uses the tip
            # position p_ee and the E^-1 block uses the tip rotation R_ee (both
            # reduce to the legacy point path when R_tool = I).
            X_frame = X_ee @ ee_offset_Xtools[0]
            p_ee = X_frame[:3, 3]
            R_ee = X_frame[:3, :3]
            Jv = np.zeros((3, nv), dtype=np.float64)
            Jw = np.zeros((3, nv), dtype=np.float64)
            for j in chain_jids:
                S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
                if S.ndim == 1:
                    S = S.reshape(-1, 1)
                R_j = Xw[j][:3, :3]
                p_j = Xw[j][:3, 3]
                vinds = vinds_for(j)
                # Mimic joints fold into the mimicked joint's v-column scaled
                # by their multiplier; accumulate (not assign) so the proper
                # column gets BOTH the mimicked joint's direct contribution
                # AND the mimic joint's chain contribution (when both happen
                # to lie in the same chain).
                scale = mimic_scale(j)
                for c in range(S.shape[1]):
                    vi = vinds[c] if c < len(vinds) else vinds[-1]
                    ang_local = S[:3, c]
                    lin_local = S[3:6, c]
                    if np.linalg.norm(ang_local) > 0.5:    # rotational DOF
                        aw = R_j @ ang_local
                        Jw[:, vi] += scale * aw
                        Jv[:, vi] += scale * np.cross(aw, p_ee - p_j)
                    else:                                   # translational DOF
                        Jv[:, vi] += scale * (R_j @ lin_local)
            Einv = np.linalg.inv(E_world(R_ee))
            return np.vstack([Jv, Einv @ Jw])

        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        deePos_arr = []

        for jid in ee_jids:
            chain = sorted(self.robot.get_ancestors_by_id(jid)) + [jid]
            deePos_arr.append(jacobian_for_chain(chain, Xw[jid]))

        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            X_fixed = np.asarray(fj.get_transformation_matrix_hom(), dtype=np.float64)
            if fj.parent_name == -1:
                # EE rigidly attached to the world root: no DOFs in the chain,
                # gradient is zero (the offset-shifted position is constant).
                deePos_arr.append(np.zeros((6, nv), dtype=np.float64))
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                pid = parent.get_id()
                X_ee = Xw[pid] @ X_fixed
                chain = sorted(self.robot.get_ancestors_by_id(pid)) + [pid]
                deePos_arr.append(jacobian_for_chain(chain, X_ee))

        return deePos_arr

    # ------------------------------------------------------------------
    # General-frame geometric Jacobians + operational-space inertia (E2)
    # ------------------------------------------------------------------

    @staticmethod
    def _skew(v):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]], dtype=np.float64)

    def _frame_world_placement_and_chain(self, q):
        """Single forward-kinematics pass.

        Returns (Xw, q_arg) where Xw[j] is the 4x4 world homogeneous transform
        of joint j (mimic-aware) and q_arg(j) the joint's local q slice.
        """
        q = self._normalize_kinematics_q(q)
        n_joints = self.robot.get_num_joints()

        def q_arg(jid):
            return self.robot.q_for_joint(jid, q)

        Xw = [None] * n_joints
        for j in range(n_joints):
            X_local = np.asarray(self.robot.get_Xmat_hom_Func_by_id(j)(q_arg(j)),
                                 dtype=np.float64)
            par = self.robot.get_parent_id(j)
            Xw[j] = X_local if par == -1 else (Xw[par] @ X_local)
        return Xw, q_arg

    def _resolve_frame_joint(self, frame_name):
        """Map a frame/joint name to (target_joint_id, X_offset_hom).

        Accepts an articulated joint name (offset = identity) or a fixed-joint
        name (offset = the fixed joint's constant transform relative to its
        movable parent). Mirrors `end_effector_pose`'s target resolution.
        """
        joint = self.robot.get_joint_by_name(frame_name)
        if joint is not None:
            return joint.get_id(), np.eye(4, dtype=np.float64)
        fj = self.robot.get_fixed_joint_by_name(frame_name)
        if fj is None:
            raise ValueError("Could not find joint or fixed joint named: " + str(frame_name))
        X_fixed = np.asarray(fj.get_transformation_matrix_hom(), dtype=np.float64)
        if fj.parent_name == -1:
            return -1, X_fixed
        parent = self.robot.get_joint_by_name(fj.parent_name)
        return parent.get_id(), X_fixed

    def frame_jacobian(self, q, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        """6 x nv geometric Jacobian of a frame, ordered [linear(3); angular(3)].

        Matches pinocchio's `getFrameJacobian` / `getJointJacobian` for the
        three `pin.ReferenceFrame` choices:
          * ``WORLD``               -- spatial Jacobian at the world origin.
          * ``LOCAL``               -- twist expressed in the frame's body axes.
          * ``LOCAL_WORLD_ALIGNED`` -- at the frame origin, world-aligned axes.
        """
        reference_frame = str(reference_frame).upper()
        if reference_frame not in ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED"):
            raise ValueError("reference_frame must be LOCAL/WORLD/LOCAL_WORLD_ALIGNED")

        nv = self.robot.get_num_vel()
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_offset = self._resolve_frame_joint(frame_name)

        if target_id == -1:
            # frame rigidly attached to the world root: no DOFs in the chain.
            return np.zeros((6, nv), dtype=np.float64)

        X_frame = Xw[target_id] @ X_offset
        R_f = X_frame[:3, :3]
        p_f = X_frame[:3, 3]

        def vinds_for(jid):
            try:
                inds = self.robot.get_joint_index_v(jid)
            except Exception:
                inds = self.robot.get_joint_index_q(jid)
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def mimic_scale(jid):
            joint = self.robot.get_joint_by_id(jid)
            return joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0

        # World-frame geometric Jacobian at the frame ORIGIN, world axes.
        Jv = np.zeros((3, nv), dtype=np.float64)
        Jw = np.zeros((3, nv), dtype=np.float64)
        chain = sorted(self.robot.get_ancestors_by_id(target_id)) + [target_id]
        for j in chain:
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            R_j = Xw[j][:3, :3]
            p_j = Xw[j][:3, 3]
            vinds = vinds_for(j)
            scale = mimic_scale(j)
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang_local = S[:3, c]
                lin_local = S[3:6, c]
                if np.linalg.norm(ang_local) > 0.5:       # rotational DOF
                    aw = R_j @ ang_local
                    Jw[:, vi] += scale * aw
                    Jv[:, vi] += scale * np.cross(aw, p_f - p_j)
                else:                                      # translational DOF
                    Jv[:, vi] += scale * (R_j @ lin_local)

        if reference_frame == "LOCAL_WORLD_ALIGNED":
            return np.vstack([Jv, Jw])
        if reference_frame == "WORLD":
            # Spatial Jacobian: shift the reference point from the frame origin
            # to the world origin (angular part unchanged, linear gains p_f x w).
            return np.vstack([Jv + self._skew(p_f) @ Jw, Jw])
        # LOCAL: rotate both blocks into the frame's body axes.
        Rt = R_f.T
        return np.vstack([Rt @ Jv, Rt @ Jw])

    def frame_jacobian_dot(self, q, qd, frame_name=None,
                           reference_frame="LOCAL_WORLD_ALIGNED"):
        """ANALYTIC time derivative Jdot of `frame_jacobian` along v = qd, 6 x nv.

        Jdot = d/dt J(q(t)) with q evolving under v = qd. Differentiates the SAME
        world-axis geometric Jacobian `frame_jacobian` builds (NOT a finite
        difference). Each column i owns a joint j; the value column is

            Jw[:,i] = a_w ,  Jv[:,i] = a_w x (p_f - p_j)      (rotational)
            Jv[:,i] = R_j s_lin                                (translational)

        with a_w = R_j s_ang the world joint axis. Differentiating in time
        (Ṙ_j = w_j x R_j, ṗ = the world point velocities) gives

            Jẇ[:,i] = w_j x a_w
            Jv̇[:,i] = (w_j x a_w) x (p_f - p_j) + a_w x (v_f - v_j)   (rotational)
            Jv̇[:,i] = w_j x (R_j s_lin)                               (translational)

        where w_j / v_j are the world angular / origin-linear velocities of joint
        j's frame and v_f that of the target frame origin — obtained from a forward
        velocity sweep along the chain that is consistent WITH `frame_jacobian`'s
        own construction (same S, R_j, mimic scale). Matches pinocchio's
        `getFrameJacobianTimeVariation` for the three reference frames.

        (The central-FD form is retained as `_frame_jacobian_dot_fd` for the
        machine-precision cross-check.)"""
        qd = self._normalize_v_input(np.asarray(qd, dtype=np.float64))
        reference_frame = str(reference_frame).upper()
        if reference_frame not in ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED"):
            raise ValueError("reference_frame must be LOCAL/WORLD/LOCAL_WORLD_ALIGNED")

        nv = self.robot.get_num_vel()
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_offset = self._resolve_frame_joint(frame_name)
        if target_id == -1:
            return np.zeros((6, nv), dtype=np.float64)

        X_frame = Xw[target_id] @ X_offset
        R_f = X_frame[:3, :3]
        p_f = X_frame[:3, 3]

        def vinds_for(jid):
            try:
                inds = self.robot.get_joint_index_v(jid)
            except Exception:
                inds = self.robot.get_joint_index_q(jid)
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def mimic_scale(jid):
            joint = self.robot.get_joint_by_id(jid)
            return joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0

        chain = sorted(self.robot.get_ancestors_by_id(target_id)) + [target_id]

        # ── forward velocity sweep (world frame): per chain joint j, w_j (angular
        # velocity of frame j) and v_j (linear velocity of origin p_j). Rigid transport
        # from the parent + joint j's own DOF, mirroring frame_jacobian's a_w = R_j S. ──
        w_of = {}
        v_of = {}
        for j in chain:
            par = self.robot.get_parent_id(j)
            R_j = Xw[j][:3, :3]
            p_j = Xw[j][:3, 3]
            if par in w_of:
                w_j = w_of[par].copy()
                v_j = v_of[par] + np.cross(w_of[par], p_j - Xw[par][:3, 3])
            else:  # chain root (fixed base: world; floating: the free-flyer carries qd[0:6])
                w_j = np.zeros(3, dtype=np.float64)
                v_j = np.zeros(3, dtype=np.float64)
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            vinds = vinds_for(j)
            scale = mimic_scale(j)
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang_local = S[:3, c]
                lin_local = S[3:6, c]
                if np.linalg.norm(ang_local) > 0.5:      # rotational: axis through p_j
                    w_j = w_j + scale * qd[vi] * (R_j @ ang_local)
                else:                                    # translational: moves p_j
                    v_j = v_j + scale * qd[vi] * (R_j @ lin_local)
            w_of[j] = w_j
            v_of[j] = v_j

        # target frame velocities (rigidly attached through X_offset)
        w_f = w_of[target_id]
        v_f = v_of[target_id] + np.cross(w_of[target_id], p_f - Xw[target_id][:3, 3])

        # ── column-wise value J + its time derivative (LWA basis) ──
        Jv = np.zeros((3, nv), dtype=np.float64)
        Jw = np.zeros((3, nv), dtype=np.float64)
        Jvd = np.zeros((3, nv), dtype=np.float64)
        Jwd = np.zeros((3, nv), dtype=np.float64)
        for j in chain:
            R_j = Xw[j][:3, :3]
            p_j = Xw[j][:3, 3]
            w_j = w_of[j]
            v_j = v_of[j]
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            vinds = vinds_for(j)
            scale = mimic_scale(j)
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang_local = S[:3, c]
                lin_local = S[3:6, c]
                if np.linalg.norm(ang_local) > 0.5:      # rotational DOF
                    aw = R_j @ ang_local
                    awd = np.cross(w_j, aw)              # d/dt a_w
                    Jw[:, vi] += scale * aw
                    Jv[:, vi] += scale * np.cross(aw, p_f - p_j)
                    Jwd[:, vi] += scale * awd
                    Jvd[:, vi] += scale * (np.cross(awd, p_f - p_j) + np.cross(aw, v_f - v_j))
                else:                                    # translational DOF
                    lw = R_j @ lin_local
                    Jv[:, vi] += scale * lw
                    Jvd[:, vi] += scale * np.cross(w_j, lw)

        if reference_frame == "LOCAL_WORLD_ALIGNED":
            return np.vstack([Jvd, Jwd])
        if reference_frame == "WORLD":
            # J_world_v = Jv + skew(p_f) Jw ; d/dt adds skew(v_f) Jw + skew(p_f) Jwd.
            return np.vstack([Jvd + self._skew(v_f) @ Jw + self._skew(p_f) @ Jwd, Jwd])
        # LOCAL: rows rotated into body axes; d/dt(R_f^T) = -R_f^T skew(w_f).
        Rt = R_f.T
        Rtd = -Rt @ self._skew(w_f)
        return np.vstack([Rtd @ Jv + Rt @ Jvd, Rtd @ Jw + Rt @ Jwd])

    def _frame_jacobian_dot_fd(self, q, qd, frame_name=None,
                               reference_frame="LOCAL_WORLD_ALIGNED", step=1e-6):
        """Central-FD cross-check for the analytic `frame_jacobian_dot`: differences
        the analytic `frame_jacobian` along the Lie-group integrator flow."""
        qd = self._normalize_v_input(np.asarray(qd, dtype=np.float64))
        q_plus = self.integrate(q, step * qd)
        q_minus = self.integrate(q, -step * qd)
        Jp = self.frame_jacobian(q_plus, frame_name, reference_frame)
        Jm = self.frame_jacobian(q_minus, frame_name, reference_frame)
        return (Jp - Jm) / (2.0 * step)

    def osc_inertia(self, q, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        """Operational-space (task) inertia Lambda = (J M^{-1} J^T)^{-1}, 6 x 6.

        Composes the frame Jacobian with the inverse joint-space mass matrix
        (`self.minv`) and inverts the resulting 6x6 task-space matrix."""
        J = self.frame_jacobian(q, frame_name, reference_frame)
        Minv = np.asarray(self.minv(q), dtype=np.float64)
        task = J @ Minv @ J.T
        return np.linalg.inv(task)

    """
    End Effector Pose Hessian
    """
    def end_effector_pose_hessian(self, q, offsets = None, ee_joint_names = None):
        """Hessian d^2(pose)/dv^2 of the end-effector pose w.r.t. generalized velocity v.

        Convention: **d^2/dv^2 (tangent space)**, matching pinocchio. Output is a
        list of 6 x nv x nv tensors (one per ee). For fixed-base nv == nq so the
        shape is unchanged from earlier versions; for floating-base the (nv x nv)
        block now indexes spatial twist components, not quaternion derivatives.

        Implementation: central-difference FD of the (now-correct) d/dv pose
        Jacobian on the Lie-group integrator `self.integrate(q, h*e_i)`. This
        reuses the geometric-Jacobian gradient (machine-precision vs the proven
        prototype + pinocchio backend) and SYMMETRIZES the result. FD is fine
        for a reference oracle; the (separate) GPU codegen path is the same
        FD-on-Jacobian (the analytic d/dv Hessian on GPU is on the backlog --
        see HANDOFF.md A.1 + docs/d2ee_analytic_derivation.md).

        Parameters
        ----------
        q : numpy.ndarray
            Generalized position (floating base: [xyz, quat_xyzw, joints]).
        offsets : list, optional
            Per-ee point offsets [x, y, z, 1] (currently the first offset is
            applied to every ee, matching the gradient).
        ee_joint_names : list of str, optional
            Joint names to use as end-effectors; defaults to the robot leaves.

        Returns
        -------
        list of numpy.ndarray
            Per end-effector 6 x nv x nv Hessian of d^2(pose)/dv^2.
        """
        q = self._normalize_kinematics_q(q)
        nv = self.robot.get_num_vel()
        h = 1e-5

        # gradient base point + perturbed points; pose_gradient handles offsets/ee selection
        ee_offsets = self._normalize_ee_offsets(offsets)
        J_plus_per_i = []
        J_minus_per_i = []
        for i in range(nv):
            v = np.zeros(nv, dtype=np.float64); v[i] = h
            q_plus = self.integrate(q, v)
            q_minus = self.integrate(q, -v)
            J_plus_per_i.append(
                self.end_effector_pose_gradient(q_plus, ee_joint_names=ee_joint_names,
                                                ee_offsets=offsets if offsets is not None else None)
            )
            J_minus_per_i.append(
                self.end_effector_pose_gradient(q_minus, ee_joint_names=ee_joint_names,
                                                ee_offsets=offsets if offsets is not None else None)
            )

        # Each J_*_per_i[i] is a list[ee] of (6, nv) arrays. Stack into Hessians.
        num_ees = len(J_plus_per_i[0])
        d2eePos_arr = []
        for ee_idx in range(num_ees):
            H = np.zeros((6, nv, nv), dtype=np.float64)
            for i in range(nv):
                Jp = J_plus_per_i[i][ee_idx]
                Jm = J_minus_per_i[i][ee_idx]
                # dJ/dv_i -> H[:, :, i]; rows 3..5 are angle derivatives (still
                # continuous since rpy branch cuts live in the POSE not its
                # Jacobian) so no wrap-handling needed at this step.
                H[:, :, i] = (Jp - Jm) / (2.0 * h)
            # Symmetrize: analytic d^2/dv_j dv_i = d^2/dv_i dv_j; FD won't
            # be exact, so average to suppress per-pair noise.
            H = 0.5 * (H + np.transpose(H, axes=(0, 2, 1)))
            d2eePos_arr.append(H)
        return d2eePos_arr

    def end_effector_pose_hessian_analytic(self, q, offsets=None, ee_joint_names=None):
        """Analytic d^2(pose)/dv^2 of the end-effector pose w.r.t. generalized velocity.

        Convention: d^2/dv^2 (TANGENT, pinocchio convention). Pose is [xyz; rpy]
        with R = Rz(yaw) Ry(pitch) Rx(roll). Output is a list of (6, nv, nv)
        tensors (one per ee).

        Algorithm: per-chain second-order Taylor expansion of the EE world
        transform T_ee(v) along the chain joints. Each chain joint contributes
        an isolated local perturbation Delta_j(v_j) (revolute exp, prismatic
        translation, or SE(3) free-flyer exp). The chain composition
        M(v) = X_0 Delta_0 X_1 Delta_1 ... X_k Delta_k is then differentiated to
        second order at v=0 by piggy-backing on left/right cumulative products
        L_a, R_a around each joint. This handles ALL DOF interaction patterns
        uniformly: (a) DOFs in separate joints (proximal/distal in chain), (b)
        intra-joint pairs of the SAME multi-DOF joint (notably the SE(3)
        free-flyer at jid=0), and (c) cross-joint pairs where one side is a
        multi-DOF joint. The rpy rows are obtained by the analytic chain rule
        d^2 rpy / dv_i dv_j = (dE^{-1}/dv_j) J_w[:, i] + E^{-1} dJ_w[:, i]/dv_j,
        where dJ_w/dv is the world-angular kinematic Hessian computed from
        skew^{-1}(d^2 R T^{-1}) on the Taylor-expanded rotation block.

        Validated against the FD oracle (end_effector_pose_hessian) AND against
        pinocchio's analytic getJointKinematicHessian(LOCAL_WORLD_ALIGNED) on the
        full manifest fleet (iiwa14/go2/g1/h1_2/fr3/rizon4/gen3/fetch/baxter,
        fixed + floating, incl. the mimic robots fr3/h1_2) to the FD noise floor
        (~1e-11 analytic-vs-FD, ~1e-5 vs pinocchio's FD-step). The earlier
        orientation-hessian bug lived in the retired analytic d^2/dq^2 path; this
        chain-composition d^2/dv^2 derivation is correct fleet-wide.

        Parameters
        ----------
        q : numpy.ndarray
            Generalized position (floating base: [xyz, quat_xyzw, joints]).
        offsets : list, optional
            Per-ee point offsets [x, y, z, 1]; the first offset is applied to
            every ee, matching `end_effector_pose_gradient`.
        ee_joint_names : list of str, optional
            Joint names to use as end-effectors; defaults to leaf joints.

        Returns
        -------
        list of numpy.ndarray
            Per end-effector (6, nv, nv) Hessian of d^2(pose)/dv^2.
        """
        q = self._normalize_kinematics_q(q)
        ee_offsets = self._normalize_ee_offsets(offsets)
        nv = self.robot.get_num_vel()
        n_joints = self.robot.get_num_joints()

        # ----- helpers (mirroring end_effector_pose_gradient) -----
        def vinds_for(jid):
            try:
                inds = self.robot.get_joint_index_v(jid)
            except Exception:
                inds = self.robot.get_joint_index_q(jid)
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def q_arg(jid):
            # Mimic-aware: feeds the joint's transform the value the URDF
            # mimic relation prescribes (`multiplier * q[target] + offset`).
            return self.robot.q_for_joint(jid, q)

        def mimic_scale(jid):
            joint = self.robot.get_joint_by_id(jid)
            return joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0

        # one forward-kinematics pass: world transform of every joint
        Xw = [None] * n_joints
        Xlocal = [None] * n_joints
        for j in range(n_joints):
            Xlocal[j] = np.asarray(
                self.robot.get_Xmat_hom_Func_by_id(j)(q_arg(j)),
                dtype=np.float64,
            )
            par = self.robot.get_parent_id(j)
            Xw[j] = Xlocal[j] if par == -1 else (Xw[par] @ Xlocal[j])

        # ----- E(rpy) rate->omega map and its rpy-derivatives (closed form) -----
        # For R = Rz(yaw) Ry(pitch) Rx(roll), omega_world = E(rpy) * [roll_d; pitch_d; yaw_d]:
        #   E = [[cy*cp, -sy, 0], [sy*cp, cy, 0], [-sp, 0, 1]]
        def rpy_from_R(R_ee):
            roll = np.arctan2(R_ee[2, 1], R_ee[2, 2])
            pitch = np.arctan2(-R_ee[2, 0],
                               np.sqrt(R_ee[2, 2] * R_ee[2, 2] + R_ee[2, 1] * R_ee[2, 1]))
            yaw = np.arctan2(R_ee[1, 0], R_ee[0, 0])
            return roll, pitch, yaw

        def E_and_deriv(rpy):
            roll, pitch, yaw = rpy
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            E = np.array([[cy * cp, -sy, 0.0],
                          [sy * cp,  cy, 0.0],
                          [-sp,     0.0, 1.0]], dtype=np.float64)
            # ∂E/∂roll = 0 (E does not depend on roll)
            dE_droll = np.zeros((3, 3), dtype=np.float64)
            dE_dpitch = np.array([[-cy * sp, 0.0, 0.0],
                                  [-sy * sp, 0.0, 0.0],
                                  [-cp,      0.0, 0.0]], dtype=np.float64)
            dE_dyaw = np.array([[-sy * cp, -cy, 0.0],
                                [ cy * cp, -sy, 0.0],
                                [ 0.0,      0.0, 0.0]], dtype=np.float64)
            return E, dE_droll, dE_dpitch, dE_dyaw

        # ----- local perturbation A_local (4x4 dDelta/dv) and B_local (4x4 d^2 Delta/dv_a dv_b) -----
        # For revolute around axis a (body frame): A = [[ [a]_x, 0 ], [ 0, 0 ]],  B = [[ [a]_x^2, 0], [0, 0]]
        # For prismatic along axis a (body frame): A = [[ 0, a ], [0, 0]],  B = 0
        # For floating base (6 DOFs, body frame, v=[v_lin(3); omega(3)]):
        #   - linear DOF c=0..2 (axis e_c): A = [[0, e_c], [0, 0]],  B(c, c') = 0
        #   - angular DOF c=3..5 (axis e_{c-3}): A = [[ [e_{c-3}]_x, 0 ], [ 0, 0 ]]
        #   - mixed lin(a)/ang(b): B(a, b) = [[0, 0.5 * (e_{b-3} x e_a)], [0, 0]]
        #   - mixed ang(a)/ang(b): B(a, b) = [[ 0.5 * ([e_{a-3}]_x [e_{b-3}]_x + [e_{b-3}]_x [e_{a-3}]_x), 0 ], [ 0, 0 ]]
        def axis_skew(a):
            x, y, z = a
            return np.array([[0.0, -z,  y],
                             [z,    0.0, -x],
                             [-y,   x,   0.0]], dtype=np.float64)

        def A_for_dof(j, c):
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            ang_local = S[:3, c]
            lin_local = S[3:6, c]
            A = np.zeros((4, 4), dtype=np.float64)
            if np.linalg.norm(ang_local) > 0.5:
                A[:3, :3] = axis_skew(ang_local)
            else:
                A[:3, 3] = lin_local
            return A

        def B_for_dofpair(j, c_a, c_b):
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            ang_a = S[:3, c_a]; lin_a = S[3:6, c_a]
            ang_b = S[:3, c_b]; lin_b = S[3:6, c_b]
            a_is_rot = np.linalg.norm(ang_a) > 0.5
            b_is_rot = np.linalg.norm(ang_b) > 0.5
            B = np.zeros((4, 4), dtype=np.float64)
            if a_is_rot and b_is_rot:
                Sa = axis_skew(ang_a)
                Sb = axis_skew(ang_b)
                B[:3, :3] = 0.5 * (Sa @ Sb + Sb @ Sa)
            elif a_is_rot and not b_is_rot:
                # ang(a), lin(b): mixed contribution = 0.5 * (e_a x lin_b) translation column
                B[:3, 3] = 0.5 * np.cross(ang_a, lin_b)
            elif (not a_is_rot) and b_is_rot:
                B[:3, 3] = 0.5 * np.cross(ang_b, lin_a)
            # else: lin-lin -> 0
            return B

        def skew_inv(M):
            """Extract a 3-vector from a 3x3 (approximately) skew-symmetric matrix."""
            return 0.5 * np.array([M[2, 1] - M[1, 2],
                                   M[0, 2] - M[2, 0],
                                   M[1, 0] - M[0, 1]], dtype=np.float64)

        def hessian_for_chain(chain_jids, X_ee, ee_offset_col):
            """Compute the (6, nv, nv) pose Hessian for a given chain & EE transform.

            X_ee is the WORLD transform of the EE FRAME (used for rpy extraction).
            ee_offset_col is the 4x1 point in the LAST CHAIN JOINT's frame at
            which we evaluate the position (so the chain-product M(v) is
            differentiated, then dotted with this constant offset). For a fixed-
            joint EE, the caller pre-applies the fixed transform: pass
            chain ending at the parent of the fixed joint, X_ee = Xw[pid] @ X_fixed,
            and ee_offset_col = X_fixed @ user_offset.
            """
            R_ee0 = X_ee[:3, :3]  # used for rpy and E(rpy)
            p_ee0 = (X_ee @ ee_offset_col).reshape(-1)[:3]

            # Build the per-joint X_local list for this chain. Each "block" b
            # encapsulates joint chain_jids[b] with: prefix_to_b @ X_local[chain_jids[b]] @ Delta_b @ ...
            # We need L_a = prefix BEFORE Delta_a (i.e. up to and including X_local for joint a).
            n_chain = len(chain_jids)
            X_chain_local = [Xlocal[j] for j in chain_jids]
            # The chain starts at root joint chain_jids[0], whose parent might be -1 or a fixed root.
            # The local transforms in X_chain_local already encode parent->joint placement.

            # L[a] = product of X_chain_local[0..a] @ Delta[0..a-1](0)  — but Delta(0) = I, so:
            #   L[a] = X_chain_local[0] @ X_chain_local[1] @ ... @ X_chain_local[a]
            # R[a] = X_chain_local[a+1] @ ... @ X_chain_local[k] @ ee_offset_handling
            # i.e. M(0) = L[a] @ R[a] for any a, and pp = M(0) @ ee_offset = X_ee @ ee_offset.
            # We also want "between" prefixes: P[a, b] for a < b is the chunk
            #   between Delta_a and Delta_b, i.e. X_chain_local[a+1..b].
            L = [None] * n_chain
            R_post = [None] * n_chain
            # Prefixes:
            acc = np.eye(4)
            for a in range(n_chain):
                acc = acc @ X_chain_local[a]
                L[a] = acc.copy()
            # Suffixes (after the joint-a Delta):
            acc = np.eye(4)
            R_post[n_chain - 1] = acc.copy()
            for a in range(n_chain - 2, -1, -1):
                acc = X_chain_local[a + 1] @ acc
                R_post[a] = acc.copy()
            # Sanity: L[n-1] should equal X_ee (the EE world transform).
            # Note: for floating base chain[0] = 0, X_chain_local[0] = Xlocal[0] = T_base_world; OK.

            # Compute the "between" matrices P[a, b] = X_{a+1} ... X_b for a < b.
            # Stored as P_between[(a, b)] for a < b. We won't actually need all
            # pairs precomputed — we can do it on-the-fly via L and L_inv.
            # But L_inv is unstable for SE(3) (not just invertible by skipping —
            # the L[a] matrices are SE(3), so their inverse is closed-form).
            def se3_inv(T):
                Ti = np.eye(4)
                Rt = T[:3, :3].T
                Ti[:3, :3] = Rt
                Ti[:3, 3] = -Rt @ T[:3, 3]
                return Ti

            # Map each chain block to (vi, c, scale) entries. A v-index can
            # appear in MULTIPLE blocks when the chain contains a mimic joint
            # and its mimicked target (their contributions both fold into the
            # same column with the multiplier scaling).
            block_entries = []  # block_entries[a] = list of (vi, c, scale)
            vi_to_blocks = {}   # vi -> list of (a, c, scale)
            for a, j in enumerate(chain_jids):
                S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
                if S.ndim == 1:
                    S = S.reshape(-1, 1)
                vinds = vinds_for(j)
                scale = mimic_scale(j)
                this_entries = []
                for c in range(S.shape[1]):
                    vi = vinds[c] if c < len(vinds) else vinds[-1]
                    this_entries.append((vi, c, scale))
                    vi_to_blocks.setdefault(vi, []).append((a, c, scale))
                block_entries.append(this_entries)
            chain_dofs = sorted(vi_to_blocks.keys())

            # First derivatives dM/dv_i = sum_{(a, c, scale) for i} scale * L[a] @ A_local @ R_post[a]
            dM = {i: np.zeros((4, 4), dtype=np.float64) for i in chain_dofs}
            for i in chain_dofs:
                for a, c, scale in vi_to_blocks[i]:
                    A = A_for_dof(chain_jids[a], c)
                    dM[i] += scale * (L[a] @ A @ R_post[a])

            # Second derivatives d2M/dv_i dv_j follow the product rule across
            # all (a, c, s_i) for i and (b, c', s_j) for j. With at most one
            # block per non-mimic vi and a mimic joint adding a second block,
            # we just sum the per-pair contributions.
            d2M = np.zeros((len(chain_dofs), len(chain_dofs), 4, 4), dtype=np.float64)
            chain_dof_index = {vi: idx for idx, vi in enumerate(chain_dofs)}

            # Precompute between products P_ab for a <= b via L / Linv.
            Linv = [se3_inv(L[a]) for a in range(n_chain)]

            def pair_contribution(a, c_i, b, c_j):
                """4x4 contribution from block-pair (a, b) with local-cols (c_i, c_j).
                Caller multiplies in the (s_i * s_j) scaling.
                """
                if a == b:
                    if c_i <= c_j:
                        B = B_for_dofpair(chain_jids[a], c_i, c_j)
                    else:
                        B = B_for_dofpair(chain_jids[a], c_j, c_i)
                    return L[a] @ B @ R_post[a]
                if a < b:
                    A_prox = A_for_dof(chain_jids[a], c_i)
                    A_dist = A_for_dof(chain_jids[b], c_j)
                    return L[a] @ A_prox @ (Linv[a] @ L[b]) @ A_dist @ R_post[b]
                # a > b: same as (b, a) by chain-order symmetry of the product
                A_prox = A_for_dof(chain_jids[b], c_j)
                A_dist = A_for_dof(chain_jids[a], c_i)
                return L[b] @ A_prox @ (Linv[b] @ L[a]) @ A_dist @ R_post[a]

            for ii, i in enumerate(chain_dofs):
                for jj, j in enumerate(chain_dofs):
                    acc = np.zeros((4, 4), dtype=np.float64)
                    for (a, c_i, s_i) in vi_to_blocks[i]:
                        for (b, c_j, s_j) in vi_to_blocks[j]:
                            acc += s_i * s_j * pair_contribution(a, c_i, b, c_j)
                    d2M[ii, jj] = acc

            # ----- Extract xyz Hessian: H_xyz[:, i, j] = (d2M @ ee_offset)[:3] -----
            H_xyz = np.zeros((3, nv, nv), dtype=np.float64)
            # First also extract first derivatives of p_ee for use in rpy term.
            ee_off_col = ee_offset_col.reshape(4)
            dp = np.zeros((3, nv), dtype=np.float64)  # d p_ee / d v_i
            for i in chain_dofs:
                dp[:, i] = (dM[i] @ ee_off_col)[:3]
            for ii, i in enumerate(chain_dofs):
                for jj, j in enumerate(chain_dofs):
                    H_xyz[:, i, j] = (d2M[ii, jj] @ ee_off_col)[:3]

            # ----- Extract dR/dv_i and d2R/dv_i dv_j (3x3 blocks) -----
            dR = np.zeros((nv, 3, 3), dtype=np.float64)
            for i in chain_dofs:
                dR[i] = dM[i][:3, :3]
            d2R = np.zeros((nv, nv, 3, 3), dtype=np.float64)
            for ii, i in enumerate(chain_dofs):
                for jj, j in enumerate(chain_dofs):
                    d2R[i, j] = d2M[ii, jj][:3, :3]

            # ----- Build the angular Jacobian J_w in world frame: -----
            # The world angular velocity of the EE link == that of the last
            # chain joint (rigid offset doesn't add angular velocity), so use
            # the chain rotation (L[n-1])[:3, :3] = R_chain0, NOT R_ee0 (which
            # might include an extra fixed-joint rotation in the fixed-EE case).
            R_chain0 = L[n_chain - 1][:3, :3]
            R0T = R_chain0.T
            J_w = np.zeros((3, nv), dtype=np.float64)
            for i in chain_dofs:
                J_w[:, i] = skew_inv(dR[i] @ R0T)

            # ----- World-angular kinematic Hessian: dJ_w[:, i]/dv_j -----
            # From R(v) ≈ R(0) · exp(omega_per_unit_v · v + ...), we have
            #   dR/dv_i = [J_w_i]_x · R(0).
            # Differentiating w.r.t. v_j (and using d(R(0))/dv_j = 0; R(0) is the
            # base point, not the curve):
            #   d2R/(dv_i dv_j) = [dJ_w_i/dv_j]_x · R(0) + [J_w_i]_x · dR/dv_j
            #                   = [dJ_w_i/dv_j]_x · R(0) + [J_w_i]_x · [J_w_j]_x · R(0)
            # ⇒  [dJ_w_i/dv_j]_x = d2R/(dv_i dv_j) · R(0)^T - [J_w_i]_x · [J_w_j]_x
            H_w = np.zeros((3, nv, nv), dtype=np.float64)  # H_w[:, i, j] = dJ_w[:, i]/dv_j
            for ii, i in enumerate(chain_dofs):
                Jw_i_x = axis_skew(J_w[:, i])
                for jj, j in enumerate(chain_dofs):
                    Jw_j_x = axis_skew(J_w[:, j])
                    M_skew = d2R[i, j] @ R0T - Jw_i_x @ Jw_j_x
                    H_w[:, i, j] = skew_inv(M_skew)

            # ----- rpy chain rule -----
            rpy = rpy_from_R(R_ee0)
            E, dE_dr, dE_dp, dE_dy = E_and_deriv(rpy)
            Einv = np.linalg.inv(E)
            dE_drpy = [dE_dr, dE_dp, dE_dy]

            # drpy/dv_i = Einv @ J_w[:, i]
            drpy_dv = np.zeros((3, nv), dtype=np.float64)
            for i in chain_dofs:
                drpy_dv[:, i] = Einv @ J_w[:, i]

            # H_rpy[:, i, j] = (dEinv/dv_j) J_w[:, i] + Einv @ dJ_w[:, i]/dv_j
            # dEinv/dv_j = -Einv @ (sum_k (dE/drpy_k) (drpy_k/dv_j)) @ Einv
            H_rpy = np.zeros((3, nv, nv), dtype=np.float64)
            for ii, i in enumerate(chain_dofs):
                for jj, j in enumerate(chain_dofs):
                    dE_dvj = (dE_drpy[0] * drpy_dv[0, j]
                              + dE_drpy[1] * drpy_dv[1, j]
                              + dE_drpy[2] * drpy_dv[2, j])
                    dEinv_dvj = -Einv @ dE_dvj @ Einv
                    H_rpy[:, i, j] = dEinv_dvj @ J_w[:, i] + Einv @ H_w[:, i, j]

            # Symmetrize over (i, j) — H_rpy from the formula is per-pair but
            # the true Hessian of a scalar w.r.t. v is symmetric in (i, j).
            H_rpy = 0.5 * (H_rpy + np.transpose(H_rpy, axes=(0, 2, 1)))
            # H_xyz is already symmetric by construction (d2M[ii, jj] == d2M[jj, ii]
            # is enforced by our case logic above), but symmetrize defensively.
            H_xyz = 0.5 * (H_xyz + np.transpose(H_xyz, axes=(0, 2, 1)))

            return np.concatenate([H_xyz, H_rpy], axis=0)

        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        d2eePos_arr = []
        ee_off_col = ee_offsets[0]

        for jid in ee_jids:
            chain = sorted(self.robot.get_ancestors_by_id(jid)) + [jid]
            d2eePos_arr.append(hessian_for_chain(chain, Xw[jid], ee_off_col))

        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            X_fixed = np.asarray(fj.get_transformation_matrix_hom(), dtype=np.float64)
            if fj.parent_name == -1:
                d2eePos_arr.append(np.zeros((6, nv, nv), dtype=np.float64))
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                pid = parent.get_id()
                X_ee = Xw[pid] @ X_fixed
                chain = sorted(self.robot.get_ancestors_by_id(pid)) + [pid]
                # When the EE is a fixed offset from joint pid, the chain ends
                # at pid and the "ee_offset" we apply to the chain product L[k]
                # is X_fixed @ ee_off_col (i.e. translate further into the
                # fixed-joint frame before applying the user-supplied offset).
                ee_off_after_fixed = (X_fixed @ ee_off_col).reshape(4, 1)
                # Note: hessian_for_chain uses X_ee for R_ee0 / p_ee0 baseline
                # which we pre-multiply by X_fixed implicitly via X_ee.
                d2eePos_arr.append(hessian_for_chain(chain, X_ee, ee_off_after_fixed))

        return d2eePos_arr

    def apply_external_forces(self, f_in, f_ext):
        """Subtract per-body external forces from the internal force array.

        Convention (matches GATO `*_fext.cuh` and pinocchio `fext`): `f_ext`
        is a per-body spatial force expressed in each link's LOCAL frame,
        ordered [angular(n); linear(f)] (same layout as the rows of `f`),
        and it is SUBTRACTED from the per-body force:

            f[:, i] -= f_ext[i]

        No coordinate transform is applied: the external force is already
        given in the body's local frame, which is the frame the RNEA force
        recursion / ABA bias accumulates in.

        Parameters
        ----------
        f_in : numpy.ndarray
            (6, NB) per-body spatial forces to be corrected in place.
        f_ext : sequence
            Per-body external spatial forces (each a length-6 local-frame
            vector). An empty `f_ext` is a no-op (byte-identical no-fext path).

        Returns
        -------
        numpy.ndarray
            The corrected `(6, NB)` force array.
        """
        f_out = f_in
        if f_ext is None or len(f_ext) == 0:
            return f_out
        NB = self.robot.get_num_bodies()
        for curr_id in range(NB):
            fe = f_ext[curr_id]
            if fe is None:
                continue
            fe = np.asarray(fe, dtype=np.float64).reshape(-1)
            if fe.shape[0] == 6:
                f_out[:, curr_id] -= fe
        return f_out

    def _mimic_multiplier(self, jid):
        """Return the URDF mimic multiplier for `jid` (1.0 for non-mimic).

        Centralizes the mimic-scaling pattern used in dynamics passes:
        every place that reads `qd[get_joint_index_v(jid)]` (or qdd) for a
        mimic joint must scale by this multiplier (since the mimic's
        generalized velocity is `multiplier * v_target`); every place
        that writes to `tau[get_joint_index_v(jid)]` for a mimic joint
        must also scale by it (since the mimic's torque contribution
        folds into the target column with the same multiplier).
        """
        joint = self.robot.get_joint_by_id(jid)
        if getattr(joint, "is_mimic", False):
            return float(joint.get_mimic_multiplier())
        return 1.0

    def _mimic_offset(self, jid):
        joint = self.robot.get_joint_by_id(jid)
        if getattr(joint, "is_mimic", False):
            return float(joint.get_mimic_offset())
        return 0.0

    @staticmethod
    def _Sq(S, qvec):
        """Project a joint coordinate-rate vector through the motion subspace S.

        Returns the spatial motion 6-vector ``S @ qvec`` for ANY joint:

        - 1-DOF cardinal/skew joint: ``S`` is a flat 6-vector and ``qvec`` is a
          scalar, so this is the elementwise ``S * qvec`` (a 6-vector).
        - Multi-DOF joint (floating root 6x6, spherical 6x3, planar 6x3): ``S``
          is a 6xN matrix and ``qvec`` is an N-vector, so this is the true
          matrix-vector product (a 6-vector).

        Unifies the previous floating-root-only ``matmul`` special-case so any
        mid-chain multi-column joint (spherical/planar) gets the correct
        contraction instead of a broadcast. Always returns a flat (6,) array.
        """
        S = np.asarray(S, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        if S.ndim == 2 and S.shape[1] > 1:
            return (S @ qvec.reshape(-1)).reshape(6)
        return (S.reshape(6) * float(qvec)).reshape(6)

    def _robot_has_multidof_nonfloating_joint(self):
        """True if any NON-root joint has dof>1 (spherical/planar mid-chain).

        The floating root (jid 0) is excluded: its 6-DOF block is handled by the
        established floating-base recursion. Used to route fixed-base ABA through
        the reduced-model identity (its scalar-only recursion can't handle a
        multi-column motion subspace).
        """
        for joint in self.robot.get_joints_ordered_by_id():
            if joint.get_id() == 0 and self.robot.floating_base:
                continue
            if joint.get_num_dof() > 1:
                return True
        return False

    def inverse_dynamics_fpass(self, q, qd, qdd=None, GRAVITY=-9.81, f_ext=None):
        """Perform the forward pass of the Recursive Newton-Euler Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (v, a, f) : tuple
            Spatial velocities, accelerations, and forces of each link.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        v = np.zeros((6, NB))
        a = np.zeros((6, NB))
        f = np.zeros((6, NB))
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY  # a_base is gravity vec

        # forward pass
        for curr_id in range(NB):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            # Mimic-aware: feed the joint's transform `multiplier * q[target]
            # + offset` so a URDF mimic joint correctly sees the mimicked
            # joint's coordinate (with the prescribed scaling).
            _q = self.robot.q_for_joint(curr_id, q)
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
            # compute v and a
            if parent_id == -1:  # parent is fixed base or world
                # v_base is zero so v[:,ind] remains 0
                if self.robot.floating_base:
                    a[:, curr_id] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, curr_id] = np.matmul(Xmat, gravity_vec)
            else:
                v[:, curr_id] = np.matmul(Xmat, v[:, parent_id])
                a[:, curr_id] = np.matmul(Xmat, a[:, parent_id])
            inds_v = self.robot.get_joint_index_v(curr_id)
            mimic_scale = self._mimic_multiplier(curr_id)
            _qd = mimic_scale * qd[inds_v]

            vJ = self._Sq(S, _qd)
            v[:, curr_id] += vJ
            a[:, curr_id] += self.mxS(vJ, v[:, curr_id])
            if qdd is not None:
                _qdd = mimic_scale * qdd[inds_v]
                aJ = self._Sq(S, _qdd)
                a[:, curr_id] += aJ
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:, curr_id] = np.matmul(Imat, a[:, curr_id]) + self.vxIv(v[:, curr_id], Imat)

        # subtract local-frame external forces from the per-body force
        # (single subtract site, mirrors the GATO/CUDA `f -= f_ext` convention)
        f = self.apply_external_forces(f, f_ext)

        return (v, a, f)

    def _joint_dynamics_bias(self, qd):
        """Per-v-slot joint-dynamics bias tau += damping*qd + friction*sign(qd).

        Viscous damping and Coulomb friction are joint-local generalized forces
        that oppose motion. They are returned as an nv-vector added to the RNEA
        bias (inverse_dynamics) and subtracted from the available torque in ABA.

        Returns the zero vector when no joint declares damping/friction (the
        common case), so this is numerically a no-op there. `qd` is the
        already-normalized internal nv-velocity vector.
        """
        n = self.robot.get_num_vel()
        bias = np.zeros(n)
        if not self.use_joint_dynamics:
            return bias
        has_damping = self.robot.robot_has_joint_damping()
        has_friction = self.robot.robot_has_joint_friction()
        if not (has_damping or has_friction):
            return bias
        # Iterate bodies and fold each joint's coefficient into its v-slot.
        # Mimic joints share their target's slot; their damping/friction add
        # into that shared slot (scaled by the mimic multiplier, matching the
        # alpha-weighted reduction used throughout the reduced-model path).
        for jid in range(self.robot.get_num_bodies()):
            b = float(self.robot.get_damping_by_id(jid)) if has_damping else 0.0
            fr = float(self.robot.get_friction_by_id(jid)) if has_friction else 0.0
            if b == 0.0 and fr == 0.0:
                continue
            idx = self.robot.get_joint_index_v(jid)
            alpha = self._mimic_multiplier(jid)
            idx_list = idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]
            for k in idx_list:
                qd_k = qd[k]
                bias[k] += alpha * (b * qd_k + fr * np.sign(qd_k))
        return bias

    def inverse_dynamics_bpass(self, q, f):
        """Perform the backward pass of the Recursive Newton-Euler Algorithm.

        Parameters
        ----------
        f : numpy.ndarray
            6N-element internal spatial forces per link.

        Returns
        -------
        (c, f) : tuple
            Generalized forces and updated internal spatial forces.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        m = self.robot.get_num_vel()
        c = np.zeros(m)

        # backward pass
        for curr_id in range(NB - 1, -1, -1):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            inds_f = self.robot.get_joint_index_f(curr_id)
            mimic_scale = self._mimic_multiplier(curr_id)
            # compute c. Mimic joints fold into the mimicked joint's slot
            # scaled by the multiplier, so use ACCUMULATE (not assign) and
            # apply the scale: c[inds_f] += scale * S^T @ f[:, curr_id].
            # For non-mimic joints this is `+= 1.0 * (...)`, which is
            # bit-identical to the legacy assign because each non-mimic
            # joint's inds_f is hit exactly once over the bpass.
            c[inds_f] = c[inds_f] + mimic_scale * np.matmul(np.transpose(S), f[:, curr_id])
            # update f if applicable
            if parent_id != -1:
                _q = self.robot.q_for_joint(curr_id, q)
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                temp = np.matmul(np.transpose(Xmat), f[:, curr_id])
                f[:, parent_id] = f[:, parent_id] + temp.flatten()

        return (c, f)

    def inverse_dynamics(
        self,
        q,
        qd,
        qdd=None,
        GRAVITY=-9.81,
        f_ext=None,
        public_output=True,
        normalize_input=True,
    ):
        """Compute the generalized forces using the Recursive Newton-Euler
        Algorithm (RNEA). This is the canonical inverse-dynamics routine.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.
        f_ext : numpy.ndarray
            External forces.

        Returns
        -------
        (c, v, a, f) : tuple
            Generalized forces and intermediate link quantities.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            if qdd is not None:
                qdd = self._normalize_v_input(qdd)
        # forward pass (external forces are subtracted from f inside fpass)
        (v, a, f) = self.inverse_dynamics_fpass(q, qd, qdd, GRAVITY, f_ext=f_ext)
        # backward pass
        (c, f) = self.inverse_dynamics_bpass(q, f)
        # joint-local viscous damping + Coulomb friction bias (no-op when the
        # robot declares neither; gated inside the helper)
        c = c + self._joint_dynamics_bias(qd)
        if public_output:
            c = self._denormalize_v_output(c)
        return (c, v, a, f)

    def f_ext_jacobian_transpose(self, q, normalize_input=True):
        """Stacked body-Jacobian transpose ``J^T`` (nv x 6*NB) for the f_ext column.

        Column block ``i`` (6 columns) is the joint-torque produced by a UNIT
        local-frame spatial wrench applied to body ``i`` and back-propagated by
        the RNEA backward sweep. By construction this is exactly the geometric
        mapping ``J^T[:, 6*i:6*i+6][:, k] = (S_j^T X_{i->j}^T) e_k`` summed over
        ``j`` on the path root->i (zero off-path), i.e. the transpose of the
        stacked spatial body Jacobian in each link's LOCAL frame.

        The forward convention (apply_external_forces / inverse_dynamics_fpass) SUBTRACTS the
        local wrench: ``f[:, i] -= f_ext[i]``. Therefore the gradient of the RNEA
        output ``tau`` w.r.t. ``f_ext`` is ``d(tau)/d(f_ext) = -J^T`` (see
        ``f_ext_gradient``). This method returns the (un-signed) ``J^T`` so both
        ``-J^T`` (tau) and ``+M^{-1} J^T`` (qdd) consumers can apply their own sign.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()
        JT = np.zeros((nv, 6 * NB), dtype=np.float64)
        # For each body i and each of the 6 local-wrench basis vectors, seed the
        # internal force array and run the RNEA backward sweep (S^T f projection +
        # X^T f parent propagation). The resulting c is column (6*i + k) of J^T.
        for i in range(NB):
            for k in range(6):
                f = np.zeros((6, NB), dtype=np.float64)
                f[k, i] = 1.0
                (c, _f) = self.inverse_dynamics_bpass(q, f)
                JT[:, 6 * i + k] = c
        return JT

    def f_ext_jacobian_transpose_dq(self, q, normalize_input=True):
        """Analytic ``dJ^T/dq`` (nv x 6*NB x nv) of the stacked body-Jacobian
        transpose, for FIXED and FLOATING base (section A.3 oracle).

        ``J^T[v_j, 6*i + k] = col_{i,j}[k]`` where the geometric-Jacobian column
        of body ``i`` for chain joint ``j`` (in body ``i``'s LOCAL frame) is

            col_{i,j} = X[i] X[i-1] ... X[j+1] S_j ,

        the Featherstone motion-transform pushdown of the (constant in its own
        joint frame) motion subspace ``S_j`` from joint ``j`` down to body ``i``.
        Only the local transforms ``X[m]`` for ``m`` on the chain (j, i] carry a
        q-dependence, through their own coordinate ``q_m`` (the joint transform
        ``X_J(q_m)``), with the Featherstone identity

            d X[m] / d q_m = -crm(S_m) X[m]      (crm == motion cross operator).

        Differentiating ``col_{i,j}`` w.r.t. ``q_m`` (m on (j, i]) gives the closed
        form (no finite differencing):

            d col_{i,j} / d q_m = -X_{m->i} ( S_m x col_{m,j} ),

        where ``X_{m->i} = X[i]...X[m+1]`` pushes the perturbation from frame ``m``
        down to frame ``i`` and ``col_{m,j} = X[m]...X[j+1] S_j`` is the partial
        pushdown to body ``m``. The result is written to ``dJT[v_j, 6*i+k, v_m]``.
        Out-of-chain ``(i, m)`` pairs are zero.

        FLOATING BASE: the same closed form holds for the 6-DoF free-flyer root
        (jid 0) with NO new identity. The root's motion subspace is the 6-column
        ``S_0`` (the free-flyer twist basis, pinocchio [v_lin; omega] order); each
        column ``S_0^{(a)}`` owns a distinct root v-slot ``a in [0,6)``. As a SOURCE
        joint it contributes 6 geometric-Jacobian columns (one per root v-slot); as
        a PERTURBED joint, ``d col_{i,j}/d(root v-slot a) = -X_{m->i}(S_0^{(a)} x
        col_{0,j})`` -- exactly the Featherstone ``-crm(S)X`` retract along the
        a-th body-frame twist column, the same SE(3) right-retract that
        ``self.integrate`` applies to the root in the central-FD self-check. Per
        root v-slot it is a separate ``a`` with ``-crm(S_0^{(a)})X[0]``; no special
        6-DoF code is needed -- the multi-column loop over each chain joint's S
        subsumes both the scalar revolute/prismatic joints (1 column, 1 v-slot) and
        the 6-column root. Validated against a central FD of
        ``f_ext_jacobian_transpose`` through ``self.integrate`` to FD-truncation
        accuracy (~1e-10) on iiwa14-floating / go2-floating.

        MIMIC: a mimic joint shares its target's reduced v-slot, so its
        geometric-Jacobian column accumulates (alpha-weighted via the motion
        subspace) into the shared (v_j / v_m) entry — the same ``+=`` reduction the
        first-order ``f_ext_jacobian_transpose`` and the GRiD emit perform.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()
        dJT = np.zeros((nv, 6 * NB, nv), dtype=np.float64)

        def _vslots(jid):
            # Full ordered list of v-slots a joint owns: a single scalar for
            # revolute/prismatic, the 6-wide free-flyer block for the floating
            # root (one per motion-subspace column).
            v = self.robot.get_joint_index_v(jid)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(x) for x in v]
            return [int(v)]

        def _Sof(jid):
            S = np.asarray(self.robot.get_S_by_id(jid), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            return S

        def _Xof(m):
            return np.asarray(
                self.robot.get_Xmat_Func_by_id(m)(self.robot.q_for_joint(m, q)),
                dtype=np.float64,
            )

        for i in range(NB):
            chain = sorted(self.robot.get_ancestors_by_id(i)) + [i]
            for j in chain:
                S = _Sof(j)
                vj_list = _vslots(j)  # per-column source v-slots (root: 6 of them)
                # ordered chain joints (j, i] whose local transforms push S_j down
                tf = []
                mm = i
                while mm != j:
                    tf.append(mm)
                    mm = self.robot.get_parent_id(mm)
                tf = list(reversed(tf))  # j+1, j+2, ..., i
                # MIMIC source scaling: a mimic source joint j contributes its
                # geometric-Jacobian column α_j-weighted into the shared v_j slot
                # (matching the α-weighted reduction the first-order J^T performs).
                # For a leaf mimic (h1_2 hand) the chain (j, i] is empty so this
                # never fires, but it keeps a mid-chain mimic-with-descendants
                # source correct. α_j = 1 for non-mimic joints (no-op).
                alpha_j = self._mimic_multiplier(j)
                # one source geometric-Jacobian column per S column / source v-slot
                for c in range(S.shape[1]):
                    vj = vj_list[c]
                    # precompute the running pushdown col_{m,j} = X[m]..X[j+1] S_j
                    # at each frame m on the chain (col_at[j] = S_j[:, c]).
                    col = S[:6, c].astype(np.float64)
                    col_at = {j: col.copy()}
                    for m in tf:
                        col = _Xof(m) @ col
                        col_at[m] = col.copy()
                    # derivative wrt each chain coordinate q_m, m in (j, i]
                    for midx, m in enumerate(tf):
                        Sm = _Sof(m)
                        vm_list = _vslots(m)  # per-column perturbed v-slots
                        # MIMIC chain-rule scaling: the perturbed joint m's angle
                        # is θ_m = α_m·q_{vm} + offset, so dX[m]/dq_{vm} =
                        # α_m·(-crm(S_m)X[m]). Multiple joints can share v-slot vm
                        # (a primary α=1 plus its α≠1 mimics); each is a separate m
                        # on the chain and accumulates its α_m-scaled term via +=.
                        # Omitting α_m undercounts non-unit-mult mimic slots (h1_2
                        # hand: α=1.6/2.4). α_m = 1 for non-mimic (no-op).
                        alpha_m = self._mimic_multiplier(m)
                        # X_{m->i} = X[i]...X[m+1]
                        Xmi = np.eye(6)
                        for m2 in tf[midx + 1:]:
                            Xmi = _Xof(m2) @ Xmi
                        # one perturbed coordinate per S_m column / v-slot (root: 6)
                        for cm in range(Sm.shape[1]):
                            vm = vm_list[cm]
                            # S_m^(cm) x col_{m,j} (spatial motion cross product)
                            term = self.cross_operator(Sm[:6, cm]) @ col_at[m]
                            dcol = -(Xmi @ term)
                            dJT[vj, 6 * i:6 * i + 6, vm] += (alpha_j * alpha_m) * dcol
        return dJT

    def f_ext_gradient(self, q, normalize_input=True):
        """Gradients of dynamics w.r.t. external forces ``f_ext`` (the f_ext column).

        ``f_ext`` is a per-body spatial wrench in each link's LOCAL frame
        (layout [angular(3); linear(3)] per body, body-major 6*NB), SUBTRACTED
        from the RNEA per-body force (see ``apply_external_forces``). Because it
        enters RNEA additively and linearly, the dynamics are affine in f_ext
        with a q-only Jacobian. Returns a dict of the three exact analytic blocks:

          ``dtau_dfext``  = -J^T            (nv x 6*NB)   [section A.1]
          ``dqdd_dfext``  =  M^{-1} J^T     (nv x 6*NB)   [section A.2]
          ``did_du_dfext_dq`` = -dJ^T/dq    (nv x 6*NB x nv)   [section A.3]
          ``did_du_dfext_dqd`` = 0          (nv x 6*NB x nv)   (q-only -> zero)

        ``J^T`` is the stacked spatial body-Jacobian transpose (local frame, all
        bodies) from ``f_ext_jacobian_transpose``. ``M^{-1}`` is ``minv``. The
        q-derivative ``dJ^T/dq`` is the analytic closed form from
        ``f_ext_jacobian_transpose_dq`` for BOTH fixed and floating base (no finite
        differencing): the floating free-flyer root's 6 motion-subspace columns
        slot into the same Featherstone ``-crm(S)X`` pushdown as the scalar joints
        (one ``-crm(S_0^{(a)})X[0]`` per root v-slot a). See that method's
        derivation + FD self-check.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()
        JT = self.f_ext_jacobian_transpose(q, normalize_input=False)
        Minv = self.minv(q, public_output=False, normalize_input=False)

        dtau_dfext = -JT
        dqdd_dfext = Minv @ JT

        # Analytic dJ^T/dq (closed form, no FD) for fixed AND floating base.
        dJT_dq = self.f_ext_jacobian_transpose_dq(q, normalize_input=False)

        return {
            "dtau_dfext": dtau_dfext,
            "dqdd_dfext": dqdd_dfext,
            "did_du_dfext_dq": -dJT_dq,
            "did_du_dfext_dqd": np.zeros((nv, 6 * NB, nv), dtype=np.float64),
        }

    def _has_mimic_joints(self):
        """Return True if any actuated joint is a URDF <mimic> joint.

        Used to gate algorithms (minv / aba / forward-dynamics gradient) that
        need a different reduced-model treatment when mimic joints are
        present -- the standard ABA `Ia = IA - U U^T / d` recursion uses
        per-body (S, U, d) that diverge by alpha / alpha^2 factors for mimic
        joints, and pinocchio handles this via reduced-model constraint
        projection rather than the `+=` accumulation that suffices for CRBA
        / RNEA-grad. See OPEN ISSUE on `aba`.
        """
        for joint in self.robot.joints:
            if getattr(joint, "is_mimic", False):
                return True
        return False

    def minv_bpass(self, q):
        """Backward pass for the Articulated-Body Algorithm to compute inverse inertia.

        Parameters
        ----------
        I_art : numpy.ndarray
            Articulated-body inertia matrices.

        Returns
        -------
        (Minv, F, U, Dinv) : tuple
            Matrices for inverse inertia composition.
        """
        # Allocate memory. Size in v-space (nv) is the correct size for the
        # reduced output matrix; mimic joints don't carry their own v-slot.
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        Minv = np.zeros((n, n))
        F = np.zeros((n, 6, n))
        U = np.zeros((n, 6))
        Dinv = np.zeros(n)

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

        # Backward pass. see docs/open-tasks/notes.md (RBDReference.py:1907)
        for ind in range(NB - 1, -1, -1):
            subtreeInds = self.robot.get_subtree_by_id(ind)
            adj_subtreeInds = self._vinds_for_subtree(subtreeInds)
            matrix_ind = self.robot.get_joint_index_v(ind)
            parent_ind = self.robot.get_parent_id(ind)
            if (
                parent_ind == -1 and self.robot.floating_base
            ):  # floating base joint check
                # Compute U, D over the floating-base 6-wide v-block (matrix_ind is [0..5]).
                S = self.robot.get_S_by_id(ind)  # np.eye(6) for floating base
                U[:6, :] = np.matmul(IA[ind], S)
                fb_Dinv = np.linalg.inv(
                    np.matmul(S.transpose(), U[:6, :])
                )  # vectorized Dinv calc
                # Update Minv and subtrees - subtree calculation for Minv -= Dinv * S.T * F with clever indexing
                Minv[:6, :6] = Minv[0, 0] + fb_Dinv
                Minv[np.ix_(list(range(6)), adj_subtreeInds)] -= (
                    np.matmul(
                        np.matmul(fb_Dinv, S),
                        F[np.ix_(list(range(6)), list(range(6)), adj_subtreeInds)],
                    )
                )[-1]
            else:
                # Compute U, D
                S = self.robot.get_S_by_id(
                    ind
                )  # NOTE Can S be an np.array not np.matrix? np.matrix outdated...
                U[matrix_ind, :] = np.matmul(IA[ind], S).reshape(6,)
                Dinv[matrix_ind] = np.matmul(S.transpose(), U[matrix_ind, :])
                # Update Minv and subtrees
                Minv[matrix_ind, matrix_ind] = 1 / Dinv[matrix_ind]
                # Deals with issue where result is np.matrix instead of np.array (can't shape np.matrix as 1 dimension)
                Minv[matrix_ind, adj_subtreeInds] -= np.squeeze(
                    np.array(
                        1
                        / (Dinv[matrix_ind])
                        * np.matmul(S.transpose(), F[matrix_ind, :, adj_subtreeInds].T)
                    )
                )
                # update parent if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    matrix_parent_ind = self.robot.get_joint_index_v(parent_ind)
                    _q = self.robot.q_for_joint(ind, q)
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    # update F
                    for subInd in adj_subtreeInds:
                        F[matrix_ind, :, subInd] += (
                            U[matrix_ind, :] * Minv[matrix_ind, subInd]
                        )
                        F[matrix_parent_ind, :, subInd] += np.matmul(
                            np.transpose(Xmat), F[matrix_ind, :, subInd]
                        )
                    # update IA
                    Ia = IA[ind] - np.outer(
                        U[matrix_ind, :],
                        ((1 / Dinv[matrix_ind]) * np.transpose(U[matrix_ind, :])),
                    )  # replace 1/Dinv if using linalg.inv
                    IaParent = np.matmul(np.transpose(Xmat), np.matmul(Ia, Xmat))
                    IA[parent_ind] += IaParent

        return Minv, F, U, Dinv

    def _vinds_for_subtree(self, subtreeInds):
        """Flatten a subtree's joint ids into a list of v-space indices.

        Handles the floating-base root (whose v-index is a 6-wide list).
        """
        result = []
        for s in subtreeInds:
            vs = self.robot.get_joint_index_v(s)
            if isinstance(vs, (list, tuple, np.ndarray)):
                result.extend(list(vs))
            else:
                result.append(vs)
        return result

    def minv_fpass(self, q, Minv, F, U, Dinv):
        """Forward pass for the Articulated-Body Algorithm to compute inverse inertia.

        Parameters
        ----------
        Minv : numpy.ndarray
            N-element vector or matrix for Minv.
        F : numpy.ndarray
            Intermediate matrix.
        U : numpy.ndarray
            Intermediate matrix.
        Dinv : numpy.ndarray
            Inverse of articulated inertia projected on subspace.

        Returns
        -------
        Minv : numpy.ndarray
            Inverse of the joint-space inertia matrix.
        """
        NB = self.robot.get_num_bodies()
        # # Forward pass.
        # Mimic-aware: use the v-space index (get_joint_index_v) and the
        # mimic-aware q_for_joint helper so the loop runs over all bodies
        # (including mimics) without ever indexing beyond nv.
        for ind in range(NB):
            matrix_ind = self.robot.get_joint_index_v(ind)
            _q = self.robot.q_for_joint(ind, q)
            parent_ind = self.robot.get_parent_id(ind)
            S = self.robot.get_S_by_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            if parent_ind != -1:
                Minv[matrix_ind, :] -= (1 / Dinv[matrix_ind]) * np.matmul(
                    np.matmul(U[matrix_ind].transpose(), Xmat), F[parent_ind]
                )
                F[ind] = np.matmul(Xmat, F[parent_ind]) + np.outer(
                    S, Minv[matrix_ind, :]
                )
            else:
                if self.robot.floating_base:
                    F[ind] = np.matmul(S, Minv[:6, :])
                else:
                    F[ind] = np.outer(S, Minv[matrix_ind, :])

        return Minv

    def minv(self, q, output_dense=True, public_output=True, normalize_input=True):
        """Compute the inverse of the joint-space inertia matrix.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.

        Returns
        -------
        Minv : numpy.ndarray
            N x N inverse joint-space inertia matrix.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        # CRBA-invert fast path. Two cases route here:
        #  (1) MIMIC: the ABA-based recursion below uses per-body (S, U, d) that
        #      diverge by alpha/alpha^2 factors for mimic joints (see
        #      _has_mimic_joints docstring and OPEN ISSUE for `aba`).
        #      Pinocchio's reduced-model inverse is (G^T M_full G)^{-1}, which is
        #      NOT equal to G^T M_full^{-1} G in general -- so the only correct
        #      general handling is to invert the reduced M directly. CRBA already
        #      produces the reduced M (G^T M_full G) via its mimic-aware +=
        #      pattern, so we simply invert that.
        #  (2) MULTI-DoF non-floating joint (SPHERICAL/planar mid-chain): the
        #      minv_bpass/fpass ABA recursion is scalar-per-DoF (it assumes a
        #      single-column motion subspace) and raises / mis-handles a body
        #      that owns a 6x3 S. CRBA handles the multi-column block correctly,
        #      so invert the dense reduced M for these robots too. The else ABA
        #      recursion (the cardinal single-DoF case) is untouched, so its
        #      behavior is unchanged -- this only ADDS coverage for spherical,
        #      unblocking the fd / SO oracle paths that compose minv.
        if self._has_mimic_joints() or self._robot_has_multidof_nonfloating_joint():
            # Invert the INTERNAL-order M so the floating-base v reorder is
            # applied exactly once (here), not twice: crba's public output
            # already denormalizes, and the [3,4,5,0,1,2] root reorder is an
            # involution, so inverting the public M and denormalizing again
            # would cancel the reorder (legacy output would wrongly come back
            # in pinocchio order). See crba(public_output=...).
            M = self.crba(q, normalize_input=False, public_output=False)
            Minv = np.linalg.inv(M)
            if public_output:
                return self._denormalize_qv_matrix_output(
                    Minv, row_space="v", col_space="v"
                )
            return Minv
        # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        # backward pass
        (Minv, F, U, Dinv) = self.minv_bpass(q)

        # forward pass
        Minv = self.minv_fpass(q, Minv, F, U, Dinv)

        # fill in full matrix (currently only upper triangular)
        if output_dense:
            NB = self.robot.get_num_bodies()
            for col in range(NB):
                for row in range(NB):
                    if col < row:
                        Minv[row, col] = Minv[col, row]

        if public_output:
            return self._denormalize_qv_matrix_output(Minv, row_space="v", col_space="v")
        return Minv
    

    def crm(self,v):
        """Spatial velocity cross product operator.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_cross : numpy.ndarray
            6x6 cross product matrix.
        """
        if len(v) == 6:
            vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
        else:
            vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
        return vcross


    def aba(self, q, qd, tau, f_ext=[], GRAVITY = -9.81, normalize_input=True):
        """Compute forward dynamics using the Articulated-Body Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.
        f_ext : numpy.ndarray
            External forces.

        Returns
        -------
        qdd : numpy.ndarray
            N-element joint accelerations.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            tau = self._normalize_v_input(tau)
        # Mimic-aware fast path: the ABA recursion below uses per-body
        # (S, U, d) that diverge from slot-accumulated by alpha/alpha^2
        # factors for mimic joints. The += CRBA/RNEA-grad pattern is
        # insufficient because `Ia = IA - U U^T / d` is a per-body update
        # whose contribution to the parent's articulated inertia must be
        # scaled by alpha^2, while the qdd-solve must accumulate the
        # mimic's S^T*tau_eff contribution into the mimicked v-slot
        # scaled by alpha. Rather than re-derive the recursion, we use
        # the equivalent reduced-model forward dynamics:
        #     qdd = M_reduced^{-1} * (tau - inverse_dynamics(q, qd, 0))
        # where M_reduced and inverse_dynamics(q, qd, 0) are already mimic-aware via
        # CRBA and the bpass `+= alpha * S^T f` pattern. This matches
        # pinocchio's constraint-aware forward dynamics for mimic models
        # (their `aba` on the unreduced model would diverge similarly).
        if self._has_mimic_joints():
            # Mimic reduced-model forward dynamics:
            #   qdd = M_reduced^{-1} * (tau - inverse_dynamics(q, qd, 0; f_ext))
            # External forces enter purely through the inverse_dynamics bias (which
            # subtracts the local-frame f_ext from the per-body force); the
            # reduced mass matrix is unaffected by f_ext. T3 owns the mimic
            # ABA recursion fallback; here f_ext only flows into the bias.
            n = len(qd)
            bias = self.inverse_dynamics(
                q, qd, np.zeros(n),
                GRAVITY=GRAVITY,
                f_ext=f_ext,
                public_output=False,
                normalize_input=False,
            )[0]
            Minv = self.minv(
                q, output_dense=True, public_output=False, normalize_input=False
            )
            qdd = Minv @ (tau - bias)
            return self._denormalize_v_output(qdd)
        # Multi-DOF non-floating joint (spherical/planar mid-chain): the in-place
        # ABA recursion below is written for SCALAR 1-DOF joints (per-body U, d,
        # q[ind]/qd[ind] scalar indexing). Rather than re-derive the multi-column
        # articulated-inertia recursion, use the equivalent reduced-model forward
        # dynamics qdd = M^{-1} (tau - rnea_bias) (with f_ext folded into the
        # bias). M (crba) and the RNEA bias are already multi-column-correct, so
        # this is exact. The cardinal 1-DOF fixed-base path below is untouched.
        if self._robot_has_multidof_nonfloating_joint():
            n = len(qd)
            bias = self.inverse_dynamics(
                q, qd, np.zeros(n),
                GRAVITY=GRAVITY,
                f_ext=f_ext,
                public_output=False,
                normalize_input=False,
            )[0]
            # Solve in INTERNAL order (bias/tau are internal here): use the raw
            # internal-order M so a floating+multidof robot doesn't mix a
            # denormalized M with an internal-order rhs. Single denormalize on qdd.
            M = self.crba(q, normalize_input=False, public_output=False)
            qdd = np.linalg.solve(M, tau - bias)
            return self._denormalize_v_output(qdd)
        # joint-local viscous damping + Coulomb friction reduce the torque
        # available to accelerate: qdd = Minv*(tau - rnea_bias - dyn_bias).
        # No-op (zero vector) when no joint declares damping/friction.
        dyn_bias = self._joint_dynamics_bias(qd)
        if self.robot.floating_base:
            # allocate memory. see docs/open-tasks/notes.md (RBDReference.py:2158)
            n = len(qd)
            NB = self.robot.get_num_bodies()
            v = np.zeros((6,NB))
            c = np.zeros((6,NB))
            a = np.zeros((6,NB))
            IA = np.zeros((NB,6,6))
            pA = np.zeros((6,NB))
            # variables may require special indexing
            f = np.zeros((6,n))
            # d = np.zeros(n)
            d = {}
            U = np.zeros((6,n))
            u = np.zeros(n)
            qdd = np.zeros(n)

            gravity_vec = np.zeros((6))
            gravity_vec[5] = -GRAVITY  # a_base is gravity vec

            # Initial Forward Pass
            for ind in range(NB): # curr_id = ind for this loop
                parent_ind = self.robot.get_parent_id(ind)
                _q = q[self.robot.get_joint_index_q(ind)]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                S = self.robot.get_S_by_id(ind)
                inds_v = self.robot.get_joint_index_v(ind)

                if parent_ind == -1: # parent is base
                    if self.robot.floating_base:
                        v[:, ind] = np.matmul(S, qd[ind:ind+6])
                    else:
                        v[:, ind] = np.squeeze(np.array(S*qd[ind]))
                else:
                    v[:, ind] = np.matmul(Xmat, v[:, parent_ind]) 
                    vJ = np.squeeze(np.array(S * qd[inds_v])) # reduces shape to (6,) matching v[:,curr_id]
                    v[:, ind] += vJ
                    c[:, ind] = np.matmul(self.cross_operator(v[:, ind]), vJ)

                Imat = self.robot.get_Imat_by_id(ind)
                # print(f'Imat:{Imat.shape}\n {Imat}')
                # print(IA[:,:,ind].shape)
                IA[ind] = Imat

                vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
                [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
                [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
                [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
                [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
                [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

                crf = -np.transpose(vcross) 
                temp = np.matmul(crf, Imat)

                pA[:, ind] = np.matmul(temp, v[:, ind])

            # apply external forces (subtract local-frame f_ext from the bias)
            pA = self.apply_external_forces(pA, f_ext)

            # Backward Pass
            for ind in range(NB-1, -1, -1): # ind != ind for bpass
                S = self.robot.get_S_by_id(ind)
                parent_ind = self.robot.get_parent_id(ind)
                inds_v = self.robot.get_joint_index_v(ind)
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)

                U[:, inds_v] = np.squeeze(np.matmul(IA[ind], S))
                d[ind] = np.matmul(np.transpose(S), U[:, inds_v])
                u[inds_v] = tau[inds_v] - dyn_bias[inds_v] - (np.matmul(S.T, pA[:, ind])) - (np.matmul(U[:, inds_v].T, c[:, ind]))

                if parent_ind != -1:
                    U[:, inds_v] = np.matmul(Xmat.T, U[:, inds_v]) # spatial edit

                    rightSide = np.reshape(U[:, inds_v], (6,1)) @ np.reshape(U[:, inds_v], (6,1)).T / d[ind]
                    Ia = np.matmul(Xmat.T, np.matmul(IA[ind], Xmat)) - rightSide # spatial edit

                    pa = np.matmul(Xmat.T, pA[:, ind] + np.matmul(IA[ind], c[:, ind]))
                    pa = pa + np.reshape(U[:, inds_v], (6, 1)).flatten() * ((1 / d[ind]) * u[inds_v])
                    
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    temp = np.matmul(np.transpose(Xmat), Ia)

                    IA[parent_ind] = IA[parent_ind] + Ia # spatial edit

                    pA[:, parent_ind] = pA[:, parent_ind] + pa # spatial edit


            # Final Forward Pass
            for ind in range(NB): # ind != ind for bpass
                parent_ind = self.robot.get_parent_id(ind)
                inds_q = self.robot.get_joint_index_q(ind)
                inds_v = self.robot.get_joint_index_v(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)

                if parent_ind == -1: # parent is base
                    a[:, ind] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, ind] = a[:, parent_ind]
                
                S = self.robot.get_S_by_id(ind)
                temp = u[inds_v] - np.matmul(np.transpose(U[:, inds_v]), a[:, ind])

                if parent_ind == -1:
                    # qdd[inds_v] = np.matmul(np.linalg.inv(d[ind]), temp)
                    if self.robot.floating_base:
                        qdd[inds_v] = np.linalg.solve(d[ind], temp)
                        a[:, ind] = a[:, ind] + np.matmul(S, qdd[inds_v]) + c[:, ind]
                    else:
                        qdd[ind] = temp / d[ind]
                        a[:, ind] = np.matmul(Xmat, a[:, ind]) + qdd[ind]*S.T + c[:, ind]
                else:
                    # qdd[inds_v] = np.linalg.inv(d[ind]) * temp
                    qdd[inds_v] = temp / d[ind]
                    a[:, ind] = np.matmul(Xmat, a[:, ind]) + np.dot(S.T,qdd[inds_v]) + c[:, ind]
        else:
            n = len(qd)
            v = np.zeros((6,n))
            c = np.zeros((6,n))
            a = np.zeros((6,n))
            f = np.zeros((6,n))
            d = np.zeros(n)
            U = np.zeros((6,n))
            u = np.zeros(n)
            IA = np.zeros((6,6,n))
            pA = np.zeros((6,n))
            qdd = np.zeros(n)
            
            
            gravity_vec = np.zeros((6))
            gravity_vec[5] = -GRAVITY # a_base is gravity vec
                    
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)

                if parent_ind == -1: # parent is base
                    v[:,ind] = np.squeeze(np.array(S*qd[ind]))
                    
                else:
                    v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
                    v[:,ind] += np.squeeze(np.array(S*qd[ind]))
                    c[:,ind] = self._mxS(S,v[:,ind],qd[ind])

                Imat = self.robot.get_Imat_by_id(ind)

                IA[:,:,ind] = Imat

                vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
                [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
                [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
                [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
                [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
                [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

                crf=-np.transpose(vcross)
                temp=np.matmul(crf,Imat)

                pA[:,ind] = np.matmul(temp, v[:,ind])

            # apply external forces (subtract local-frame f_ext from the bias)
            pA = self.apply_external_forces(pA, f_ext)

            for ind in range(n-1,-1,-1):
                S = self.robot.get_S_by_id(ind)
                parent_ind = self.robot.get_parent_id(ind)

                U[:,ind] = np.squeeze(np.array(np.matmul(IA[:,:,ind],S)))
                d[ind] = np.matmul(np.transpose(S),U[:,ind])
                u[ind] = tau[ind] - dyn_bias[ind] - np.matmul(np.transpose(S),pA[:,ind])

                if parent_ind != -1:

                    rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
                    Ia = IA[:,:,ind] - rightSide

                    pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind]

                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    temp = np.matmul(np.transpose(Xmat), Ia)

                    IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)

                    temp = np.matmul(np.transpose(Xmat), pa)
                    pA[:,parent_ind]=pA[:,parent_ind] + temp.flatten()
                                                
            for ind in range(n):

                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

                if parent_ind == -1: # parent is base
                    a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
                else:
                    a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

                S = self.robot.get_S_by_id(ind)
                temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
                qdd[ind] = temp / d[ind]
                a[:,ind] = a[:,ind] + qdd[ind]*S.T
        
        return self._denormalize_v_output(qdd)




    def crba(self, q, normalize_input=True, public_output=True):
        """Compute the joint-space inertia matrix using the Composite Rigid Body Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        public_output : bool
            When True (default) the returned matrix is mapped back to the
            user-facing floating-base v convention (the legacy [3,4,5,0,1,2]
            root reorder under the legacy convention). Set False to get the
            raw INTERNAL-order matrix -- used by callers (e.g. ``minv``'s
            mimic path) that invert/solve in internal order and apply their
            own single denormalization afterward, so the involution perm is
            not applied twice.

        Returns
        -------
        M : numpy.ndarray
            N x N joint-space inertia matrix.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        if self.robot.floating_base:
            # Floating-base CRBA via the SAME generalized block machinery as the
            # fixed-base path below. The free-flyer ROOT (jid 0) is treated as an
            # ordinary 6-DOF joint whose motion subspace S = get_S_by_id(0) is the
            # permuted identity [[0,I],[I,0]] that already encodes Pinocchio's
            # [v_lin; omega] root ordering -- so S^T (chain) writes the correctly-
            # ordered root block and root<->joint cross terms with NO post-hoc
            # [3,4,5,0,1,2] reorder. A spherical/planar MID-CHAIN joint is then
            # just a second multi-column S (6xN) handled by the same np.ix_ block
            # writes as the root, validated vs pin free-flyer+JointModelSpherical.
            NB = self.robot.get_num_bodies()
            n = self.robot.get_num_vel()
            H = np.zeros((n, n))

            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            # Pass 1: compose each body's composite inertia up through its own
            # local transform (mimic-aware q_for_joint so a mimic joint sees the
            # scaled+offset coordinate). Root (jid 0) has parent -1 and is skipped.
            for ind in range(NB - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    _q = self.robot.q_for_joint(ind, q)
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )

            # Pass 2: H[v_i, v_j] = alpha_i alpha_j S_i^T (chain X^T) S_j, assembled
            # as full (N_i x N_j) blocks via np.ix_. For 1-DOF joints these are
            # 1x1 and reduce to the historical scalar; for the 6-DOF root and any
            # 3-DOF spherical/planar joint they place the multi-column block. The
            # chain walk runs to the root (parent_id > -1) so root<->joint coupling
            # is produced by the same loop (no separate root special-case).
            for ind in range(NB):
                vi = self._as_index_list(self.robot.get_joint_index_v(ind))
                alpha_i = self._mimic_multiplier(ind)
                S = self.robot.get_S_by_id(ind)
                fh = np.matmul(IC[ind], S)
                diag = (alpha_i * alpha_i) * np.matmul(S.T, fh)
                H[np.ix_(vi, vi)] += np.asarray(diag, dtype=np.float64).reshape(len(vi), len(vi))
                j = ind
                while self.robot.get_parent_id(j) > -1:
                    _qj = self.robot.q_for_joint(j, q)
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(_qj)
                    fh = np.matmul(Xmat.T, fh)
                    j = self.robot.get_parent_id(j)
                    S = self.robot.get_S_by_id(j)
                    alpha_j = self._mimic_multiplier(j)
                    vj = self._as_index_list(self.robot.get_joint_index_v(j))
                    block = (alpha_i * alpha_j) * np.asarray(
                        np.matmul(S.T, fh), dtype=np.float64
                    ).reshape(len(vj), len(vi))
                    # Off-diagonal chain pair (ind, j): both halves contribute to
                    # H[v_i, v_j] and H[v_j, v_i]. When v_i == v_j (mimic-ancestor
                    # sharing a slot) both writes land in the same cell.
                    H[np.ix_(vj, vi)] += block
                    H[np.ix_(vi, vj)] += block.T
        else:
            # # Fixed base implmentation of CRBA
            NB = self.robot.get_num_bodies()
            n = self.robot.get_num_vel()
            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            # IA-up-the-chain: each body's inertia (mimic or not) still
            # composes upward via the body's own X_local. Loop over bodies,
            # use the mimic-aware q_for_joint helper so a mimic joint sees
            # `multiplier * q[target] + offset` in its transform.
            for ind in range(NB - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                _q = self.robot.q_for_joint(ind, q)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)

                if parent_ind != -1:
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )

            H = np.zeros((n, n))

            # Mimic-aware H assembly. Each joint i contributes its column
            # scaled by alpha_i (the URDF mimic multiplier; 1.0 for non-mimic);
            # H[v_i, v_j] = alpha_i * alpha_j * S_i^T (chain) S_j. Both the
            # mimic and the mimicked write to the same v-slot, so we ACCUMULATE
            # with += (the symmetric H[v_j, v_i] also accumulates, avoiding
            # double-count when v_i == v_j).
            for ind in range(NB):
                vi = self._as_index_list(self.robot.get_joint_index_v(ind))
                alpha_i = self._mimic_multiplier(ind)
                S = self.robot.get_S_by_id(ind)
                # fh = IC_i S_i is (6 x N_i); diag = S_i^T fh is the (N_i x N_i)
                # joint-block of the mass matrix. For a 1-DOF joint these are
                # 6x1 / 1x1 and reduce to the historical scalar. ix_ places the
                # full block (off-diagonal entries within a multi-DOF joint's
                # own block, e.g. the spherical 3x3, are otherwise dropped by
                # the diagonal-only `H[vi, vi]` fancy index).
                fh = np.matmul(IC[ind], S)
                diag = (alpha_i * alpha_i) * np.matmul(S.T, fh)
                H[np.ix_(vi, vi)] += np.asarray(diag, dtype=np.float64).reshape(len(vi), len(vi))
                j = ind

                while self.robot.get_parent_id(j) > -1:
                    _qj = self.robot.q_for_joint(j, q)
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(_qj)
                    fh = np.matmul(Xmat.T, fh) # add an addition Xmat.T everytime
                    j = self.robot.get_parent_id(j)
                    S = self.robot.get_S_by_id(j)
                    alpha_j = self._mimic_multiplier(j)
                    vj = self._as_index_list(self.robot.get_joint_index_v(j))
                    # (N_j x N_i) coupling block H[v_j, v_i] = S_j^T (chain) S_i.
                    block = (alpha_i * alpha_j) * np.asarray(
                        np.matmul(S.T, fh), dtype=np.float64
                    ).reshape(len(vj), len(vi))
                    # Off-diagonal chain pair (ind, j): both halves contribute to
                    # H[v_i, v_j] and H[v_j, v_i]. When v_i == v_j (mimic-ancestor
                    # sharing a slot) both writes land in the same cell.
                    H[np.ix_(vj, vi)] += block
                    H[np.ix_(vi, vj)] += block.T

        if public_output:
            return self._denormalize_qv_matrix_output(H, row_space="v", col_space="v")
        return H

    ##### Testing original RNEA_grad to help with CUDA 
    def inverse_dynamics_gradient_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        """Forward pass gradient with respect to joint positions for RNEA.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (dv_dq, da_dq, df_dq) : tuple
            Gradients of velocity, acceleration, and force wrt q.
        """
        
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        dv_dq = np.zeros((6,n,NB))  # each body has its own derivative matrix with a column for each position
        da_dq = np.zeros((6,n,NB))
        df_dq = np.zeros((6,n,NB))

        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        # Mimic-aware: idx is the joint's v-slot (a list for the floating-base
        # root, a scalar otherwise). Mimic joints SHARE their target's v-slot,
        # so writes to `[:, idx, ind]` use `+=` with an alpha-multiplier so
        # that both the mimic and the mimicked accumulate cleanly; reads of
        # qd[idx] / a-parent likewise scale by alpha (the URDF mimic relation
        # v_full = alpha * v_target).
        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            idx = self.robot.get_joint_index_v(ind)
            alpha = self._mimic_multiplier(ind)
            # Xmat access sequence
            _q = self.robot.q_for_joint(ind, q)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,idx,ind] += alpha * self._mxS(S,np.matmul(Xmat,v[:,parent_ind])) # replace with new mxS

            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            # Per-DoF columns of the motion subspace. For a 1-DoF (revolute/
            # prismatic) joint this is a single column and the loop collapses
            # to the scalar case (byte-identical to before). For a multi-DoF
            # joint (k=3 spherical, k=6 floating root) S has k columns and the
            # v-block idx has k entries — the mxS_onCols(dv_dq)*qd term is the
            # SUM over the k columns, each scaled by qd[idx[j]].
            idx_cols = idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]
            S_cols = np.asarray(S).reshape(6, -1)
            for c in range(n):
                if parent_ind == -1 and self.robot.floating_base:
                    # Floating root: its own velocity derivative dv_dq[:,:,ind] is
                    # identically zero (no parent / v_base = 0), so the
                    # mxS_onCols(dv_dq)*qd term contributes nothing. (The previous
                    # code indexed the BODY axis with a root v-DOF index, which only
                    # avoided an IndexError on NB>6 robots and otherwise relied on
                    # this term being zero — fixed: skip it explicitly.)
                    pass
                else:
                    # da_dq[:,c,ind] += sum_j (alpha*qd[idx[j]]) * (dv_dq x S[:,j]).
                    for j, vcol in enumerate(idx_cols):
                        da_dq[:,c,ind] += self._mxS(S_cols[:,j], dv_dq[:,c,ind], alpha * qd[vcol])

            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,idx,ind] += alpha * self._mxS(S,np.matmul(Xmat,a[:,parent_ind])) # replace with new mxS
            else:
                if self.robot.floating_base:
                    root_gravity = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    root_gravity = np.matmul(Xmat, gravity_vec)
                da_dq[:,idx,ind] += alpha * self._mxS(S,root_gravity) # replace with new mxS
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)

            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])# puts 0.0014 instead of -0.0014 in df_dq[2,7,7]
            Iv = np.matmul(Imat,v[:,ind])

            for c in range(n):

                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))

        return (dv_dq, da_dq, df_dq)

    def inverse_dynamics_gradient_fpass_dqd(self, q, qd, v):
        """Forward pass gradient with respect to joint velocities for RNEA.

        Parameters
        ----------
        qd : numpy.ndarray
            N-element joint velocities.

        Returns
        -------
        (dv_dqd, da_dqd, df_dqd) : tuple
            Gradients of velocity, acceleration, and force wrt qd.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = len(qd)
        dv_dqd = np.zeros((6,n,NB))
        da_dqd = np.zeros((6,n,NB))
        df_dqd = np.zeros((6,n,NB))

        # forward pass.
        # Mimic-aware: same idiom as inverse_dynamics_gradient_fpass_dq above. inds_v / idx
        # is the joint's v-slot (a list for the floating-base root); mimic
        # joints fold into their target's v-slot via `+=` scaled by alpha,
        # and reads of qd[idx] / S contributions likewise scale by alpha.
        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            idx = self.robot.get_joint_index_v(ind)
            inds_v = idx  # legacy local name kept for parity with the original code
            alpha = self._mimic_multiplier(ind)
            _q = self.robot.q_for_joint(ind, q)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){alpha * S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
            dv_dqd[:,inds_v,ind] += alpha * np.squeeze(np.array(S)) # added squeeze and mxS
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
            # mxS_onCols(dv_dqd)*qd is the SUM over the joint's k DoF columns,
            # each scaled by qd[idx[j]]. For a 1-DoF joint this is the single
            # scalar term (byte-identical to before); for a multi-DoF joint
            # (k=3 spherical, k=6 floating root) it sums the k columns. (This
            # also replaces the old floating-root special-case, which indexed
            # S by ROW S[ii] — only coincidentally correct for the identity
            # root subspace — with the uniform column form S[:,j].)
            idx_cols = idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]
            S_cols = np.asarray(S).reshape(6, -1)
            for c in range(n):
                for j, vcol in enumerate(idx_cols):
                    da_dqd[:,c,ind] += self._mxS(S_cols[:,j], dv_dqd[:,c,ind], alpha * qd[vcol])

            da_dqd[:,idx,ind] += alpha * self._mxS(S,v[:,ind])
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)

            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):

                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))


        return (dv_dqd, da_dqd, df_dqd)

    def inverse_dynamics_gradient_bpass_dq(self, q, f, df_dq):
        """Backward pass gradient with respect to joint positions for RNEA.

        Parameters
        ----------
        df_dq : numpy.ndarray
            Gradient of spatial forces wrt q.

        Returns
        -------
        dc_dq : numpy.ndarray
            Gradient of generalized forces wrt q.
        """

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel() # assuming len(q) = len(qd)
        dc_dq = np.zeros((n,n))

        # Mimic-aware: idx is the joint's v-slot; mimic joints share their
        # target's slot, so both the dc_dq row-write and the per-joint S^T
        # term must `+=` with the alpha multiplier so mimic and mimicked
        # contributions fold together correctly.
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)
            idx = self.robot.get_joint_index_v(ind)
            alpha = self._mimic_multiplier(ind)

            # dc_du is alpha * S^T * df_du (accumulate on shared mimic slots)
            S = self.robot.get_S_by_id(ind)
            dc_dq[idx,:] += alpha * np.matmul(np.transpose(S),df_dq[:,:,ind])
            # df_du_parent += X^T*df_du + (if ind == c){alpha * X^T*fxS(f)}
            if parent_ind != -1:
                _q = self.robot.q_for_joint(ind, q)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                # X^T * fxS(S, f[ind]) scattered into the parent's own v-columns.
                # One column per DoF: df_dq[:, idx[j], parent] += X^T fxS(S[:,j], f).
                # For a 1-DoF joint this is the single-column scalar case
                # (byte-identical); for k>1 each column j uses S[:,j].
                idx_cols = idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]
                S_cols = np.asarray(S).reshape(6, -1)
                for j, vcol in enumerate(idx_cols):
                    delta_dq = np.matmul(np.transpose(Xmat), self.fxS(S_cols[:,j], f[:,ind]))
                    df_dq[:,vcol,parent_ind] += alpha * delta_dq


        return dc_dq

    def inverse_dynamics_gradient_bpass_dqd(self, q, df_dqd):
        """Backward pass gradient with respect to joint velocities for RNEA.

        Parameters
        ----------
        df_dqd : numpy.ndarray
            Gradient of spatial forces wrt qd.

        Returns
        -------
        dc_dqd : numpy.ndarray
            Gradient of generalized forces wrt qd.
        """

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel() # len(qd) always
        dc_dqd = np.zeros((n,n))

        # Mimic-aware: idx is the v-slot (a list for the floating-base root).
        # Mimic joints share their target's slot, so the dc_dqd row-write
        # uses += with the alpha multiplier.
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)
            idx = self.robot.get_joint_index_v(ind)
            alpha = self._mimic_multiplier(ind)
            # dc_du is alpha * S^T * df_du
            S = self.robot.get_S_by_id(ind)
            dc_dqd[idx,:] += alpha * np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            if parent_ind != -1:
                _q = self.robot.q_for_joint(ind, q)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind])


        # Joint-local viscous damping contributes a diagonal term to dc_dqd:
        # the value bias folds c += alpha*damping*qd_v into each joint's v-slot
        # (see _joint_dynamics_bias), so d(c)/d(qd_v) = alpha*damping on the
        # diagonal. Coulomb friction (alpha*f*sign(qd)) has zero (sub)gradient
        # a.e., so it contributes NOTHING here -- gate on damping only. Mirrors
        # _joint_dynamics_bias exactly: iterate bodies, map to the reduced
        # v-slot via get_joint_index_v + the mimic multiplier, skip the floating
        # root (it carries no damping), and ACCUMULATE so mimic joints sharing a
        # slot sum. Gated on use_joint_dynamics (connected to the value path),
        # NOT a separate kwarg.
        if self.use_joint_dynamics and self.robot.robot_has_joint_damping():
            for jid in range(NB):
                if self.robot.floating_base and self.robot.get_parent_id(jid) == -1:
                    continue  # floating root carries no damping
                b = float(self.robot.get_damping_by_id(jid))
                if b == 0.0:
                    continue
                idx = self.robot.get_joint_index_v(jid)
                alpha = self._mimic_multiplier(jid)
                idx_list = idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]
                for k in idx_list:
                    dc_dqd[k, k] += alpha * b

        return dc_dqd

    def inverse_dynamics_gradient(
        self,
        q,
        qd,
        qdd = None,
        GRAVITY = -9.81,
        f_ext=None,
        public_output=True,
        normalize_input=True,
    ):
        """Compute the gradients of RNEA wrt joint positions and velocities.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (dc_dq, dc_dqd) : tuple
            Gradients of generalized forces wrt q and qd.
        """
        
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            if qdd is not None:
                qdd = self._normalize_v_input(qdd)
        # Key physics: for a CONSTANT local-frame f_ext there is NO new
        # gradient term. f_ext enters only as a constant offset to the
        # per-body force f (f[:,i] -= f_ext[i]); since it is q/qd-independent
        # its derivative is zero, so df_dq/df_dqd from the grad forward passes
        # are unchanged. The gradient inherits f_ext purely through the
        # f_ext-corrected `f` that `inverse_dynamics_gradient_bpass_dq` consumes in its
        # `X^T * fxS(S, f[:,ind])` term. (FD-verified against pinocchio.)
        (c, v, a, f) = self.inverse_dynamics(
            q,
            qd,
            qdd,
            GRAVITY,
            f_ext=f_ext,
            public_output=False,
            normalize_input=False,
        )

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.inverse_dynamics_gradient_fpass_dq(q, qd, v, a, GRAVITY)
 
        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.inverse_dynamics_gradient_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.inverse_dynamics_gradient_bpass_dq(q, f, df_dq)

        # backward pass, dqd (joint-damping gradient gated internally on
        # self.use_joint_dynamics, so it follows the value path automatically --
        # forward_dynamics_gradient inherits it with no extra plumbing).
        dc_dqd = self.inverse_dynamics_gradient_bpass_dqd(q, df_dqd)

        if public_output:
            return self._denormalize_inverse_dynamics_gradient_output(dc_dq, dc_dqd)
        return np.hstack((dc_dq, dc_dqd))


    def forward_dynamics(self, q, qd, u, f_ext=None, public_output=True, normalize_input=True):
        """Compute the joint accelerations for the given state and torques.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        qdd : numpy.ndarray
            N-element joint accelerations.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            u = self._normalize_v_input(u)
        (c,v,a,f) = self.inverse_dynamics(q, qd, f_ext=f_ext, public_output=False, normalize_input=False)
        minv = self.minv(q, public_output=False, normalize_input=False)
        qdd = np.matmul(minv, u - c)
        if public_output:
            return self._denormalize_v_output(qdd)
        return qdd
    
    def forward_dynamics_gradient(self, q, qd, u, f_ext=None, normalize_input=True):
        """Compute the gradients of the forward dynamics.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        (qdd_dq, qdd_dqd) : tuple
            Gradients of joint accelerations wrt q and qd.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            u = self._normalize_v_input(u)
        # f_ext flows into qdd (via forward_dynamics) and into dc_du (via
        # inverse_dynamics_gradient's f_ext-corrected f). For a constant local-frame f_ext
        # there is no additional fd-gradient term: minv is f_ext-independent
        # and the only f_ext dependence is through the qdd/dc_du arguments
        # already threaded here.
        qdd = self.forward_dynamics(q, qd, u, f_ext=f_ext, public_output=False, normalize_input=False)
        dc_du = self.inverse_dynamics_gradient(q, qd, qdd, f_ext=f_ext, public_output=False, normalize_input=False)
        dc_dq, dc_dqd = np.hsplit(dc_du, [len(qd)])

        minv = self.minv(q, public_output=False, normalize_input=False)
        qdd_dq = np.matmul(-minv, dc_dq)
        qdd_dqd = np.matmul(-minv, dc_dqd)
        return (
            self._denormalize_reduced_q_matrix_output(qdd_dq, row_space="v"),
            self._denormalize_qv_matrix_output(qdd_dqd, row_space="v", col_space="v"),
        )

    @staticmethod
    def _skew_from_vector(vec):
        x, y, z = np.asarray(vec, dtype=np.float64)
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _vector_from_skew(skew):
        skew = np.asarray(skew, dtype=np.float64)
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)

    @staticmethod
    def _rotation_from_grid_rotvec(rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64)
        theta = np.linalg.norm(rotvec)
        generator = RBDReference._skew_from_vector(rotvec)
        if theta < 1e-12:
            return np.eye(3, dtype=np.float64) + generator + 0.5 * (generator @ generator)
        return (
            np.eye(3, dtype=np.float64)
            + (np.sin(theta) / theta) * generator
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (generator @ generator)
        )

    @staticmethod
    def _spatial_transform_from_motion(rot, trans):
        rot = np.asarray(rot, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        zeros = np.zeros((3, 3), dtype=np.float64)
        xlt = np.block(
            [
                [np.eye(3, dtype=np.float64), zeros],
                [-RBDReference._skew_from_vector(trans), np.eye(3, dtype=np.float64)],
            ]
        )
        xrot = np.block([[rot, zeros], [zeros, rot]])
        return xrot @ xlt

    @staticmethod
    def _floating_root_q_from_spatial_transform(xmat, reference_quat):
        xmat = np.asarray(xmat, dtype=np.float64)
        rot = xmat[:3, :3]
        trans_skew = -rot.T @ xmat[3:6, :3]
        quat = RBDReference._quat_xyzw_from_rotation_matrix(rot, reference_quat)
        return np.hstack((RBDReference._vector_from_skew(trans_skew), quat))

    def _floating_lie_perturbed_q(self, q, dind, step):
        q_perturbed = np.asarray(q, dtype=np.float64).copy()
        if dind >= 6:
            q_perturbed[dind + 1] += step
            return q_perturbed

        root_delta = np.zeros(6, dtype=np.float64)
        root_delta[dind] = step
        delta_x = self._spatial_transform_from_motion(
            self._rotation_from_grid_rotvec(root_delta[3:6]),
            root_delta[:3],
        )
        root_q_inds = self.robot.get_joint_index_q(0)
        root_x = self.robot.get_Xmat_Func_by_id(0)(q_perturbed[root_q_inds])
        perturbed_x = root_x @ delta_x
        q_perturbed[root_q_inds] = self._floating_root_q_from_spatial_transform(
            perturbed_x,
            q_perturbed[3:7],
        )
        return q_perturbed

    def _floating_idsva_d2tau_dq_lie_finite_diff(self, q, qd, qdd, GRAVITY=-9.81, step=1e-6):
        n = self.robot.get_num_vel()
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.using_quaternion:
            q[3:7] = self._normalize_xyzw_quaternion(q[3:7])
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        d2tau_dq = np.zeros((n, n, n), dtype=np.float64)
        for dind in range(n):
            q_pos = self._floating_lie_perturbed_q(q, dind, step)
            q_neg = self._floating_lie_perturbed_q(q, dind, -step)
            dc_dq_pos, _dc_dqd_pos = np.hsplit(
                self.inverse_dynamics_gradient(q_pos, qd, qdd, GRAVITY), [n]
            )
            dc_dq_neg, _dc_dqd_neg = np.hsplit(
                self.inverse_dynamics_gradient(q_neg, qd, qdd, GRAVITY), [n]
            )
            d2tau_dq[:, :, dind] = (dc_dq_pos - dc_dq_neg) / (2.0 * step)
        return d2tau_dq

    @staticmethod
    def _as_index_list(index):
        if isinstance(index, list):
            return index
        if isinstance(index, tuple):
            return list(index)
        if isinstance(index, np.ndarray):
            return list(index.flatten())
        return [index]

    def _spatial_xmat_derivative_func(self, jid, local_index):
        key = (jid, local_index)
        if key not in self._spatial_xmat_derivative_func_cache:
            joint = self.robot.get_joint_by_id(jid)
            dX = sp.diff(joint.get_transformation_matrix(), joint.position_symbols[local_index])
            self._spatial_xmat_derivative_func_cache[key] = sp.utilities.lambdify(
                joint._local_q_lambdify_args(),
                dX,
                "numpy",
            )
        return self._spatial_xmat_derivative_func_cache[key]

    def _spatial_xmat_second_derivative_func(self, jid, local_index_i, local_index_j):
        key = (jid, local_index_i, local_index_j)
        if key not in self._spatial_xmat_second_derivative_func_cache:
            joint = self.robot.get_joint_by_id(jid)
            self._spatial_xmat_second_derivative_func_cache[key] = (
                joint.get_d2transformation_matrix_local_function(
                    local_index_i,
                    local_index_j,
                )
            )
        return self._spatial_xmat_second_derivative_func_cache[key]

    def _floating_gravity_d2tau_dq_lie_direct(self, q, GRAVITY=-9.81):
        """Lie-tangent gravity Hessian d²τ_grav/dq² for floating-base robots.

        Body-major layout (axis 0 = body id) keeps per-body slices contiguous so
        the recursive operations dispatch as batched GEMMs. Gravity-only `a` is
        propagated `a[i] = X[i] @ a[parent]` down the tree (rooted at
        `inv(X[0]) @ g`), carrying first- and second-order q-derivatives, then
        `f = I @ a` is back-propagated and projected onto each joint's S.

        Mimic-aware: when any actuated joint is a URDF <mimic> joint, multiple
        bodies share the same project v-slot. We allocate per-body UNIQUE
        internal slots for the n-axis (so the dX / d2X / d2tau bookkeeping
        doesn't overwrite a mimicked-joint slot with the mimic's contribution
        and vice versa), then fold all three axes to the project layout with
        the URDF mimic multiplier on each axis. Also use `q_for_joint` so a
        mimic joint sees the scaled+offset slice of the target's q in its
        transform.
        """
        if not self.robot.floating_base:
            raise ValueError("_floating_gravity_d2tau_dq_lie_direct requires a floating-base robot.")

        q = self._normalize_q_input(q)
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.using_quaternion:
            q[3:7] = self._normalize_xyzw_quaternion(q[3:7])

        NB, n = self.robot.get_num_bodies(), self.robot.get_num_vel()

        # Build the per-body internal-vs-true slot mapping for the n-axis. For
        # non-mimic robots the internal and true slots coincide (R below
        # reduces to the identity and the fold is a no-op).
        true_v_inds = [self._as_index_list(self.robot.get_joint_index_v(i)) for i in range(NB)]
        has_mimic = self._has_mimic_joints()
        if has_mimic:
            body_v_inds = []
            offs = 0
            for i in range(NB):
                width = len(true_v_inds[i])
                body_v_inds.append(list(range(offs, offs + width)))
                offs += width
            n_int = offs
        else:
            body_v_inds = true_v_inds
            n_int = n
        body_alpha = [self._mimic_multiplier(i) for i in range(NB)]

        gravity_vec = np.zeros(6); gravity_vec[5] = -GRAVITY
        X = np.zeros((NB, 6, 6))
        dX = np.zeros((NB, n_int, 6, 6))
        d2X = np.zeros((NB, n_int, n_int, 6, 6))
        Imats = np.stack([np.asarray(self.robot.get_Imat_by_id(jid)) for jid in range(NB)])  # (NB, 6, 6)

        for jid in range(NB):
            q_arg = self.robot.q_for_joint(jid, q)
            X[jid] = np.asarray(self.robot.get_Xmat_Func_by_id(jid)(q_arg)).reshape(6, 6)
            if not self.robot.get_joint_by_id(jid).position_symbols:
                continue
            vinds = body_v_inds[jid]
            if jid == 0:
                # Root joint: Lie generators with Featherstone xlt translation-sign flip.
                S_root = np.asarray(self.robot.get_S_by_id(0))
                B = [(-1 if li < 3 else 1) * self.cross_operator(S_root[:, li]) for li in range(len(vinds))]
                for li, vi in enumerate(vinds):
                    dX[jid, vi] = X[jid] @ B[li]
                    for lj, vj in enumerate(vinds):
                        d2X[jid, vi, vj] = X[jid] @ B[lj] @ B[li]
            else:
                # In the internal (unreduced) layout, slot body_v_inds[jid]
                # represents body jid's OWN q (one slot per body). The
                # local q-derivative is therefore taken without an alpha
                # factor; the final R-fold below propagates the alpha for
                # mimic joints when collapsing the internal axes into the
                # project layout.
                for li, vi in enumerate(vinds):
                    dX[jid, vi] = np.asarray(self._spatial_xmat_derivative_func(jid, li)(q_arg)).reshape(6, 6)
                    for lj, vj in enumerate(vinds):
                        d2X[jid, vi, vj] = np.asarray(
                            self._spatial_xmat_second_derivative_func(jid, li, lj)(q_arg)
                        ).reshape(6, 6)

        # Forward sweep. Per-body work is per-body sequential (parent dependency)
        # but expressed as batched matmuls/broadcasts over the n direction axis.
        a = np.zeros((NB, 6))
        da = np.zeros((NB, n_int, 6))
        d2a = np.zeros((NB, n_int, n_int, 6))
        for jid in range(NB):
            pid = self.robot.get_parent_id(jid)
            if pid == -1:
                inv_X = np.linalg.inv(X[jid])
                a[jid] = inv_X @ gravity_vec
                dX_inv = dX[jid] @ inv_X                                                      # (n_int, 6, 6)
                inv_dX_inv = inv_X @ dX_inv                                                   # (n_int, 6, 6)
                da[jid] = (-inv_dX_inv) @ gravity_vec                                         # (n_int, 6)
                cross = inv_dX_inv[:, None] @ dX_inv[None, :]                                 # (n_int, n_int, 6, 6)
                d2a[jid] = (cross + cross.transpose(1, 0, 2, 3) - inv_X @ d2X[jid] @ inv_X) @ gravity_vec
            else:
                a[jid] = X[jid] @ a[pid]
                da[jid] = dX[jid] @ a[pid] + da[pid] @ X[jid].T
                cross = (dX[jid] @ da[pid].T).transpose(0, 2, 1)                              # (n_int, n_int, 6)
                d2a[jid] = d2X[jid] @ a[pid] + cross + cross.transpose(1, 0, 2) + d2a[pid] @ X[jid].T

        # f = Imat @ a as a batched GEMM. Reshape d2a's (k, l) dirs into a single
        # flat axis so the contraction is a clean (NB, *, 6) @ (NB, 6, 6) matmul.
        Imats_T = np.transpose(Imats, (0, 2, 1))                                              # (NB, 6, 6)
        f = (Imats @ a[..., None]).squeeze(-1)                                                # (NB, 6)
        df = da @ Imats_T                                                                     # (NB, n_int, 6)
        d2f = (d2a.reshape(NB, n_int * n_int, 6) @ Imats_T).reshape(NB, n_int, n_int, 6)      # (NB, n_int, n_int, 6)

        # Backward sweep: project onto S, then accumulate parent f-derivatives.
        d2tau_dq = np.zeros((n_int, n_int, n_int))
        for jid in range(NB - 1, -1, -1):
            S = np.asarray(self.robot.get_S_by_id(jid))
            if S.ndim == 1:
                S = S.reshape(6, 1)
            # Use the internal per-body slot for the output (axis 0) row so
            # mimic / target body contributions live in distinct slots; the
            # final fold below collapses them with the URDF multiplier.
            d2tau_dq[body_v_inds[jid], :, :] = (d2f[jid] @ S).transpose(2, 0, 1)
            pid = self.robot.get_parent_id(jid)
            if pid == -1:
                continue
            Xt = X[jid].T
            dXt = np.transpose(dX[jid], (0, 2, 1))
            d2Xt = np.transpose(d2X[jid], (0, 1, 3, 2))
            f[pid] += Xt @ f[jid]
            df[pid] += dXt @ f[jid] + df[jid] @ Xt.T
            cross_f = (dXt @ df[jid].T).transpose(0, 2, 1)                                    # (n_int, n_int, 6)
            d2f[pid] += d2Xt @ f[jid] + cross_f + cross_f.transpose(1, 0, 2) + d2f[jid] @ Xt.T

        if has_mimic:
            # Fold each axis (output torque + two q-derivative axes) from the
            # per-body unique slot layout to the project's reduced (nv) layout
            # with the URDF mimic multiplier on each axis.
            R = np.zeros((n_int, n))
            for b in range(NB):
                a_b = body_alpha[b]
                for local_idx, int_slot in enumerate(body_v_inds[b]):
                    true_slot = true_v_inds[b][local_idx]
                    R[int_slot, true_slot] += a_b
            d2tau_dq = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dq, R, R, optimize=True)

        return d2tau_dq

    def idsva_so_body_frame(self, q, qd, qdd, GRAVITY = -9.81):
        """Compute second-order derivatives of inverse dynamics via parallel IDSVA.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq) : tuple
            Second-order derivatives of torques and inertia matrix.
        """
        # Floating-base gravity Hessian is added analytically after the main
        # sweep; the sweep itself runs with zero gravity to avoid double-counting.
        idsva_gravity = 0.0 if self.robot.floating_base else GRAVITY
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        v = np.zeros((6,NB))
        a = np.zeros((6,NB))
        f = np.zeros((6,NB))
        Xup0 =  [None] * NB #list of transformation matrices in the world frame
        Xdown0 = [None] * NB
        IC = [None] * NB
        BC = [None] * NB
        S = [None] * NB
        Sd = [None] * NB
        vJ = np.zeros((6,NB))
        aJ = np.zeros((6,NB))
        psid = [None] * NB
        psidd = [None] * NB
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -idsva_gravity # a_base is gravity vec

        # Mimic-aware indexing strategy: when ANY actuated joint is a URDF
        # <mimic> joint, multiple bodies share the same project-layout v-slot.
        # The triple-nested algorithm below writes into per-(jid, jid, jid)
        # output cells; folding the writes inline with the proper alpha
        # multipliers requires tracking three independent axes and converting
        # every `=` to a `+= alpha_a * alpha_b * alpha_c` accumulation, with
        # special care for the row-axis `st_j_inds` fancy-index assignments
        # (which would otherwise silently lose contributions when the index
        # list contains repeats from mimic siblings). Cleaner is to run the
        # algorithm with a UNIQUE per-body internal slot allocation, then fold
        # each axis of the (n_int, n_int, n_int) output to the (nv, nv, nv)
        # project layout with alpha weights at the very end.
        true_v_inds = [self._as_index_list(self.robot.get_joint_index_v(i)) for i in range(NB)]
        has_mimic = self._has_mimic_joints()
        if has_mimic:
            # Allocate a unique block of slots per body: body 0 keeps its
            # native v-width (6 for floating-base, 1 for fixed-base 1-DoF
            # root), each subsequent body gets its own single slot. This
            # makes every internal index list collision-free regardless of
            # mimic relations.
            body_v_inds = []
            offs = 0
            for i in range(NB):
                width = len(true_v_inds[i])
                body_v_inds.append(list(range(offs, offs + width)))
                offs += width
            n_int = offs
        else:
            body_v_inds = true_v_inds
            n_int = n
        body_alpha = [self._mimic_multiplier(i) for i in range(NB)]

        def subtree_vel_inds(subtree):
            return [vi for body_id in subtree for vi in body_v_inds[body_id]]

        # forward pass
        modelNB = NB
        modelNV = n_int
        for i in range(modelNB):
            parent_i = self.robot.get_parent_id(i)
            # Mimic-aware: use `q_for_joint` so a mimic joint sees the
            # scaled+offset slice of the mimicked joint's q (which is what
            # determines its body transform).
            _q = self.robot.q_for_joint(i, q)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(_q)
          # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                if self.robot.floating_base:
                    a[:, i] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, i] = gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i])
            S[i] = self.robot.get_S_by_id(i)
            if len(S[i].shape) == 1:
                S[i] = np.reshape(S[i], (6,1))
            S[i] = Xdown0[i] @ S[i]
            # Mimic-aware joint velocity: the body's spatial joint velocity is
            # `alpha_i * qd[v_target]` for a mimic; the multiplier scales the
            # qd / qdd read from the (shared) project v-slot.
            inds_v_true = true_v_inds[i]
            alpha_i = body_alpha[i]
            _qd = alpha_i * np.atleast_1d(qd[inds_v_true])
            _qdd = alpha_i * np.atleast_1d(qdd[inds_v_true])
            vJ[:,i] = np.reshape(np.matmul(S[i], _qd), (6,))
            aJ[:,i] = self.cross_operator(v[:,i])@vJ[:,i] + np.reshape(np.matmul(S[i], _qdd), (6,))
            psid[i] = self.cross_operator(v[:,i])@S[i]
            psidd[i] = self.cross_operator(a[:,i])@S[i] + self.cross_operator(v[:,i])@psid[i]
            v[:,i] = v[:,i] + vJ[:,i]
            a[:,i] = a[:,i] + aJ[:,i]
            I = self.robot.get_Imat_by_id(i)
            IC[i] = np.array(Xup0[i]).T @ (I @ Xup0[i])
            Sd[i] = self.cross_operator(v[:,i]) @ S[i]
            BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
            f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @v[:,i]

        #backward pass: Can be parallelized across all j,d
        for i in range(modelNB-1,-1,-1):
            pi = self.robot.get_parent_id(i)
            if pi >= 0:
                    IC[pi] = IC[pi] + IC[i]
                    BC[pi] = BC[pi] + BC[i]
                    f[:, pi] = f[:, pi] + f[:, i]


        T1 = np.zeros((6,n_int))
        T2 = np.zeros((6,n_int))
        T3 = np.zeros((6,n_int))
        T4 = np.zeros((6,n_int))
        D1 = np.zeros((36,n_int))
        D2 = np.zeros((36,n_int))
        D3 = np.zeros((36,n_int))
        D4 = np.zeros((36,n_int))

        for j in range(modelNB-1,-1,-1):
            for d in range(S[j].shape[1]):
                S_d = S[j][:, d]
                Sd_d = Sd[j][:, d]
                psid_d = psid[j][:, d]
                psidd_d = psidd[j][:, d]


                Bic_phii1 =  self.dual_cross_operator(S_d)@IC[j]
                Bic_phii2 = self.icrf(IC[j] @ S_d)
                Bic_phii3 = -IC[j] @ self.cross_operator(S_d)

                Bic_phii = Bic_phii1+Bic_phii2+Bic_phii3 # almost complete

                Bic_psii_dot = 2 * 0.5 * (self.dual_cross_operator(psid_d) @ IC[j] + self.icrf(IC[j] @ psid_d) - IC[j] @ self.cross_operator(psid_d))

                dd = body_v_inds[j][d]
                A1 = self.dot_matrix(IC[j], S_d) # crf(S_d) @ IC[j] - (IC @ crm(S_d))
                A2 = Bic_psii_dot + self.dot_matrix(BC[j], S_d) # crf(S_d) @ BC[j] - (BC[j] @ crm(S_d))
                A3 = self.icrf(IC[j].T @ S_d)


                T1[:, dd] = IC[j] @ S_d
                T2[:, dd] = -BC[j].T @ S_d
                T3[:, dd] = BC[j] @ psid_d + IC[j] @ psidd_d + self.icrf(f[:, j]) @ S_d
                T4[:, dd] = BC[j] @ S_d + IC[j] @ (psid_d + Sd_d)



                D1[:, dd] = A1.flatten()
                D2[:, dd] = A2.flatten(order='F')
                D3[:, dd] = Bic_phii.flatten(order='F')
                D4[:, dd] = A3.flatten(order='F')

        dM_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dqd = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dvdq = np.zeros((modelNV,modelNV,modelNV))

        #backward pass: Can be parallelized over all j,d,k,c
        for j in range(modelNB-1,-1,-1):
            st_j = self.robot.get_subtree_by_id(j) # Subtree of j
            st_j_inds = subtree_vel_inds(st_j)
            succ_j = [i for i in st_j if i != j] # Joint successors
            succ_j_inds = subtree_vel_inds(succ_j)
            for d in range(S[j].shape[1]):
                dd = body_v_inds[j][d]
                S_d = S[j][:, d]
                Sd_d = Sd[j][:, d]
                psid_d = psid[j][:, d]
                psidd_d = psidd[j][:, d]
                ancestor_j = self.robot.get_ancestors_by_id(j)
                ancestor_j.insert(0, j)
                ancestor_j = ancestor_j[::-1]
                for k in ancestor_j:  # Assuming model['ancestors'][j] provides a list of ancestor indices
                    for c in range(S[k].shape[1]):
                        cc = body_v_inds[k][c]
                        S_c = S[k][:, c]
                        Sd_c = Sd[k][:, c]
                        psid_c = psid[k][:, c]
                        psidd_c = psidd[k][:, c]

                        # Compute temporary vectors
                        t1 = np.outer(S_d, psid_c.transpose()).flatten(order='F')
                        t2 = np.outer(S_d, S_c.transpose()).flatten(order='F')
                        t3 = np.outer(psid_d, psid_c.transpose()).flatten(order='F')
                        t4 = np.outer(S_d, psidd_c.transpose()).flatten(order='F')
                        t5 = np.outer(S_d, Sd_c + psid_c.transpose()).flatten(order='F')
                        t8 = np.outer(S_c, S_d.transpose()).flatten(order='F')
                        
                        # Computing the cross products
                        p1 = self.cross_operator(psid_c) @ S_d
                        p2 = self.cross_operator(psidd_c) @ S_d
                        
                        d2tau_dq[st_j_inds, dd, cc] = -np.dot(t3, D3[:, st_j_inds]) - np.dot(p1, T2[:, st_j_inds]) + np.dot(p2, T1[:, st_j_inds])
                        d2tau_dvdq[st_j_inds, dd, cc] = -np.dot(t1, D3[:, st_j_inds])

                        # st_j is list of all ancestors of j
                        if k < j:
                            t6 = np.outer(S_c, psid_d.transpose()).flatten(order='F')
                            t7 = np.outer(S_c, psidd_d.transpose()).flatten(order='F')
                            p3 = self.cross_operator(S_c) @ S_d
                            p4 = self.cross_operator(Sd_c + psid_c) @ S_d - 2 * self.cross_operator(psid_d) @ S_c
                            p5 = self.cross_operator(S_d) @ S_c
                            
                            d2tau_dq[st_j_inds, cc, dd] = d2tau_dq[st_j_inds, dd, cc]

                            d2tau_dqd[st_j_inds, cc, dd] = -np.dot(t2.T, D3[:, st_j_inds])
                            d2tau_dqd[st_j_inds, dd, cc] = d2tau_dqd[st_j_inds, cc, dd]
                            
                            
                            d2tau_dvdq[st_j_inds, cc, dd] = -np.dot(t6, D3[:, st_j_inds]) - np.dot(p3, T2[:, st_j_inds]) + np.dot(p4, T1[:, st_j_inds])

                            d2tau_dq[cc, st_j_inds, dd] = np.dot(t6, D2[:, st_j_inds]) + np.dot(t7, D1[:, st_j_inds]) - np.dot(p5, T3[:, st_j_inds])

                            d2tau_dvdq[cc, st_j_inds, dd] = np.dot(t6, D3[:, st_j_inds]) - np.dot(p5, T4[:, st_j_inds])


                            # S_d @ IC[j] is just T1
                            # self.dual_cross_operator(S_d) @ IC[j] is first part of D1
                            # Reuse these in CUDA
                            # Scatter into every (cc, dd, ee) cell where e is a
                            # DoF-column of the SAME joint j as d: the trailing
                            # motion-subspace factor is S_e (the third tensor
                            # axis), distinct from the row factor S_d for a
                            # multi-DoF (spherical/ball) joint. For a 1-DoF joint
                            # this loop runs once with ee==dd, S_e==S_d, exactly
                            # reproducing the original diagonal write.
                            for e in range(S[j].shape[1]):
                                ee = body_v_inds[j][e]
                                S_e = S[j][:, e]
                                d2tau_dqd[cc,dd,ee] = (S_d.T @ IC[j] @ self.cross_operator(S_c) + S_c.T @ self.dual_cross_operator(S_d) @ IC[j] )  @ S_e
                            
                            dM_dq[cc,st_j_inds,dd] = t8.T @ D4[:, st_j_inds]
                            dM_dq[st_j_inds,cc,dd] = dM_dq[cc,st_j_inds,dd]
                            
                            if succ_j_inds:
                                t9 = np.outer(S_c, Sd_d + psid_d) 
                                t9 = t9.flatten(order='F')

                                
                                d2tau_dqd[cc, succ_j_inds, dd] = np.dot(t8, D3[:, succ_j_inds])
                                d2tau_dqd[cc, dd, succ_j_inds] = d2tau_dqd[cc, succ_j_inds, dd]
                                
                                
                                d2tau_dvdq[cc, dd, succ_j_inds] = np.dot(t8, D2[:, succ_j_inds]) + np.dot(t9, D1[:, succ_j_inds])

                                d2tau_dq[cc, dd, succ_j_inds] = d2tau_dq[cc, succ_j_inds, dd]

                        if succ_j_inds:
                            d2tau_dq[dd, cc, succ_j_inds] = np.dot(t1, D2[:, succ_j_inds]) + np.dot(t4, D1[:, succ_j_inds])

                            d2tau_dqd[dd, cc, succ_j_inds] = np.dot(t2, D3[:, succ_j_inds])
                            d2tau_dqd[dd, succ_j_inds, cc] = d2tau_dqd[dd, cc, succ_j_inds]

                            d2tau_dvdq[dd, succ_j_inds, cc] = np.dot(t1, D3[:, succ_j_inds])

                            d2tau_dq[dd, succ_j_inds, cc] = d2tau_dq[dd, cc, succ_j_inds]

                            d2tau_dvdq[dd, cc, succ_j_inds] = np.dot(t2, D2[:, succ_j_inds]) + np.dot(t5, D1[:, succ_j_inds])
                            
                            
                            dM_dq[cc, dd, succ_j_inds] = np.dot(t8, D1[:, succ_j_inds])
                            dM_dq[dd, cc, succ_j_inds] = dM_dq[cc, dd, succ_j_inds]
                        
                        if k == j:
                            d2tau_dqd[st_j_inds, dd, cc] = -np.dot(t2, D1[:, st_j_inds])

        if has_mimic:
            # Fold each axis of the (n_int, n_int, n_int) unreduced output to
            # the (n, n, n) project layout. The reduction matrix `R` has
            # shape (n_int, n) with R[int_slot, true_slot] = alpha_b for the
            # body b that owns that internal slot. Applying R along each
            # axis collapses bodies that share a true v-slot (mimic+target)
            # into the same project-layout cell, with the URDF multiplier
            # scaling on every axis the cell appears on.
            R = np.zeros((n_int, n))
            for b in range(NB):
                a_b = body_alpha[b]
                for local_idx, int_slot in enumerate(body_v_inds[b]):
                    true_slot = true_v_inds[b][local_idx]
                    # Root in floating-base has alpha=1 and a 1:1 internal->true
                    # mapping; mimic bodies are 1-DoF with alpha=multiplier and
                    # their true slot equals the target's true slot.
                    R[int_slot, true_slot] += a_b
            d2tau_dq = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dq, R, R, optimize=True)
            d2tau_dqd = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dqd, R, R, optimize=True)
            d2tau_dvdq = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dvdq, R, R, optimize=True)
            dM_dq = np.einsum('ia,ijk,jb,kc->abc', R, dM_dq, R, R, optimize=True)

        if self.robot.floating_base:
            d2tau_dq = d2tau_dq + self._floating_gravity_d2tau_dq_lie_direct(q, GRAVITY)

        return d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq

    def idsva_so_world_frame(self, q, qd, qdd, GRAVITY=-9.81):
        """Single-pass IDSVA-SO reference matching spatial_v2_extended's `ID_SO_derivatives.m`.

        Implements Singh/Russell/Wensing 2023 (arXiv:2302.06001) Algorithm 1 with
        the full triple-nested ancestor walk (i over bodies, j over ancestors of
        i, k over ancestors of j) spelled out per-DoF. World-frame quantities
        throughout (`S = Xdown0 @ S_local`, `IC = Xup0^T @ I @ Xup0`).

        The root-acceleration initialisation `a[root] = -a_grav` plus the
        explicit per-DoF accumulation captures the gravity contribution to the
        Hessian natively — no separate gravity helper required.

        Output convention (matches `idsva_so`):
            (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq), shape (n, n, n) each, with
            d2tau_dvdq indexed [τ, qd, q].

        Equivalence with `idsva_so`:
            * Fixed-base: matches to machine precision (tested on iiwa14).
            * Floating-base: matches to ~1e-13 on all four output tensors at
              non-identity quaternion. The body-frame Lie-tangent convention
              is achieved by inverting `X_local[0]` for the floating-base root
              in the prepass (see comment at the `if parent == -1:` branch
              below). Validated against the project's Lie FD oracle in
              `test_floating_idsva_so_matches_lie_finite_difference`.

        Bloated by design (~5× the inner work of `idsva_so`'s subtree-broadcast
        form). Not on any hot path; not GPU-friendly.
        """
        q = self._normalize_q_input(q)
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.floating_base and self.robot.using_quaternion:
            q[3:7] = self._normalize_xyzw_quaternion(q[3:7])

        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        # MATLAB convention: a_grav is the *gravitational* acceleration (negative if
        # gravity points down); the "base acceleration" of the inertial frame is -a_grav.
        a_grav = np.zeros(6)
        a_grav[5] = GRAVITY

        Xup0 = [None] * NB
        Xdown0 = [None] * NB
        S = [None] * NB
        Sd = [None] * NB
        psid = [None] * NB
        psidd = [None] * NB
        IC = [None] * NB
        BC = [None] * NB
        v = np.zeros((6, NB))
        a = np.zeros((6, NB))
        f = np.zeros((6, NB))

        # Mimic-aware indexing: when any actuated joint is a URDF <mimic>, the
        # native v-slot per body is shared with the target's, so the assigning
        # writes below would silently overwrite contributions. Use per-body
        # UNIQUE internal slots for the inner algorithm and fold each axis to
        # the project layout (alpha-weighted) at the end. (See
        # idsva_so_body_frame for the same idiom.)
        true_v_inds = [self._as_index_list(self.robot.get_joint_index_v(i)) for i in range(NB)]
        has_mimic = self._has_mimic_joints()
        if has_mimic:
            body_v_inds = []
            offs = 0
            for i in range(NB):
                width = len(true_v_inds[i])
                body_v_inds.append(list(range(offs, offs + width)))
                offs += width
            n_int = offs
        else:
            body_v_inds = true_v_inds
            n_int = n
        body_alpha = [self._mimic_multiplier(i) for i in range(NB)]

        # Forward sweep: kinematic & dynamic per-body quantities in world frame.
        for i in range(NB):
            parent = self.robot.get_parent_id(i)
            # Mimic-aware: `q_for_joint` gives the body its scaled+offset q
            # slice when it's a mimic of another joint.
            Xmat = np.asarray(self.robot.get_Xmat_Func_by_id(i)(self.robot.q_for_joint(i, q))).reshape(6, 6)
            if parent == -1:
                # Floating-base root: the codebase's get_Xmat_Func_by_id(0) returns the
                # body-to-world spatial transform, but the algorithm expects Xup (world-to-body)
                # at this slot. Invert so the body-frame gravity at the root matches the
                # shim/Lie-tangent oracle: aB_root = inv(X_local[0]) @ g_world.
                Xup0[i] = np.linalg.inv(Xmat) if self.robot.floating_base else Xmat
                a[:, i] = -a_grav
            else:
                Xup0[i] = Xmat @ Xup0[parent]
                v[:, i] = v[:, parent]
                a[:, i] = a[:, parent]
            Xdown0[i] = np.linalg.inv(Xup0[i])

            S_local = np.asarray(self.robot.get_S_by_id(i), dtype=np.float64)
            if S_local.ndim == 1:
                S_local = S_local.reshape(6, 1)
            S[i] = Xdown0[i] @ S_local

            # Mimic-aware joint velocity: read from the true (shared) project
            # v-slot and scale by the joint's mimic multiplier (1.0 for
            # non-mimic). The S[i] @ _qd / S[i] @ _qdd product is then the
            # body's actual spatial velocity / acceleration contribution.
            inds_v_true = true_v_inds[i]
            alpha_i = body_alpha[i]
            _qd = alpha_i * np.atleast_1d(qd[inds_v_true])
            _qdd = alpha_i * np.atleast_1d(qdd[inds_v_true])
            vJ = (S[i] @ _qd).reshape(6)
            aJ = self.cross_operator(v[:, i]) @ vJ + (S[i] @ _qdd).reshape(6)
            psid[i] = self.cross_operator(v[:, i]) @ S[i]
            psidd[i] = self.cross_operator(a[:, i]) @ S[i] + self.cross_operator(v[:, i]) @ psid[i]
            v[:, i] = v[:, i] + vJ
            a[:, i] = a[:, i] + aJ

            I_body = np.asarray(self.robot.get_Imat_by_id(i))
            IC[i] = Xup0[i].T @ I_body @ Xup0[i]
            Sd[i] = self.cross_operator(v[:, i]) @ S[i]
            BC[i] = (self.dual_cross_operator(v[:, i]) @ IC[i]
                     + self.icrf(IC[i] @ v[:, i])
                     - IC[i] @ self.cross_operator(v[:, i]))
            f[:, i] = IC[i] @ a[:, i] + self.dual_cross_operator(v[:, i]) @ IC[i] @ v[:, i]

        d2tau_dq = np.zeros((n_int, n_int, n_int))
        d2tau_dqd = np.zeros((n_int, n_int, n_int))
        d2tau_dvdq = np.zeros((n_int, n_int, n_int))
        dM_dq = np.zeros((n_int, n_int, n_int))

        # Triple ancestor walk: i = body, j = ancestor-or-self of i, k = ancestor-or-self of j.
        # IC/BC/f are aggregated up to the parent at the end of each i-iteration so that
        # when we process body i, IC[i] already contains its full subtree contribution.
        for i in range(NB - 1, -1, -1):
            for p in range(S[i].shape[1]):
                S_p, Sd_p = S[i][:, p], Sd[i][:, p]
                psid_p, psidd_p = psid[i][:, p], psidd[i][:, p]
                i_v = body_v_inds[i][p]

                # Per-body-i intermediate 6x6 matrices (A0..A7) used to form
                # the per-(i,j) vectors u1..u12 inside the j-loop.
                Bic_phi = (self.dual_cross_operator(S_p) @ IC[i]
                           + self.icrf(IC[i] @ S_p)
                           - IC[i] @ self.cross_operator(S_p))
                Bic_psid = (self.dual_cross_operator(psid_p) @ IC[i]
                            + self.icrf(IC[i] @ psid_p)
                            - IC[i] @ self.cross_operator(psid_p))
                A0 = self.icrf(IC[i] @ S_p)
                A1 = self.dot_matrix(IC[i], S_p)
                A2 = 2.0 * A0 - Bic_phi
                A3 = Bic_psid + self.dot_matrix(BC[i], S_p)
                A4 = self.icrf(BC[i].T @ S_p)
                A5 = self.icrf(BC[i] @ psid_p + IC[i] @ psidd_p
                               + self.dual_cross_operator(S_p) @ f[:, i])
                A6 = self.dual_cross_operator(S_p) @ IC[i] + A0
                A7 = self.icrf(BC[i] @ S_p + IC[i] @ (psid_p + Sd_p))

                j = i
                while j >= 0:
                    for t in range(S[j].shape[1]):
                        S_t, Sd_t = S[j][:, t], Sd[j][:, t]
                        psid_t, psidd_t = psid[j][:, t], psidd[j][:, t]
                        j_v = body_v_inds[j][t]

                        # Per-(i,j) vectors. Each is a 6-vector contracted against
                        # k's spatial quantities to give scalar tensor entries.
                        u1  = A3.T @ S_t
                        u2  = A1.T @ S_t
                        u3  = A3 @ psid_t + A1 @ psidd_t + A5 @ S_t
                        u4  = A6 @ S_t
                        u5  = A2 @ psid_t + A4 @ S_t
                        u6  = Bic_phi @ psid_t + A7 @ S_t
                        u7  = A3 @ S_t + A1 @ (psid_t + Sd_t)
                        u8  = A4 @ S_t - Bic_phi.T @ psid_t
                        u9  = A0 @ S_t
                        u10 = Bic_phi @ S_t
                        u11 = Bic_phi.T @ S_t
                        u12 = A1 @ S_t

                        k = j
                        while k >= 0:
                            for r in range(S[k].shape[1]):
                                S_r, Sd_r = S[k][:, r], Sd[k][:, r]
                                psid_r, psidd_r = psid[k][:, r], psidd[k][:, r]
                                k_v = body_v_inds[k][r]

                                p1 = u11 @ psid_r
                                p2 = u8 @ psid_r + u9 @ psidd_r

                                d2tau_dq[i_v, j_v, k_v] = p2
                                d2tau_dvdq[i_v, k_v, j_v] = -p1

                                if j != i:
                                    d2tau_dq[j_v, k_v, i_v] = u1 @ psid_r + u2 @ psidd_r
                                    d2tau_dq[j_v, i_v, k_v] = d2tau_dq[j_v, k_v, i_v]
                                    d2tau_dvdq[j_v, k_v, i_v] = p1
                                    d2tau_dvdq[j_v, i_v, k_v] = u1 @ S_r + u2 @ (psid_r + Sd_r)
                                    d2tau_dqd[j_v, k_v, i_v] = u11 @ S_r
                                    d2tau_dqd[j_v, i_v, k_v] = d2tau_dqd[j_v, k_v, i_v]
                                    dM_dq[k_v, j_v, i_v] = S_r @ u12
                                    dM_dq[j_v, k_v, i_v] = dM_dq[k_v, j_v, i_v]

                                if k != j:
                                    d2tau_dq[i_v, k_v, j_v] = p2
                                    d2tau_dq[k_v, i_v, j_v] = S_r @ u3
                                    d2tau_dqd[i_v, j_v, k_v] = -u11 @ S_r
                                    d2tau_dqd[i_v, k_v, j_v] = -u11 @ S_r
                                    d2tau_dvdq[i_v, j_v, k_v] = S_r @ u5 + u9 @ (psid_r + Sd_r)
                                    d2tau_dvdq[k_v, j_v, i_v] = S_r @ u6
                                    dM_dq[k_v, i_v, j_v] = S_r @ u9
                                    dM_dq[i_v, k_v, j_v] = dM_dq[k_v, i_v, j_v]

                                    if j != i:
                                        d2tau_dq[k_v, j_v, i_v] = d2tau_dq[k_v, i_v, j_v]
                                        d2tau_dqd[k_v, i_v, j_v] = S_r @ u10
                                        d2tau_dqd[k_v, j_v, i_v] = d2tau_dqd[k_v, i_v, j_v]
                                        d2tau_dvdq[k_v, i_v, j_v] = S_r @ u7
                                    else:
                                        d2tau_dqd[k_v, j_v, i_v] = S_r @ u4
                                else:
                                    d2tau_dqd[i_v, j_v, k_v] = -u2 @ S_r
                            k = self.robot.get_parent_id(k)
                    j = self.robot.get_parent_id(j)

            # Bubble this body's subtree-aggregated IC/BC/f up to its parent.
            parent = self.robot.get_parent_id(i)
            if parent >= 0:
                IC[parent] = IC[parent] + IC[i]
                BC[parent] = BC[parent] + BC[i]
                f[:, parent] = f[:, parent] + f[:, i]

        # MATLAB's `d2tau_cross` stores [τ, q, qd]; our convention is [τ, qd, q].
        # Swap the trailing axes so the output matches `idsva_so`.
        d2tau_dvdq = d2tau_dvdq.transpose(0, 2, 1)

        if has_mimic:
            # Fold each axis of the (n_int, n_int, n_int) unreduced output to
            # (n, n, n) project layout with the URDF mimic multiplier scaling
            # per axis. See idsva_so_body_frame for the same idiom.
            R = np.zeros((n_int, n))
            for b in range(NB):
                a_b = body_alpha[b]
                for local_idx, int_slot in enumerate(body_v_inds[b]):
                    true_slot = true_v_inds[b][local_idx]
                    R[int_slot, true_slot] += a_b
            d2tau_dq = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dq, R, R, optimize=True)
            d2tau_dqd = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dqd, R, R, optimize=True)
            d2tau_dvdq = np.einsum('ia,ijk,jb,kc->abc', R, d2tau_dvdq, R, R, optimize=True)
            dM_dq = np.einsum('ia,ijk,jb,kc->abc', R, dM_dq, R, R, optimize=True)

        return d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq

    def idsva_so(self, q, qd, qdd, GRAVITY = -9.81):
        """Compute second-order derivatives of inverse dynamics.

        Dispatches at runtime by base type — body-frame is faster on fixed-base
        (~7-10x at low DOF), world-frame is faster on floating-base (2-20x).
        Both produce numerically equivalent output (validated against
        Pinocchio). Either `idsva_so_body_frame` and `idsva_so_world_frame`
        can be called directly if you want to compare; this dispatcher is
        the convenience entry point.

        Note (sm_120 / 2026-05-18): the body-vs-world crossover on the GPU is
        DOF-sensitive. At NV<=12 (iiwa14, go2 fixed-base) body wins by 7-10x;
        at NV=29 (g1 fixed-base) world is actually 6-15% faster. The current
        rule is the safe choice (preserves the large wins on the common
        low-DOF case), but a DOF-threshold refinement is on the open list
        once more high-DOF fixed-base data points exist.
        """
        if self.robot.floating_base:
            return self.idsva_so_world_frame(q, qd, qdd, GRAVITY)
        return self.idsva_so_body_frame(q, qd, qdd, GRAVITY)

    def fdsva_so(self, q, qd, u, GRAVITY = -9.81):
        """Compute second-order derivatives of forward dynamics.

        Internally delegates to `idsva_so` (dispatched body/world frame by
        base type) — mirrors the C++ codegen, which embeds the world-frame
        inner for floating-base and the body-frame inner for fixed-base.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq) : tuple
            Second-order gradients of forward dynamics.
        """
        Minv = self.minv(q)
        qdd = self.forward_dynamics(q, qd, u)
        di2_dq, di2_dqd, di2_dvdq, dm_dq = self.idsva_so(q, qd, qdd, GRAVITY)
        fd_dq, fd_dqd = self.forward_dynamics_gradient(q, qd, u)

        daba_dqdq = -np.einsum('il,ljk->ijk', Minv, di2_dq + np.einsum('ilk,lj->ijk', dm_dq, fd_dq) + np.einsum('ilk,lj->ikj', dm_dq, fd_dq))
        daba_dvdq = -np.einsum('il,ljk->ijk', Minv, di2_dvdq + np.einsum('ilk,lj->ijk', dm_dq, fd_dqd))
        daba_dvdv = -np.einsum('il,ljk->ijk', Minv, di2_dqd)
        daba_dtdq = -np.einsum('il,ljk->ijk', Minv, np.einsum('ilk,lj->ijk', dm_dq, Minv))
        return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq

    # ------------------------------------------------------------------
    # Energy regressors (sysID): kinetic / potential energy linear in pi
    # ------------------------------------------------------------------
    # The kinetic and potential energies are each EXACTLY affine in the stacked
    # standard inertial parameters pi = [pi_1; ...; pi_NB] (same 10-param/link
    # GRiD basis as `inverse_dynamics_regressor`: pi_i = [m, h(3)=m*c,
    # I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]]). These row regressors complete the
    # regressor family next to the joint-torque regressor and cross-check vs
    # pinocchio's `computeKineticEnergyRegressor` / `computePotentialEnergyRegressor`.

    def kinetic_energy_regressor(self, q, qd):
        """Kinetic-energy regressor y_KE (length 10*NB) with KE = y_KE . pi.

        KE = sum_i 1/2 v_i^T I_i v_i is linear in each link's spatial inertia, so
        the k-th column of link i is 1/2 v_i^T (dI_k) v_i with dI_k the k-th basis
        spatial inertia (`_regressor._BASIS_I`). v_i is the link's spatial velocity
        from the RNEA forward pass (body/local frame; the quadratic form is
        frame-invariant so the local-frame v_i and local-frame I_i agree with the
        world-frame value pinocchio reports). Param blocks are ordered by body id.
        """
        from ._regressor import _BASIS_I

        q = self._normalize_q_input(q)
        qd = self._normalize_v_input(qd)
        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()
        v, _a, _f = self.inverse_dynamics_fpass(q, qd, np.zeros(nv, dtype=np.float64))
        Y = np.zeros(10 * NB, dtype=np.float64)
        for i in range(NB):
            vi = np.asarray(v[:, i], dtype=np.float64)
            for k, dI in enumerate(_BASIS_I):
                Y[10 * i + k] = 0.5 * float(vi @ (dI @ vi))
        return Y

    def potential_energy_regressor(self, q, GRAVITY=-9.81):
        """Potential-energy regressor y_PE (length 10*NB) with PE = y_PE . pi.

        PE = -sum_i m_i g . p_{com,i} = -sum_i g . (m_i p_i + R_i h_i) with
        g = [0,0,GRAVITY], p_i / R_i the link-origin world position / rotation, and
        h_i = m_i c_i the first mass moment. So per link the only nonzero columns
        are: the mass column (-g . p_i) and the three first-moment columns
        (-(R_i^T g), since g . (R_i h_i) = (R_i^T g) . h_i). The six inertia
        columns are identically zero (PE is independent of the rotational inertia).
        Matches the [0,0,GRAVITY] gravity convention used by `potential_energy`
        and reuses the SAME mimic-aware homogeneous forward kinematics
        (`_world_transforms`) as `_total_mass_and_com`, so y_PE . pi reproduces
        `potential_energy` exactly.
        """
        NB = self.robot.get_num_bodies()
        g = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        Xw = self._world_transforms(q)            # 4x4 homogeneous world transforms
        Y = np.zeros(10 * NB, dtype=np.float64)
        for i in range(NB):
            R = Xw[i][:3, :3]                      # world<-body rotation
            p = Xw[i][:3, 3]                       # body-origin world position
            Y[10 * i + 0] = -float(g @ p)          # mass column
            Y[10 * i + 1:10 * i + 4] = -(R.T @ g)  # first-moment (h) columns
        return Y
