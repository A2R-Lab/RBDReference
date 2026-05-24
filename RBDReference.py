import numpy as np
import copy
import sympy as sp
np.set_printoptions(precision=4, suppress=True, linewidth=100)

class RBDReference:
    def __init__(self, robotObj):
        """Initialize RBDReference with a robot object.

        Parameters
        ----------
        robotObj : URDFparser
            An instance of the URDFparser class.

        Returns
        -------
        None : None
            None
        """
        self.robot = robotObj # instance of Robot Object class created by URDFparser
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

    def _denormalize_rnea_grad_output(self, dc_dq, dc_dqd):
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

    def integrate(self, q, v_dt):
        """Lie-group retract: q_new = q ⊕ v_dt.

        Matches Pinocchio's `pin.integrate(model, q, v_dt)` for both fixed-base
        and free-flyer + revolute joint robots (the only configurations in our
        manifest). For fixed-base this is just `q + v_dt`; for floating-base
        the first 6 v_dt components drive an SE(3) exponential update of the
        position+quaternion prefix and the remainder is a vector add.

        Convention (user-facing, matches Pinocchio):
          q     = [pos(3), quat_xyzw(4), joint_q...]    size nq
          v_dt  = [v_lin*dt(3), omega*dt(3), joint_v*dt...]  size nv
          q_new is in the same convention as q.
        """
        q = np.asarray(q, dtype=np.float64).copy()
        v_dt = np.asarray(v_dt, dtype=np.float64)
        if not self.robot.floating_base:
            return q + v_dt
        # Free-flyer prefix:
        rho = v_dt[0:3]   # v_lin * dt   (local/body frame)
        phi = v_dt[3:6]   # omega * dt   (local/body frame)
        # SE(3) exp returns local-frame (R_delta, p_delta).
        V = self._so3_V_matrix(phi)
        p_delta_local = V @ rho
        delta_quat = self._quat_exp_from_half_omega(0.5 * phi)
        # T_new = T_old * exp(twist_dt):
        #   R_new = R_old * R_delta;  p_new = p_old + R_old * p_delta_local
        R_old = self._rotation_from_quat_xyzw(q[3:7])
        q_pos_new = q[0:3] + R_old @ p_delta_local
        q_quat_new = self._normalize_xyzw_quaternion(self._quat_mul_xyzw(q[3:7], delta_quat))
        q_joints_new = q[7:] + v_dt[6:]
        return np.concatenate([q_pos_new, q_quat_new, q_joints_new])

    def dIntegrate(self, q, v_dt, with_respect_to):
        """Return the (nv, nv) Jacobian of `integrate(q, v_dt)` in tangent
        space. `with_respect_to` is 'q' or 'v' (matching Pinocchio's
        pin.ARG0 / pin.ARG1 — ARG1 is the Jacobian w.r.t. the v_dt argument,
        not w.r.t. v itself).

        Free-flyer block uses (in Pinocchio v_dt order [rho; phi] = [v_lin*dt; omega*dt]):
          ARG_q : Ad(exp(-v_dt))  — SE(3) adjoint of the inverse exponential
          ARG_v : SE(3) right-Jacobian J_r(v_dt) — with the Q(rho, phi) coupling
        Revolute joint block is identity for both. For fixed-base this
        collapses to identity overall.
        """
        del q  # unused for the closed-form free-flyer + revolute case
        nv = self.robot.get_num_vel()
        J = np.eye(nv)
        if not self.robot.floating_base:
            return J
        v_dt = np.asarray(v_dt, dtype=np.float64)
        rho = v_dt[0:3]   # v_lin * dt   (Pinocchio order: linear first)
        phi = v_dt[3:6]   # omega * dt
        if with_respect_to == "q":
            # exp(-v_dt) = (R_inv, p_inv) where R_inv = exp(-phi),
            #             p_inv = V(-phi) @ (-rho) = -V(-phi) @ rho.
            R_inv = self._so3_exp(-phi)
            V_neg = self._so3_V_matrix(-phi)
            p_inv = -V_neg @ rho
            # SE(3) Adjoint: [[R, [p]_x R], [0, R]]  in Pinocchio order [v_lin, omega].
            P_inv_x = self._so3_skew(p_inv)
            J[0:3, 0:3] = R_inv
            J[0:3, 3:6] = P_inv_x @ R_inv
            J[3:6, 0:3] = 0.0
            J[3:6, 3:6] = R_inv
            return J
        if with_respect_to == "v":
            # SE(3) right-Jacobian J_r(v_dt) in Pinocchio order [v_lin, omega]:
            #   [[J_r(phi),  Q(rho, phi)],
            #    [0,          J_r(phi)  ]]
            J_r = self._so3_right_jacobian(phi)
            Q = self._se3_Q_block(rho, phi)
            J[0:3, 0:3] = J_r
            J[0:3, 3:6] = Q
            J[3:6, 0:3] = 0.0
            J[3:6, 3:6] = J_r
            return J
        raise ValueError("with_respect_to must be 'q' or 'v'")

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

    def integrator(self, q, qd, u, dt, integrator_type: str = "euler"):
        """One time-integration step: x_{k+1} = integrator(x_k, u_k, dt).

        Returns x_kp1 of shape (nq + nv,) — concatenated [q_new, v_new] in the
        user-facing q/v convention. Supports 'euler', 'semi_implicit_euler',
        'midpoint', 'rk3', 'rk4'. For floating-base robots the q-update uses
        `self.integrate` (Lie-group retract); for fixed-base this collapses
        to `q + dt*v`.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        qdd1 = np.asarray(self.forward_dynamics(q, qd, u)).reshape(-1)
        if integrator_type == "euler":
            q_new = self.integrate(q, dt * qd)
            v_new = qd + dt * qdd1
            return np.concatenate([q_new, v_new])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            v_new = qd + dt * qdd1
            q_new = self.integrate(q, dt * v_new)
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
            stage_qdd = np.asarray(self.forward_dynamics(p_q, p_qd, u)).reshape(-1)
            qdd_list.append(stage_qdd)
            prev_qdd = stage_qdd
        accel = sum(b * qdd for b, qdd in zip(b_list, qdd_list))
        q_new = self.integrate(q, dt * qd)   # q update is Euler-style for every RK variant
        v_new = qd + dt * accel
        return np.concatenate([q_new, v_new])

    def integrator_grad(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Return [A | B] of shape (2*nv, 3*nv) — the Jacobian of the integrator
        step in tangent space. Column order is [d/dq | d/dqd | d/du] where
        d/dq is the nv-tangent perturbation of q (NOT the nq scalar
        perturbation). For fixed-base this matches the historical layout.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        nv = self.robot.get_num_vel()
        I_n = np.eye(nv)
        Z_n = np.zeros((nv, nv))

        def fd_grad_at(pq, pqd):
            J_qq, J_qv = self.forward_dynamics_grad(pq, pqd, u)
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
            qdd_si = np.asarray(self.forward_dynamics(q, qd, u)).reshape(-1)
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
        # ----- Multi-stage chain rule (Midpoint / RK3 / RK4) -----
        c_list, b_list = self._integrator_butcher(integrator_type)
        N = len(b_list)
        qdd_list = []
        D_qdd_list = []
        # Stage 1: FD at the original (q, qd).
        qdd_list.append(np.asarray(self.forward_dynamics(q, qd, u)).reshape(-1))
        J_qq, J_qv, Minv = fd_grad_at(q, qd)
        D_qdd_list.append(np.hstack([J_qq, J_qv, Minv]))  # (nv, 3*nv)
        # Subsequent stages: chain rule through self.integrate at the
        # intermediate point (q_orig perturbed by c_{i-1}*dt*v_orig).
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            prev_qdd = qdd_list[-1]
            p_q = self.integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            stage_qdd = np.asarray(self.forward_dynamics(p_q, p_qd, u)).reshape(-1)
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
        A =  self.dual_cross_operator(v) @ I - I @ self.cross_operator(v)
        scale_factor = 10**-15
        A = A / scale_factor
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
        #helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
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
        # helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
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
        # Flatten S to 1-D so each S[k] is a scalar: a 6x1 subspace column
        # arrives as shape (6,1), making S[k] a 1-element array. Passing that
        # as the scalar `alpha` to mx1-mx6 triggers NumPy's "ndim>0 to scalar"
        # DeprecationWarning (will error in a future NumPy) on every element write.
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

    """
    End Effector Posiitons

    offests is an array of np matricies of the form (offset_x, offset_y, offset_z, 1)
    
    TODO: Add and test floating base support.
    """

    def _normalize_ee_offsets(self, offsets=None):
        if offsets is None:
            offsets = [np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)]
        return [
            np.asarray(offset, dtype=np.float64).reshape(4, 1)
            for offset in offsets
        ]

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

        # Extract the end-effector position with the given offset(s)
        # TODO handle different offsets for different branches
        def eePos_from_Xmat_hom(Xmat_hom, ee_offsets):
            # xyz position is easy
            eePos_xyz1 = np.matmul(np.asarray(Xmat_hom, dtype=np.float64), ee_offsets[0])

            # roll pitch yaw is a bit more difficult
            eePos_roll = np.arctan2(Xmat_hom[2,1],Xmat_hom[2,2])
            pitch_temp = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
            eePos_pitch = np.arctan2(-Xmat_hom[2,0],pitch_temp)
            eePos_yaw = np.arctan2(Xmat_hom[1,0],Xmat_hom[0,0])
            eePos_rpy = np.array([[eePos_roll], [eePos_pitch], [eePos_yaw]], dtype=np.float64)

            # then stack it up!
            eePos = np.vstack((eePos_xyz1[:3,:],eePos_rpy))
            return eePos

        # do the actual computations
        ee_offsets = self._normalize_ee_offsets(ee_offsets)
        eePos_arr = []
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        for jid in ee_jids:
            # Xmat_hom = forwardChain(self, jid, q)                
            Xmat_hom = backwardChain(self, jid, q)
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offsets)
            eePos_arr.append(eePos)
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            if fj.parent_name == -1:
                Xmat_hom = fj.get_transformation_matrix_hom()
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                Xmat_hom = backwardChain(self, parent.get_id(), q, fj.get_transformation_matrix_hom())
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offsets)
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
        """Compute the Jacobian (gradient) of the end-effector pose.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        J : numpy.ndarray
            Gradient of the end-effector pose.
        """
        q = self._normalize_kinematics_q(q)
        ee_offsets = self._normalize_ee_offsets(ee_offsets)
        n = len(q)

        # local helper for handling scalar joints and the floating-base root
        def qinds_to_list(inds):
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def qeval_arg(jid, q):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            q_block = np.asarray(q[inds], dtype=np.float64)
            if q_block.size == 1:
                return float(q_block[0])
            return q_block

        def get_dlocal_info(jid, dind):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            if dind not in inds:
                return None
            return inds.index(dind)

        # chain up the transforms (version 1 for starting from the root)
        def dforward_chain(self, jidChain, dind, q):
            Xmat_hom = np.eye(4)
            dXmat_hom = np.eye(4)
            for ind in jidChain:
                dlocal_ind = get_dlocal_info(ind, dind)
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                if dlocal_ind is not None: # use derivative
                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                    dXmat_hom = np.matmul(dXmat_hom,dcurrX)
                else: # use normal transform
                    dXmat_hom = np.matmul(dXmat_hom,currX)
                Xmat_hom = np.matmul(Xmat_hom,currX)
            return Xmat_hom, dXmat_hom

        # chain up the transforms (version 2 for starting from the leaf)
        def dbackward_chain(self, jid, dind, q, finalXmat_hom = np.eye(4)):
            currId = jid
            Xmat_hom = finalXmat_hom
            dXmat_hom = finalXmat_hom
            while(currId != -1):
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                dcurrX = currX
                dlocal_ind = get_dlocal_info(currId, dind)
                if dlocal_ind is not None:
                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                dXmat_hom = np.matmul(dcurrX,dXmat_hom)
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)
            return Xmat_hom, dXmat_hom
        
        # Extract the end-effector position with the given offset(s)
        def deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets):
            # Then extract the end-effector position with the given offset(s)
            # TODO handle different offsets for different branches

            # xyz position is easy
            deePos_xyz1 = np.matmul(np.asarray(dXmat_hom, dtype=np.float64), ee_offsets[0])

            # roll pitch yaw is a bit more difficult
            # note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            def darctan2(y,x,y_prime,x_prime):
                return (-x_prime*y + x*y_prime)/(x*x + y*y)
            # in other words for each atan2 we are plugging in Xmat_hom and dXmat_hom at 
            # those indicies into the spots as specified by that equation.
            # also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            deePos_roll = darctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom[2,1],dXmat_hom[2,2])
            pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
            dpitch_sqrt_term = (Xmat_hom[2,2]*dXmat_hom[2,2] + Xmat_hom[2,1]*dXmat_hom[2,1])/pitch_sqrt_term # note canceled out the 2 in the numer and denom
            deePos_pitch = darctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom[2,0],dpitch_sqrt_term)
            deePos_yaw = darctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom[1,0],dXmat_hom[0,0])
            deePos_rpy = np.array([[deePos_roll], [deePos_pitch], [deePos_yaw]], dtype=np.float64)

            # then stack it up!
            deePos_col = np.vstack((deePos_xyz1[:3,:],deePos_rpy))
            return deePos_col

        # Then compute the gradients for each end-effector requested
        # -> For each branch chain up the transformations across all possible derivatives
        deePos_arr = []
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        # First for the standard joints
        for jid in ee_jids:
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            jidChainQInds = []
            for ind in jidChain:
                jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))
            # then compute the gradients
            deePos = None
            for dind in range(n):
                # Note: if not in branch then 0
                if dind not in jidChainQInds:
                    deePos_col = np.zeros((6,1))
                    deePos = self.equals_or_hstack(deePos,deePos_col)
                else:
                    # chain up the transforms (2 options)
                    # Xmat_hom, dXmat_hom = dforward_chain(self, jidChain, dind, q)
                    Xmat_hom, dXmat_hom = dbackward_chain(self, jid, dind, q)
                    deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets)
                    deePos = self.equals_or_hstack(deePos,deePos_col)
            deePos_arr.append(deePos)
        # Then for the fixed joints
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            if fj.parent_name == -1:
                deePos = None
                Xmat_hom = fj.get_transformation_matrix_hom()
                for _dind in range(n):
                    deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, np.zeros((4,4)), ee_offsets)
                    deePos = self.equals_or_hstack(deePos, deePos_col)
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                # first get the joints in the chain
                jidChain = sorted(self.robot.get_ancestors_by_id(parent.get_id()))
                jidChain.append(parent.get_id())
                jidChainQInds = []
                for ind in jidChain:
                    jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))
                # then compute the gradients
                deePos = None
                for dind in range(n):
                    # Note: if not in branch then 0
                    if dind not in jidChainQInds:
                        deePos_col = np.zeros((6,1))
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                    else:
                        Xmat_hom, dXmat_hom = dbackward_chain(self, parent.get_id(), dind, q, fj.get_transformation_matrix_hom())
                        deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets)
                        deePos = self.equals_or_hstack(deePos,deePos_col)
            deePos_arr.append(deePos)
        return deePos_arr

    """
    End Effector Pose Hessian
    """
    def end_effector_pose_hessian(self, q, offsets = None, ee_joint_names = None):
        """Compute the Hessian of the end-effector pose.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        H : numpy.ndarray
            Hessian of the end-effector pose.
        """
        q = self._normalize_kinematics_q(q)
        offsets = self._normalize_ee_offsets(offsets)
        n = len(q)

        # local helper for handling scalar joints and the floating-base root
        def qinds_to_list(inds):
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def qeval_arg(jid, q):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            q_block = np.asarray(q[inds], dtype=np.float64)
            if q_block.size == 1:
                return float(q_block[0])
            return q_block

        def get_dlocal_info(jid, dind):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            if dind not in inds:
                return None
            return inds.index(dind)

        # Determine which targets to compute
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        if ee_joint_names is None:
            ee_jids = self.robot.get_leaf_nodes()
            fixed_jids = []

        # For each branch chain up the transformations across all possible derivatives
        # Note: if not in branch then 0
        d2eePos_arr = []
        for jid in ee_jids: # can be done in parallel
            
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            jidChainQInds = []
            for ind in jidChain:
                jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))

            # first chain up the 1st derivative transforms
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                # n+1 for standard as well (noting that n+1 will never trigger derivative)
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                for ind in jidChain:
                    currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                    dlocal_ind = get_dlocal_info(ind, dind)
                    if dlocal_ind is not None: # use derivative
                        dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],dcurrX)
                    else: # use normal transform
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],currX)

            # chain up the 1st derivative transforms (version 2 for starting from the leaf)
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                currId = jid
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                while(currId != -1):
                    dlocal_ind = get_dlocal_info(currId, dind)
                    if dlocal_ind is not None:
                        dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                    else:
                        dcurrX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                    dXmat_hom_arr[:,:,dind] = np.matmul(dcurrX,dXmat_hom_arr[:,:,dind])
                    currId = self.robot.get_parent_id(currId)

            # then chain up the 2nd derivative transforms
            d2Xmat_hom_arr = np.zeros((4,4,n,n))
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    for ind in jidChain:
                        dlocal_ind_i = get_dlocal_info(ind, dind_i)
                        dlocal_ind_j = get_dlocal_info(ind, dind_j)
                        if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                            if (dlocal_ind_i is not None) and (dlocal_ind_j is not None): # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(ind, dlocal_ind_i, dlocal_ind_j)(qeval_arg(ind, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],d2currX)
                            else: # use first derivative values
                                dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],dcurrX)
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                            d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],currX)

            # then chain up the 2nd derivative transforms (version 2 - backward)
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    currId = jid
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    while(currId != -1):
                        dlocal_ind_i = get_dlocal_info(currId, dind_i)
                        dlocal_ind_j = get_dlocal_info(currId, dind_j)
                        if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                            if (dlocal_ind_i is not None) and (dlocal_ind_j is not None): # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(currId, dlocal_ind_i, dlocal_ind_j)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            else: # use first derivative values
                                dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(dcurrX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                            d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                        currId = self.robot.get_parent_id(currId)

            # Then extract the end-effector position with the given offset(s)
            # TODO handle different offsets for different branches
            d2eePos = np.zeros((6,n,n))
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel

                    # point to the correct transforms
                    Xmat_hom = dXmat_hom_arr[:,:,n]
                    dXmat_hom_i = dXmat_hom_arr[:,:,dind_i]
                    dXmat_hom_j = dXmat_hom_arr[:,:,dind_j]
                    d2Xmat_hom = d2Xmat_hom_arr[:,:,dind_i,dind_j]

                    # Note: if not in branch then 0
                    if (dind_i not in jidChainQInds) or (dind_j not in jidChainQInds):
                        d2eePos[:,dind_i,dind_j] = np.zeros((6,))
                    
                    else:
                        # xyz position is easy
                        d2eePos_xyz1 = np.matmul(np.asarray(d2Xmat_hom, dtype=np.float64), offsets[0])

                        # roll pitch yaw is a bit more difficult
                        # note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
                        def darctan2(y,x,y_prime,x_prime):
                            return (-x_prime*y + x*y_prime)/(x*x + y*y)
                        # then another chain / quotient rule
                        def quotient_rule(top,bottom,dtop,dbottom):
                            return (bottom*dtop - top*dbottom) / (bottom*bottom)
                        def d2arctan2(y,x,y_prime_i,x_prime_i,y_prime_j,x_prime_j,y_prime_prime,x_prime_prime,i,j):
                            top = -x_prime_i*y + x*y_prime_i
                            dtop = -x_prime_prime*y + x*y_prime_prime
                            if (i != j):
                                dtop = dtop + (-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
                            bottom = x*x + y*y
                            dbottom = 2*x*x_prime_j + 2*y*y_prime_j
                            return quotient_rule(top,bottom,dtop,dbottom)
                        # in other words for each atan2 we are plugging in Xmat_hom and dXmat_hom at 
                        # those indicies into the spots as specified by that equation.
                        # also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
                        d2eePos_roll = d2arctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom_i[2,1],dXmat_hom_i[2,2], \
                                        dXmat_hom_j[2,1],dXmat_hom_j[2,2],d2Xmat_hom[2,1],d2Xmat_hom[2,2],dind_i,dind_j)

                        pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
                        dpitch_sqrt_term_i_top = Xmat_hom[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,1]*dXmat_hom_i[2,1]
                        dpitch_sqrt_term_i = dpitch_sqrt_term_i_top/pitch_sqrt_term # note canceled out the 2 in the numer and denom
                        dpitch_sqrt_term_j_top = Xmat_hom[2,2]*dXmat_hom_j[2,2] + Xmat_hom[2,1]*dXmat_hom_j[2,1]
                        dpitch_sqrt_term_j = dpitch_sqrt_term_j_top/pitch_sqrt_term # note canceled out the 2 in the numer and denom
                        # d2pitch_sqrt_term is quotient rule of dpitch_sqrt_term_i
                        # top = dpitch_sqrt_term_i, bottom = pitch_sqrt_term, dtop = dpitch_sqrt_term_i_top_dj, dbottom = dpitch_sqrt_term_j
                        dpitch_sqrt_term_i_top_dj = dXmat_hom_j[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,2]*d2Xmat_hom[2,2] + \
                                                    dXmat_hom_j[2,1]*dXmat_hom_i[2,1] + Xmat_hom[2,1]*d2Xmat_hom[2,1]
                        d2pitch_sqrt_term = quotient_rule(dpitch_sqrt_term_i,pitch_sqrt_term,dpitch_sqrt_term_i_top_dj,dpitch_sqrt_term_j)
                        d2eePos_pitch = d2arctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom_i[2,0],dpitch_sqrt_term_i, \
                                         -dXmat_hom_j[2,0],dpitch_sqrt_term_j,-d2Xmat_hom[2,0],d2pitch_sqrt_term,dind_i,dind_j)
                        d2eePos_yaw = d2arctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom_i[1,0],dXmat_hom_i[0,0], \
                                        dXmat_hom_j[1,0],dXmat_hom_j[0,0],d2Xmat_hom[1,0],d2Xmat_hom[0,0],dind_i,dind_j)
                        d2eePos_rpy = np.array([[d2eePos_roll], [d2eePos_pitch], [d2eePos_yaw]], dtype=np.float64)

                        # then stack it up!
                        d2eePos_col = np.vstack((d2eePos_xyz1[:3,:],d2eePos_rpy))
                        d2eePos[:,dind_i,dind_j] = d2eePos_col.reshape((6,))

            d2eePos_arr.append(d2eePos)

        # Then for the fixed joints
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            d2eePos = np.zeros((6,n,n))
            if fj.parent_name != -1:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                # first get the joints in the chain
                jidChain = sorted(self.robot.get_ancestors_by_id(parent.get_id()))
                jidChain.append(parent.get_id())
                jidChainQInds = []
                for ind in jidChain:
                    jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))

                # first chain up the 1st derivative transforms
                dXmat_hom_arr = np.zeros((4,4,n+1))
                for dind in range(n+1):
                    dXmat_hom_arr[:,:,dind] = fj.get_transformation_matrix_hom() if dind == n else fj.get_transformation_matrix_hom()
                    currId = parent.get_id()
                    while(currId != -1):
                        dlocal_ind = get_dlocal_info(currId, dind)
                        if dlocal_ind is not None:
                            dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                        else:
                            dcurrX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                        dXmat_hom_arr[:,:,dind] = np.matmul(dcurrX,dXmat_hom_arr[:,:,dind])
                        currId = self.robot.get_parent_id(currId)

                # then chain up the 2nd derivative transforms (version 2 - backward)
                d2Xmat_hom_arr = np.zeros((4,4,n,n))
                for dind_i in range(n):
                    for dind_j in range(n):
                        d2Xmat_hom_arr[:,:,dind_i,dind_j] = fj.get_transformation_matrix_hom()
                        currId = parent.get_id()
                        while(currId != -1):
                            dlocal_ind_i = get_dlocal_info(currId, dind_i)
                            dlocal_ind_j = get_dlocal_info(currId, dind_j)
                            if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                                if (dlocal_ind_i is not None) and (dlocal_ind_j is not None):
                                    d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(currId, dlocal_ind_i, dlocal_ind_j)(qeval_arg(currId, q))
                                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                                else:
                                    dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(dcurrX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            else:
                                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            currId = self.robot.get_parent_id(currId)

                for dind_i in range(n):
                    for dind_j in range(n):
                        Xmat_hom = dXmat_hom_arr[:,:,n]
                        dXmat_hom_i = dXmat_hom_arr[:,:,dind_i]
                        dXmat_hom_j = dXmat_hom_arr[:,:,dind_j]
                        d2Xmat_hom = d2Xmat_hom_arr[:,:,dind_i,dind_j]

                        if (dind_i not in jidChainQInds) or (dind_j not in jidChainQInds):
                            d2eePos[:,dind_i,dind_j] = np.zeros((6,))
                        else:
                            d2eePos_xyz1 = np.matmul(np.asarray(d2Xmat_hom, dtype=np.float64), offsets[0])

                            def quotient_rule(top,bottom,dtop,dbottom):
                                return (bottom*dtop - top*dbottom) / (bottom*bottom)
                            def d2arctan2(y,x,y_prime_i,x_prime_i,y_prime_j,x_prime_j,y_prime_prime,x_prime_prime,i,j):
                                top = -x_prime_i*y + x*y_prime_i
                                dtop = -x_prime_prime*y + x*y_prime_prime
                                if (i != j):
                                    dtop = dtop + (-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
                                bottom = x*x + y*y
                                dbottom = 2*x*x_prime_j + 2*y*y_prime_j
                                return quotient_rule(top,bottom,dtop,dbottom)

                            d2eePos_roll = d2arctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom_i[2,1],dXmat_hom_i[2,2], \
                                            dXmat_hom_j[2,1],dXmat_hom_j[2,2],d2Xmat_hom[2,1],d2Xmat_hom[2,2],dind_i,dind_j)

                            pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
                            dpitch_sqrt_term_i_top = Xmat_hom[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,1]*dXmat_hom_i[2,1]
                            dpitch_sqrt_term_i = dpitch_sqrt_term_i_top/pitch_sqrt_term
                            dpitch_sqrt_term_j_top = Xmat_hom[2,2]*dXmat_hom_j[2,2] + Xmat_hom[2,1]*dXmat_hom_j[2,1]
                            dpitch_sqrt_term_j = dpitch_sqrt_term_j_top/pitch_sqrt_term
                            dpitch_sqrt_term_i_top_dj = dXmat_hom_j[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,2]*d2Xmat_hom[2,2] + \
                                                        dXmat_hom_j[2,1]*dXmat_hom_i[2,1] + Xmat_hom[2,1]*d2Xmat_hom[2,1]
                            d2pitch_sqrt_term = quotient_rule(dpitch_sqrt_term_i,pitch_sqrt_term,dpitch_sqrt_term_i_top_dj,dpitch_sqrt_term_j)
                            d2eePos_pitch = d2arctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom_i[2,0],dpitch_sqrt_term_i, \
                                             -dXmat_hom_j[2,0],dpitch_sqrt_term_j,-d2Xmat_hom[2,0],d2pitch_sqrt_term,dind_i,dind_j)
                            d2eePos_yaw = d2arctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom_i[1,0],dXmat_hom_i[0,0], \
                                            dXmat_hom_j[1,0],dXmat_hom_j[0,0],d2Xmat_hom[1,0],d2Xmat_hom[0,0],dind_i,dind_j)
                            d2eePos_rpy = np.array([[d2eePos_roll], [d2eePos_pitch], [d2eePos_yaw]], dtype=np.float64)

                            d2eePos_col = np.vstack((d2eePos_xyz1[:3,:],d2eePos_rpy))
                            d2eePos[:,dind_i,dind_j] = d2eePos_col.reshape((6,))

            d2eePos_arr.append(d2eePos)
        return d2eePos_arr
    
    def apply_external_forces(self, q, f_in, f_ext):
        """Distribute externally applied forces to the internal rigid body force array.

        Parameters
        ----------
        f_ext_total : numpy.ndarray
            Array of external forces applied to the robot.

        Returns
        -------
        f_ext : numpy.ndarray
            6N-element array of spatial forces per link.
        """
        f_out = f_in
        NB = self.robot.get_num_bodies()
        if len(f_ext) > 0:
            for curr_id in range(NB):
                parent_id = self.robot.get_parent_id(curr_id)
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                if parent_id == -1:
                    Xa = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                else:
                    Xa = np.matmul(self.robot.get_Xmat_Func_by_id(curr_id)(curr_id),Xa) 
                if len(f_ext[curr_id]) > 1:
                    f_out[curr_id] -= np.matmul(np.linalg.inv(Xa.T), f_ext[curr_id])
        return f_out

    def rnea_fpass(self, q, qd, qdd=None, GRAVITY=-9.81):
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
            inds_q = self.robot.get_joint_index_q(curr_id)
            _q = q[inds_q]
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
            _qd = qd[inds_v]
            
            if self.robot.floating_base and curr_id == 0:
                vJ = np.matmul(S, np.asarray(_qd, dtype=np.float64).reshape(-1, 1))
            else: vJ = S * _qd
            v[:, curr_id] += np.squeeze(np.array(vJ))  # reduces shape to (6,) matching v[:,curr_id]
            a[:, curr_id] += self.mxS(vJ, v[:, curr_id])
            if qdd is not None:
                _qdd = qdd[inds_v]
                if self.robot.floating_base and curr_id == 0:
                    aJ = np.matmul(S, np.asarray(_qdd, dtype=np.float64).reshape(-1, 1))
                else: aJ = S * _qdd
                a[:, curr_id] += np.squeeze(np.array(aJ))  # reduces shape to (6,) matching a[:,curr_id]
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:, curr_id] = np.matmul(Imat, a[:, curr_id]) + self.vxIv(v[:, curr_id], Imat)

        return (v, a, f)

    def rnea_bpass(self, q, f):
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
            # compute c
            c[inds_f] = np.matmul(np.transpose(S), f[:, curr_id])
            # update f if applicable
            if parent_id != -1:
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                temp = np.matmul(np.transpose(Xmat), f[:, curr_id])
                f[:, parent_id] = f[:, parent_id] + temp.flatten()

        return (c, f)

    def rnea(
        self,
        q,
        qd,
        qdd=None,
        GRAVITY=-9.81,
        f_ext=None,
        public_output=True,
        normalize_input=True,
    ):
        """Compute the generalized forces using Recursive Newton-Euler Algorithm.

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
        # forward pass
        (v, a, f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c, f) = self.rnea_bpass(q, f)
        if public_output:
            c = self._denormalize_v_output(c)
        return (c, v, a, f)

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
        # Allocate memory
        NB = self.robot.get_num_bodies()
        if self.robot.floating_base:
            n = NB + 5  # count fb_joint as 6 instead of 1 joint else set n = len(qd)
        else:
            n = self.robot.get_num_vel()
        Minv = np.zeros((n, n))
        F = np.zeros((n, 6, n))
        U = np.zeros((n, 6))
        Dinv = np.zeros(n)

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

        # # Backward pass
        for ind in range(NB - 1, -1, -1):
            subtreeInds = self.robot.get_subtree_by_id(ind)
            if self.robot.floating_base:
                matrix_ind = ind + 5  # use for Minv, F, U, Dinv
                adj_subtreeInds = list(
                    np.array(subtreeInds) + 5
                )  # adjusted for matrix calculation
            else:
                matrix_ind = ind
                adj_subtreeInds = subtreeInds
            parent_ind = self.robot.get_parent_id(ind)
            if (
                parent_ind == -1 and self.robot.floating_base
            ):  # floating base joint check
                # Compute U, D
                S = self.robot.get_S_by_id(ind)  # np.eye(6) for floating base
                U[ind : ind + 6, :] = np.matmul(IA[ind], S)
                fb_Dinv = np.linalg.inv(
                    np.matmul(S.transpose(), U[ind : ind + 6, :])
                )  # vectorized Dinv calc
                # Update Minv and subtrees - subtree calculation for Minv -= Dinv * S.T * F with clever indexing
                Minv[ind : ind + 6, ind : ind + 6] = Minv[ind, ind] + fb_Dinv
                Minv[ind : ind + 6, adj_subtreeInds] -= (
                    np.matmul(
                        np.matmul(fb_Dinv, S), F[ind : ind + 6, :, adj_subtreeInds]
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
                    if self.robot.floating_base:
                        matrix_parent_ind = parent_ind + 5
                    else:
                        matrix_parent_ind = parent_ind
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
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
        # # Forward pass
        for ind in range(NB):
            if self.robot.floating_base:
                matrix_ind = ind + 5
            else:
                matrix_ind = ind
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
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
                    F[ind] = np.matmul(S, Minv[ind : ind + 6, ind:])
                else:
                    F[ind] = np.outer(S, Minv[ind, :])

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
        if self.robot.floating_base:
            # allocate memory TODO check NB vs. n
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

            # apply external forces
            pA = self.apply_external_forces(q, pA, f_ext)

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
                u[inds_v] = tau[inds_v] - (np.matmul(S.T, pA[:, ind])) - (np.matmul(U[:, inds_v].T, c[:, ind]))

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
            
            for ind in range(n-1,-1,-1):
                S = self.robot.get_S_by_id(ind)
                parent_ind = self.robot.get_parent_id(ind)

                U[:,ind] = np.squeeze(np.array(np.matmul(IA[:,:,ind],S)))
                d[ind] = np.matmul(np.transpose(S),U[:,ind])
                u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

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




    def crba(self, q, normalize_input=True):
        """Compute the joint-space inertia matrix using the Composite Rigid Body Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.

        Returns
        -------
        M : numpy.ndarray
            N x N joint-space inertia matrix.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        if self.robot.floating_base:
            NB = self.robot.get_num_bodies()
            n = self.robot.get_num_vel()
            H = np.zeros((n, n))

            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            for ind in range(NB - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                matrix_ind = ind + 5
                if ind > 0:
                    _q = q[self.robot.get_joint_index_q(ind)]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )
                    fh = np.matmul(IC[ind], S)
                    H[matrix_ind, matrix_ind] = np.matmul(S.T, fh)
                    j = ind
                    while self.robot.get_parent_id(j) > 0:
                        Xmat = self.robot.get_Xmat_Func_by_id(j)(
                            q[self.robot.get_joint_index_q(j)]
                        )
                        fh = np.matmul(Xmat.T, fh)
                        j = self.robot.get_parent_id(j)
                        S = self.robot.get_S_by_id(j)
                        H[matrix_ind, j + 5] = np.matmul(fh.T, S)
                        H[j + 5, matrix_ind] = H[matrix_ind, j + 5]
                    # # treat floating base 6 dof joint
                    inds_q = self.robot.get_joint_index_q(j)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(_q)
                    S = np.eye(6)
                    fh = np.matmul(Xmat.T, fh)
                    H[matrix_ind, :6] = np.matmul(fh.T, S)
                    H[:6, matrix_ind] = H[matrix_ind, :6].T
                else:
                    ind = 0
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    parent_ind = self.robot.get_parent_id(ind)
                    fh = np.matmul(IC[ind], S)
                    H[:6, :6] = np.matmul(S.T, fh)

            # keep the user-facing floating-base convention in one place:
            # the root block already matches the Pinocchio-style ordering,
            # but the root-to-joint cross terms are still accumulated in the
            # internal spatial row order [wx, wy, wz, vx, vy, vz].
            root_order = [3, 4, 5, 0, 1, 2]
            root_cross = H[:6, 6:].copy()
            H[:6, 6:] = root_cross[root_order, :]
            H[6:, :6] = np.transpose(H[:6, 6:])
        else:
            # # Fixed base implmentation of CRBA
            n = len(q)
            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            for ind in range(n - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

                if parent_ind != -1:
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )

            H = np.zeros((n, n))

            for ind in range(n):
                S = self.robot.get_S_by_id(ind)
                fh = np.matmul(IC[ind], S)
                H[ind, ind] = np.matmul(S.T, fh)
                j = ind

                while self.robot.get_parent_id(j) > -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                    
                    fh = np.matmul(Xmat.T, fh) # add an addition Xmat.T everytime
                    j = self.robot.get_parent_id(j)

                    
                    S = self.robot.get_S_by_id(j)
                    H[ind, j] = np.matmul(S.T, fh)
                    H[j, ind] = H[ind, j]

        return self._denormalize_qv_matrix_output(H, row_space="v", col_space="v")

    ##### Testing original RNEA_grad to help with CUDA 
    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
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

        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            if self.robot.floating_base: 
                # dc_dqd gets idx
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else:
                idx = ind
                parent_idx = parent_ind
            # Xmat access sequence
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,idx,ind] += self._mxS(S,np.matmul(Xmat,v[:,parent_ind])) # replace with new mxS
                
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            for c in range(n):
                if parent_ind == -1 and self.robot.floating_base:
                    # dv_dq should be all 0s => this results in all 0s
                    for ii in range(len(idx)):
                        da_dq[:,c,ii] += self._mxS(S[ii],dv_dq[:,c,ii],qd[ii]) # dv/du x S*q
                else:
                    da_dq[:,c,ind] += self._mxS(S,dv_dq[:,c,ind],qd[idx]) # replace with new mxS
                    
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,idx,ind] += self._mxS(S,np.matmul(Xmat,a[:,parent_ind])) # replace with new mxS
            else:
                if self.robot.floating_base:
                    root_gravity = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    root_gravity = np.matmul(Xmat, gravity_vec)
                da_dq[:,idx,ind] += self._mxS(S,root_gravity) # replace with new mxS 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            
            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])# puts 0.0014 instead of -0.0014 in df_dq[2,7,7]
            Iv = np.matmul(Imat,v[:,ind])
       
            for c in range(n):
               
                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))
    
        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
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

        # forward pass
        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            if self.robot.floating_base:
                # dc_dqd gets idx, special matrix indexing
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else: 
                idx = ind
                parent_idx = parent_ind
            # Xmat access sequence
            inds_v = self.robot.get_joint_index_v(ind) #joint index for all joints without quaternion (does special joint indexing by itself)
            inds_q = self.robot.get_joint_index_q(ind) #joint index for all joints
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
            dv_dqd[:,inds_v,ind] += np.squeeze(np.array(S)) # added squeeze and mxS
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
            for c in range(n): 
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        da_dqd[:,c,ind] += self._mxS(S[ii],dv_dqd[:,c,ind],qd[ii]) 
                else:
                    da_dqd[:,c,ind] += self._mxS(S,dv_dqd[:,c,ind],qd[idx]) 

            
            da_dqd[:,idx,ind] += self._mxS(S,v[:,ind]) 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            
            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                
                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))
        
        
        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
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
        
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)

            if self.robot.floating_base:
                # dc_dqd gets idx
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else:
                idx = ind
                parent_idx = parent_ind
            
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            if parent_ind == -1 and self.robot.floating_base:
                dc_dq[idx,:] = np.matmul(np.transpose(S),df_dq[:,:,ind])
            else:
                dc_dq[idx,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            if parent_ind != -1:
                # Xmat access sequence
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                for entry in range(6):
                    df_dq[entry,idx,parent_ind] += delta_dq[entry]
                    
            
        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
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
        
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)

            if self.robot.floating_base:
                # dc_dqd gets idx, special matrix indexing
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else: 
                idx = ind
                parent_idx = parent_ind
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            # if parent_ind == -1 and self.robot.floating_base:
            #     for ii in range(len(idx)):
            #         dc_dqd[ii,:] = np.matmul(np.transpose(S[ii]),df_dqd[:,:,ii])
            # else:
            dc_dqd[idx,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            if parent_ind != -1:
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

            
        # add in the damping and simplify this expression later
        # suggestion: have a getter function that automatically indexes and allocates for floating base functions
        if USE_VELOCITY_DAMPING:
            for ind in range(NB):
                if self.robot.floating_base and self.robot.get_parent_id(ind) == -1:
                    dc_dqd[ind:ind+5, ind:ind+5] += self.robot.get_damping_by_id(ind)
                else:
                    dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)
        
        return dc_dqd

    def rnea_grad(
        self,
        q,
        qd,
        qdd = None,
        GRAVITY = -9.81,
        USE_VELOCITY_DAMPING = False,
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
        (c, v, a, f) = self.rnea(
            q,
            qd,
            qdd,
            GRAVITY,
            public_output=False,
            normalize_input=False,
        )

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)
 
        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        if public_output:
            return self._denormalize_rnea_grad_output(dc_dq, dc_dqd)
        return np.hstack((dc_dq, dc_dqd))


    def forward_dynamics(self, q, qd, u, public_output=True, normalize_input=True):
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
        (c,v,a,f) = self.rnea(q, qd, public_output=False, normalize_input=False)
        minv = self.minv(q, public_output=False, normalize_input=False)
        qdd = np.matmul(minv, u - c)
        if public_output:
            return self._denormalize_v_output(qdd)
        return qdd
    
    def forward_dynamics_grad(self, q, qd, u, normalize_input=True):
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
        qdd = self.forward_dynamics(q, qd, u, public_output=False, normalize_input=False)
        dc_du = self.rnea_grad(q, qd, qdd, public_output=False, normalize_input=False)
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
                self.rnea_grad(q_pos, qd, qdd, GRAVITY), [n]
            )
            dc_dq_neg, _dc_dqd_neg = np.hsplit(
                self.rnea_grad(q_neg, qd, qdd, GRAVITY), [n]
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
        """
        if not self.robot.floating_base:
            raise ValueError("_floating_gravity_d2tau_dq_lie_direct requires a floating-base robot.")

        q = self._normalize_q_input(q)
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.using_quaternion:
            q[3:7] = self._normalize_xyzw_quaternion(q[3:7])

        NB, n = self.robot.get_num_bodies(), self.robot.get_num_vel()
        gravity_vec = np.zeros(6); gravity_vec[5] = -GRAVITY
        X = np.zeros((NB, 6, 6))
        dX = np.zeros((NB, n, 6, 6))
        d2X = np.zeros((NB, n, n, 6, 6))
        Imats = np.stack([np.asarray(self.robot.get_Imat_by_id(jid)) for jid in range(NB)])  # (NB, 6, 6)

        for jid in range(NB):
            q_arg = q[self.robot.get_joint_index_q(jid)]
            X[jid] = np.asarray(self.robot.get_Xmat_Func_by_id(jid)(q_arg)).reshape(6, 6)
            if not self.robot.get_joint_by_id(jid).position_symbols:
                continue
            vinds = self._as_index_list(self.robot.get_joint_index_v(jid))
            if jid == 0:
                # Root joint: Lie generators with Featherstone xlt translation-sign flip.
                S_root = np.asarray(self.robot.get_S_by_id(0))
                B = [(-1 if li < 3 else 1) * self.cross_operator(S_root[:, li]) for li in range(len(vinds))]
                for li, vi in enumerate(vinds):
                    dX[jid, vi] = X[jid] @ B[li]
                    for lj, vj in enumerate(vinds):
                        d2X[jid, vi, vj] = X[jid] @ B[lj] @ B[li]
            else:
                for li, vi in enumerate(vinds):
                    dX[jid, vi] = np.asarray(self._spatial_xmat_derivative_func(jid, li)(q_arg)).reshape(6, 6)
                    for lj, vj in enumerate(vinds):
                        d2X[jid, vi, vj] = np.asarray(
                            self._spatial_xmat_second_derivative_func(jid, li, lj)(q_arg)
                        ).reshape(6, 6)

        # Forward sweep. Per-body work is per-body sequential (parent dependency)
        # but expressed as batched matmuls/broadcasts over the n direction axis.
        a = np.zeros((NB, 6))
        da = np.zeros((NB, n, 6))
        d2a = np.zeros((NB, n, n, 6))
        for jid in range(NB):
            pid = self.robot.get_parent_id(jid)
            if pid == -1:
                inv_X = np.linalg.inv(X[jid])
                a[jid] = inv_X @ gravity_vec
                dX_inv = dX[jid] @ inv_X                                                      # (n, 6, 6)
                inv_dX_inv = inv_X @ dX_inv                                                   # (n, 6, 6)
                da[jid] = (-inv_dX_inv) @ gravity_vec                                         # (n, 6)
                cross = inv_dX_inv[:, None] @ dX_inv[None, :]                                 # (n, n, 6, 6)
                d2a[jid] = (cross + cross.transpose(1, 0, 2, 3) - inv_X @ d2X[jid] @ inv_X) @ gravity_vec
            else:
                a[jid] = X[jid] @ a[pid]
                da[jid] = dX[jid] @ a[pid] + da[pid] @ X[jid].T
                cross = (dX[jid] @ da[pid].T).transpose(0, 2, 1)                              # (n, n, 6) with axes (k, l, a)
                d2a[jid] = d2X[jid] @ a[pid] + cross + cross.transpose(1, 0, 2) + d2a[pid] @ X[jid].T

        # f = Imat @ a as a batched GEMM. Reshape d2a's (k, l) dirs into a single
        # flat axis so the contraction is a clean (NB, *, 6) @ (NB, 6, 6) matmul.
        Imats_T = np.transpose(Imats, (0, 2, 1))                               # (NB, 6, 6)
        f = (Imats @ a[..., None]).squeeze(-1)                                 # (NB, 6)
        df = da @ Imats_T                                                      # (NB, n, 6)
        d2f = (d2a.reshape(NB, n * n, 6) @ Imats_T).reshape(NB, n, n, 6)        # (NB, n, n, 6)

        # Backward sweep: project onto S, then accumulate parent f-derivatives.
        d2tau_dq = np.zeros((n, n, n))
        for jid in range(NB - 1, -1, -1):
            S = np.asarray(self.robot.get_S_by_id(jid))
            if S.ndim == 1:
                S = S.reshape(6, 1)
            d2tau_dq[self._as_index_list(self.robot.get_joint_index_f(jid)), :, :] = (d2f[jid] @ S).transpose(2, 0, 1)
            pid = self.robot.get_parent_id(jid)
            if pid == -1:
                continue
            Xt = X[jid].T
            dXt = np.transpose(dX[jid], (0, 2, 1))
            d2Xt = np.transpose(d2X[jid], (0, 1, 3, 2))
            f[pid] += Xt @ f[jid]
            df[pid] += dXt @ f[jid] + df[jid] @ Xt.T
            cross_f = (dXt @ df[jid].T).transpose(0, 2, 1)                                    # (n, n, 6)
            d2f[pid] += d2Xt @ f[jid] + cross_f + cross_f.transpose(1, 0, 2) + d2f[jid] @ Xt.T

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

        body_v_inds = [self._as_index_list(self.robot.get_joint_index_v(i)) for i in range(NB)]

        def subtree_vel_inds(subtree):
            return [vi for body_id in subtree for vi in body_v_inds[body_id]]

        # forward pass
        modelNB = NB
        modelNV = n
        for i in range(modelNB):
            parent_i = self.robot.get_parent_id(i)
            inds_q = self.robot.get_joint_index_q(i)
            _q = q[inds_q]
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
            inds_v = self.robot.get_joint_index_v(i)
            _qd = np.atleast_1d(qd[inds_v])
            _qdd = np.atleast_1d(qdd[inds_v])
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


        T1 = np.zeros((6,n))
        T2 = np.zeros((6,n))
        T3 = np.zeros((6,n))
        T4 = np.zeros((6,n))
        D1 = np.zeros((36,n))
        D2 = np.zeros((36,n))
        D3 = np.zeros((36,n))
        D4 = np.zeros((36,n))
        
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
                            d2tau_dqd[cc,dd,dd] = (S_d.T @ IC[j] @ self.cross_operator(S_c) + S_c.T @ self.dual_cross_operator(S_d) @ IC[j] )  @ S_d
                            
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

        body_v_inds = [self._as_index_list(self.robot.get_joint_index_v(i)) for i in range(NB)]

        # Forward sweep: kinematic & dynamic per-body quantities in world frame.
        for i in range(NB):
            parent = self.robot.get_parent_id(i)
            Xmat = np.asarray(self.robot.get_Xmat_Func_by_id(i)(q[self.robot.get_joint_index_q(i)])).reshape(6, 6)
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

            _qd = np.atleast_1d(qd[body_v_inds[i]])
            _qdd = np.atleast_1d(qdd[body_v_inds[i]])
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

        d2tau_dq = np.zeros((n, n, n))
        d2tau_dqd = np.zeros((n, n, n))
        d2tau_dvdq = np.zeros((n, n, n))
        dM_dq = np.zeros((n, n, n))

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
        fd_dq, fd_dqd = self.forward_dynamics_grad(q, qd, u)

        daba_dqdq = -np.einsum('il,ljk->ijk', Minv, di2_dq + np.einsum('ilk,lj->ijk', dm_dq, fd_dq) + np.einsum('ilk,lj->ikj', dm_dq, fd_dq))
        daba_dvdq = -np.einsum('il,ljk->ijk', Minv, di2_dvdq + np.einsum('ilk,lj->ijk', dm_dq, fd_dqd))
        daba_dvdv = -np.einsum('il,ljk->ijk', Minv, di2_dqd)
        daba_dtdq = -np.einsum('il,ljk->ijk', Minv, np.einsum('ilk,lj->ijk', dm_dq, Minv))
        return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq
