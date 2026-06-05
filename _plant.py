"""Plant / cost / barrier numpy reference (mirror of the CUDA `grid_plant`).

This is the CPU oracle for the CUDA `grid_plant::` surface emitted by
`GRiDCodeGenerator/algorithms/_plant.py`. It is a pure composition of methods
that already exist on `RBDReference` (the integrator + the end-effector pose /
pose-gradient), and it adds NO new rigid-body math.

Conventions mirror the CUDA `_plant.py` verbatim:
  - state          x = [q (nq); qd (nv)]   (so nx = nq + nv)
  - input          u = control torques (nv)
  - quadratic cost = 1/2 * r^T diag(W) r;  grad = W .* r;  hess = diag(W)
  - ee-position    cost over rows 0..2 of the EE pose; grad_q = J_p^T (W .* r),
                   the qd-block of grad_x is exactly zero; GN hess = J_p^T diag(W) J_p
  - log-barrier    b = -mu*(log(x-lo)+log(hi-x)), isfinite-guarded per side so an
                   +/-inf bound contributes exactly zero. Interior margin floored
                   the same way the CUDA helpers floor it (1e-10 for the value,
                   1e-6 for grad/hess) so the two agree at/near a bound.

The Gauss-Newton hessian is the RATIFIED choice (the true 2nd-order term is
intentionally dropped — same `// TODO(plant-2nd-order)` / `// TODO(ee-2nd-order)`
decision the CUDA emit documents).
"""

import numpy as np


# Margin floors matching the CUDA `grid_plant_log_barrier*` helpers.
_BARRIER_VALUE_FLOOR = 1e-10
_BARRIER_GRAD_HESS_FLOOR = 1e-6


def _plant_log_barrier(x, lo, hi, mu):
    """Scalar log-barrier value: -mu*(log(x-lo)+log(hi-x)), isfinite-guarded.

    An +/-inf bound on either side contributes exactly zero. The interior
    margin is floored at 1e-10 to stay finite at the boundary (matching the
    CUDA `grid_plant_log_barrier`).
    """
    b = 0.0
    if np.isfinite(lo):
        d = x - lo
        if d <= _BARRIER_VALUE_FLOOR:
            d = _BARRIER_VALUE_FLOOR
        b -= np.log(d)
    if np.isfinite(hi):
        d = hi - x
        if d <= _BARRIER_VALUE_FLOOR:
            d = _BARRIER_VALUE_FLOOR
        b -= np.log(d)
    return mu * b


def _plant_log_barrier_grad(x, lo, hi, mu):
    """Scalar log-barrier gradient: -mu*(1/(x-lo) - 1/(hi-x)), isfinite-guarded.

    The signed margin is floored in magnitude at 1e-6 (sign preserved), matching
    the CUDA `grid_plant_log_barrier_grad`.
    """
    g = 0.0
    eps = _BARRIER_GRAD_HESS_FLOOR
    if np.isfinite(lo):
        d = x - lo
        a = abs(d)
        if a < eps:
            a = eps
        d = -a if d < 0 else a
        g -= 1.0 / d
    if np.isfinite(hi):
        d = hi - x
        a = abs(d)
        if a < eps:
            a = eps
        d = -a if d < 0 else a
        g += 1.0 / d
    return mu * g


def _plant_log_barrier_hess(x, lo, hi, mu):
    """Scalar log-barrier hessian: mu*(1/(x-lo)^2 + 1/(hi-x)^2), isfinite-guarded.

    Margin magnitude floored at 1e-6, matching the CUDA
    `grid_plant_log_barrier_hess`.
    """
    h = 0.0
    eps = _BARRIER_GRAD_HESS_FLOOR
    if np.isfinite(lo):
        a = abs(x - lo)
        if a < eps:
            a = eps
        h += 1.0 / (a * a)
    if np.isfinite(hi):
        a = abs(hi - x)
        if a < eps:
            a = eps
        h += 1.0 / (a * a)
    return mu * h


class _PlantMixin:
    """Plant-step / cost / barrier numpy reference (mirror of CUDA `grid_plant`)."""

    # ----- plant step (thin wrappers over the integrator) -----

    def plant_step(self, q, qd, u, dt, integrator_type="euler"):
        """x_{k+1} = integrator(q, qd, u, dt) — pass-through to `self.integrator`.

        Returns the (nq + nv,) next state, mirroring CUDA `plant_step`.
        """
        return np.asarray(
            self.integrator(q, qd, u, dt, integrator_type=integrator_type),
            dtype=np.float64,
        )

    def plant_step_gradient(self, q, qd, u, dt, integrator_type="euler"):
        """[A | B] = [d x_{k+1}/dq | d/dqd | d/du], shape (2*nv, 3*nv).

        Pass-through to `self.integrator_gradient`, mirroring CUDA
        `plant_step_gradient` (the s_dAB surface).
        """
        return np.asarray(
            self.integrator_gradient(q, qd, u, dt, integrator_type=integrator_type),
            dtype=np.float64,
        )

    def _d2qdd_tangent(self, q, qd, u):
        """Assemble the full forward-dynamics Hessian over the 3*nv tangent
        z = [dq (nv); dqd (nv); du (nv)] from the four `fdsva_so` blocks.

        Returns ``D2`` of shape ``(nv, 3*nv, 3*nv)`` with
        ``D2[i, a, b] = d^2 qdd[i] / dz[a] dz[b]``. Everything is in the nv
        tangent (correct for continuous joints where nq != nv).

        Block structure (qdd = M^{-1}(q)(u - c(q,qd)) is linear in u, and the
        M^{-1} coupling does not depend on qd, so the u-u, u-qd, qd-u blocks
        vanish):

              q (nv)        qd (nv)        u (nv)
            +-------------+-------------+-------------+
          q | daba_dqdq   | daba_dvdq^T | daba_dtdq^T |
            +-------------+-------------+-------------+
         qd | daba_dvdq   | daba_dvdv   |     0       |
            +-------------+-------------+-------------+
          u | daba_dtdq   |     0       |     0       |
            +-------------+-------------+-------------+

        where the named blocks are exactly `RBDReference.fdsva_so`'s outputs
        ``(daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq)`` with
        ``daba_dvdq[i,j,k] = d^2 qdd_i / dqd_j dq_k`` and
        ``daba_dtdq[i,j,k] = d^2 qdd_i / dtau_j dq_k`` (velocity/torque first,
        position second) -- the off-diagonal q-blocks are their (0,2,1)
        transposes.
        """
        nv = self.robot.get_num_vel()
        daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq = self.fdsva_so(q, qd, u)
        nz = 3 * nv
        qsl, vsl, usl = slice(0, nv), slice(nv, 2 * nv), slice(2 * nv, nz)
        D2 = np.zeros((nv, nz, nz), dtype=np.float64)
        D2[:, qsl, qsl] = daba_dqdq
        D2[:, vsl, vsl] = daba_dvdv
        # q <-> qd mixed (daba_dvdq is [d/dqd, d/dq]).
        D2[:, vsl, qsl] = daba_dvdq
        D2[:, qsl, vsl] = np.transpose(daba_dvdq, (0, 2, 1))
        # q <-> u mixed (daba_dtdq is [d/dtau, d/dq]); u-u, u-qd blocks are 0.
        D2[:, usl, qsl] = daba_dtdq
        D2[:, qsl, usl] = np.transpose(daba_dtdq, (0, 2, 1))
        return D2

    def plant_step_hessian(self, q, qd, u, dt, integrator_type="euler"):
        """Second-order sensitivity of the integrator step x_{k+1} = [q; v].

        Returns ``H`` of shape ``(2*nv, 3*nv, 3*nv)`` (the s_d2AB surface),
        the Hessian of each output-state tangent component w.r.t. the
        3*nv tangent z = [dq (nv); dqd (nv); du (nv)]:

            H[o, a, b] = d^2 x_{k+1}[o] / dz[a] dz[b]
                       = d/dz[a] ( plant_step_gradient[o, b] )

        so axis ``a`` is the perturbation axis and axis ``b`` the
        gradient-column axis (the convention the FD-of-gradient oracle uses).
        Output rows split as [position-tangent (nv); velocity (nv)].

        **Scope.** ``euler`` and ``semi_implicit_euler`` on both **fixed** and
        **floating** base, fully analytic. Multi-stage RK still raises (its
        2nd-order chain rule is a separate task).

        Velocity rows (both variants, single-stage): the Hessian of
        ``v_{k+1} = qd + dt*qdd(z)`` is ``dt * D2qdd``. We assemble it from
        ``_d2qdd_tangent`` and *transpose its (a,b) axes* so the perturbation
        axis sits in ``a``. For a fixed base D2qdd is (a,b)-symmetric and the
        transpose is a no-op; for a floating base the q-q block is genuinely
        asymmetric (the body-frame ``fdsva_so`` second derivative), and the
        ``(perturb, column)`` ordering is the one matching a finite difference
        of ``integrator_gradient``.

        Position rows ``q_{k+1} = integrate(q, w)`` (Lie-group retract).
          * **Euler**, ``w = dt*qd``: ``q_{k+1}`` depends on z only through the
            increment ``w`` (the ``dIntegrate`` blocks are q-independent for a
            free-flyer), and ``w`` depends only on ``qd`` (``dw/dqd = dt*I``).
            Hence every block whose perturbation axis ``a`` is in ``q`` or
            ``u`` is exactly zero, and the nonzero pieces are
              - ``a in qd, b in q``  : ``dt   * d2Integrate(q, w, 'q', 'v')``
              - ``a in qd, b in qd`` : ``dt^2 * d2Integrate(q, w, 'v', 'v')``
            with the increment-derivative index in ``a`` and the
            gradient-column index in ``b`` (i.e. ``d2Integrate[o, b, a']``).
            No symmetrization: the FD-of-pinocchio ground truth is itself
            asymmetric in (a,b) (the q-row stays zero, only the qd-row carries
            the cross term), so the tensor is filled un-symmetrized.
          * **Semi-implicit Euler**, ``w = dt*v_{k+1} = dt*(qd + dt*qdd(z))``:
            the increment depends on all of z through ``v_{k+1}``, so the
            position-row Hessian is the chain rule of ``integrate`` (its 1st +
            2nd derivatives at ``w``) composed with ``Vgrad = d v_{k+1}/dz``
            (the bottom rows of ``integrator_gradient``) and the velocity-row
            Hessian ``d^2 v_{k+1}/dz^2 = dt * D2qdd``:
              t1 (b in q only) dt   * d2Int_qv : Vgrad
              t2               dt^2 * Vgrad^T : d2Int_vv : Vgrad
              t3               dt^2 * dIntegrate_v . D2qdd

        For a fixed base ``d2Integrate`` is identically zero and
        ``dIntegrate_v = I``, so this collapses to the historical fixed-base
        result (position rows ``0`` for Euler, ``dt^2 * D2qdd`` for SI-Euler).
        See `docs/open-tasks/f1_plant_step_hessian_plan.md`.
        """
        if integrator_type not in ("euler", "semi_implicit_euler", "si_euler"):
            raise NotImplementedError(
                f"plant_step_hessian: integrator '{integrator_type}' not yet "
                "supported (multi-stage RK 2nd-order chain rule is deferred; "
                "see f1_plant_step_hessian_plan.md)."
            )
        nv = self.robot.get_num_vel()
        nz = 3 * nv
        qsl, vsl = slice(0, nv), slice(nv, 2 * nv)
        si = integrator_type in ("semi_implicit_euler", "si_euler")

        D2 = self._d2qdd_tangent(q, qd, u)  # (nv, 3nv, 3nv), axes [out, b, a]
        # Reorder to (out, a=perturb, b=column). Symmetric for a fixed base;
        # carries the asymmetric free-flyer q-q block otherwise.
        D2T = np.transpose(D2, (0, 2, 1))
        H = np.zeros((2 * nv, nz, nz), dtype=np.float64)

        # ---- velocity rows: d^2 v_{k+1} / dz^2 = dt * D2qdd ----
        H[nv:, :, :] = dt * D2T

        if not self.robot.floating_base:
            # Fixed base: linear retract -> position rows 0 (Euler) or the same
            # dt^2 * D2qdd as the velocity rows (SI-Euler, q_{k+1}=q+dt*v_{k+1}).
            if si:
                H[:nv, :, :] = (dt * dt) * D2T
            return H

        # ---- floating base position rows via the SE(3) retract derivatives ----
        if not si:
            # Euler: q_{k+1} = integrate(q, dt*qd). Only the qd perturbation
            # axis is nonzero; b in q -> d2Int_qv, b in qd -> dt * d2Int_vv.
            w = dt * np.asarray(qd, dtype=np.float64)
            d2qv = self.d2Integrate(q, w, "q", "v")  # [out, col, incr]
            d2vv = self.d2Integrate(q, w, "v", "v")  # [out, col, incr]
            for a_loc in range(nv):
                a = nv + a_loc  # perturbation axis lives in the qd block
                H[:nv, a, qsl] = dt * d2qv[:, :, a_loc]
                H[:nv, a, vsl] = (dt * dt) * d2vv[:, :, a_loc]
            return H

        # SI-Euler: q_{k+1} = integrate(q, dt*v_{k+1}), v_{k+1} = qd + dt*qdd(z).
        qdd = np.asarray(self.forward_dynamics(q, qd, u), dtype=np.float64).reshape(-1)
        v_new = np.asarray(qd, dtype=np.float64) + dt * qdd
        w = dt * v_new
        Jv = self.dIntegrate(q, w, "v")          # (nv, nv)  d q_{k+1}/d w
        d2qv = self.d2Integrate(q, w, "q", "v")  # [out, col, incr]  d dInt_q / d w
        d2vv = self.d2Integrate(q, w, "v", "v")  # [out, col, incr]  d dInt_v / d w
        J_qq, J_qv = self.forward_dynamics_gradient(q, qd, u)
        Minv = np.asarray(self.minv(q), dtype=np.float64)
        I_n = np.eye(nv)
        # Vgrad = d v_{k+1}/dz = [dt*J_qq | I + dt*J_qv | dt*Minv]  (nv, 3nv).
        Vgrad = np.hstack([dt * np.asarray(J_qq, dtype=np.float64),
                           I_n + dt * np.asarray(J_qv, dtype=np.float64),
                           dt * Minv])
        # t1 (gradient-column b restricted to q): dt * d2Int_qv contracted with
        #    d w / dz[a] = dt * Vgrad[:, a]  -> dt * sum_c d2qv[o,b,c]*Vgrad[c,a].
        H[:nv, :, qsl] += dt * np.einsum("obc,ca->oab", d2qv, Vgrad)
        # t2: dt^2 * sum_{m,c} d2Int_vv[o,m,c]*Vgrad[c,a]*Vgrad[m,b].
        H[:nv, :, :] += (dt * dt) * np.einsum("omc,ca,mb->oab", d2vv, Vgrad, Vgrad)
        # t3: dt^2 * sum_m dIntegrate_v[o,m] * D2qdd[m,a,b].
        H[:nv, :, :] += (dt * dt) * np.einsum("om,mab->oab", Jv, D2T)
        return H

    # ----- quadratic state / input cost (value, grad, GN-diag hess) -----

    @staticmethod
    def _quadratic_cost(var, des, W):
        var = np.asarray(var, dtype=np.float64).reshape(-1)
        des = np.asarray(des, dtype=np.float64).reshape(-1)
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        r = var - des
        value = 0.5 * float(np.sum(W * r * r))
        grad = W * r
        hess = np.diag(W)
        return value, grad, hess

    def quadratic_state_cost(self, x, x_des, Q):
        """1/2 * sum_i Q_i (x_i - x_des_i)^2 over the full state x = [q; qd].

        Returns (value, grad (nx,), hess = diag(Q) (nx, nx)).
        """
        return self._quadratic_cost(x, x_des, Q)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 * sum_i R_i (u_i - u_des_i)^2 over the input u (size nv).

        Returns (value, grad (nu,), hess = diag(R) (nu, nu)).
        """
        return self._quadratic_cost(u, u_des, R)

    # ----- end-effector position cost (value, grad_x, GN hess_x) -----

    def _ee_target_name(self, ee):
        """The CUDA-test EE selection: leaf = robot.get_leaf_nodes()[ee];
        target = robot.get_joint_by_id(leaf).get_name()."""
        leaf = self.robot.get_leaf_nodes()[ee]
        return self.robot.get_joint_by_id(leaf).get_name()

    def ee_pos_cost(self, q, p_des, W, ee=0):
        """End-effector position cost over the 3 position axes (rows 0..2).

        r = p(q) - p_des (3-vector); value = 1/2 sum_r W[r] r[r]^2.
        grad_q = J_p^T (W .* r) (size nv); grad_x = [grad_q; 0] (qd-block zero).
        GN hess_x = nx x nx with the top-left nv x nv q-block = J_p^T diag(W) J_p,
        everything else zero (the W*r-weighted EE-Hessian term is dropped, GN).

        Returns (value, grad_x (nx,), hess_x (nx, nx)).
        """
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nx = nq + nv
        target = self._ee_target_name(ee)
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        p_des = np.asarray(p_des, dtype=np.float64).reshape(-1)

        pose = np.asarray(
            self.end_effector_pose(q, ee_joint_names=target)[0], dtype=np.float64
        ).reshape(-1)
        p = pose[:3]
        # J_p = rows 0..2 of the 6 x nv d/dv tangent EE Jacobian.
        Jfull = np.asarray(
            self.end_effector_pose_gradient(q, ee_joint_names=target)[0],
            dtype=np.float64,
        )
        Jp = Jfull[:3, :]  # (3, nv)

        r = p - p_des
        value = 0.5 * float(np.sum(W * r * r))

        grad_q = Jp.T @ (W * r)  # (nv,)
        grad_x = np.zeros(nx, dtype=np.float64)
        grad_x[:nv] = grad_q

        hess_x = np.zeros((nx, nx), dtype=np.float64)
        hess_x[:nv, :nv] = Jp.T @ (W[:, None] * Jp)  # J_p^T diag(W) J_p
        return value, grad_x, hess_x

    # ----- CoM-tracking cost (value, grad_x, GN hess_x) -----

    def com_cost(self, q, p_des, W):
        """Center-of-mass tracking cost over the 3 CoM axes.

        Mirrors the CUDA `grid_plant::com_cost` verbatim (built on the CoM
        position `self.com` and the CoM Jacobian `self.jacobian_com`, the two
        sub-outputs of `grid::com_device`):
          r       = p_com(q) - p_des                       (3-vector)
          value   = 1/2 sum_r W[r] r[r]^2
          grad_x  = [J_com^T (W .* r) ; 0]   (qd-block exactly zero)
          GN hess = J_com^T diag(W) J_com in the top-left NUM_VEL x NUM_VEL
                    q-block of the NX x NX x-hessian (everything else zero).

        The Gauss-Newton hessian drops the W*r-weighted CoM-Hessian term, the
        same ratified choice the CUDA emit documents (matches `ee_pos_cost`).

        Returns (value, grad_x (nx,), hess_x (nx, nx)).
        """
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nx = nq + nv
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        p_des = np.asarray(p_des, dtype=np.float64).reshape(-1)

        p = np.asarray(self.com(q), dtype=np.float64).reshape(-1)
        Jcom = np.asarray(self.jacobian_com(q), dtype=np.float64)  # (3, nv)

        r = p - p_des
        value = 0.5 * float(np.sum(W * r * r))

        grad_x = np.zeros(nx, dtype=np.float64)
        grad_x[:nv] = Jcom.T @ (W * r)  # (nv,)

        hess_x = np.zeros((nx, nx), dtype=np.float64)
        hess_x[:nv, :nv] = Jcom.T @ (W[:, None] * Jcom)  # J_com^T diag(W) J_com
        return value, grad_x, hess_x

    # ----- centroidal-momentum-tracking cost (value, grad_x, GN hess_x) -----

    def momentum_cost(self, q, qd, h_des, W):
        """Centroidal-momentum tracking cost over the 6 momentum components.

        Mirrors the CUDA `grid_plant::momentum_cost` verbatim (built on the CMM
        `A` and momentum `h = A qd`, the two sub-outputs of `grid::ccrba_device`
        via `self.ccrba`):
          r       = h(q,qd) - h_des                        (6-vector)
          value   = 1/2 sum_r W[r] r[r]^2
          grad_x  = [0 ; A^T (W .* r)]   (q-block dropped, GN on A; qd-block only)
          GN hess = A^T diag(W) A in the bottom-right NUM_VEL x NUM_VEL qd-block
                    of the NX x NX x-hessian (everything else zero).

        h depends on qd linearly (J_h = A), so the qd-block gradient/hessian are
        exact; the q-dependence of A is dropped Gauss-Newton style, matching the
        ratified `ee_pos_cost`/`com_cost` choice the CUDA emit documents.

        Returns (value, grad_x (nx,), hess_x (nx, nx)).
        """
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nx = nq + nv
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        h_des = np.asarray(h_des, dtype=np.float64).reshape(-1)

        A, h = self.ccrba(q, qd)  # A (6, nv), h (6,)
        A = np.asarray(A, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64).reshape(-1)

        r = h - h_des
        value = 0.5 * float(np.sum(W * r * r))

        grad_x = np.zeros(nx, dtype=np.float64)
        grad_x[nq:] = A.T @ (W * r)  # (nv,) in the qd-block

        hess_x = np.zeros((nx, nx), dtype=np.float64)
        hess_x[nq:, nq:] = A.T @ (W[:, None] * A)  # A^T diag(W) A
        return value, grad_x, hess_x

    # ----- log barriers (joint position / velocity / torque) -----

    @staticmethod
    def _barrier(vals, lower, upper, mu):
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper, dtype=np.float64).reshape(-1)
        n = vals.shape[0]
        value = 0.0
        grad = np.zeros(n, dtype=np.float64)
        hess_diag = np.zeros(n, dtype=np.float64)
        for i in range(n):
            value += _plant_log_barrier(vals[i], lower[i], upper[i], mu)
            grad[i] = _plant_log_barrier_grad(vals[i], lower[i], upper[i], mu)
            hess_diag[i] = _plant_log_barrier_hess(vals[i], lower[i], upper[i], mu)
        return value, grad, hess_diag

    def joint_position_barrier(self, vals, lower, upper, mu):
        """Log-barrier over joint positions. Returns (value, grad, hess_diag)."""
        return self._barrier(vals, lower, upper, mu)

    def joint_velocity_barrier(self, vals, lower, upper, mu):
        """Log-barrier over joint velocities. Returns (value, grad, hess_diag)."""
        return self._barrier(vals, lower, upper, mu)

    def joint_torque_barrier(self, vals, lower, upper, mu):
        """Log-barrier over joint torques. Returns (value, grad, hess_diag)."""
        return self._barrier(vals, lower, upper, mu)
