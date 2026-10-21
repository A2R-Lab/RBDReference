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
        self._integrator_butcher(integrator_type)
        if integrator_type not in ("euler", "semi_implicit_euler"):
            raise NotImplementedError(
                f"plant_step_hessian: integrator '{integrator_type}' not yet "
                "supported; only euler and semi_implicit_euler have step Hessians."
            )
        nv = self.robot.get_num_vel()
        nz = 3 * nv
        qsl, vsl = slice(0, nv), slice(nv, 2 * nv)
        si = integrator_type == "semi_implicit_euler"

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

    def quadratic_state_cost_tangent(self, x, x_des, Q, gauss_newton=True):
        """Tangent-space (log-map) quadratic state cost — the oracle twin of
        the grid_plant floating tracking preset (GATO ASK 3).

        x = [q (nq); qd (nv)]. The error is TANGENT-width 2*nv:
            e = [difference(q_des, q) (nv); qd - qd_des (nv)]
        and Q is a DIAGONAL weight vector of size 2*nv (NOT nq+nv).

        Returns (value, grad (2*nv,), hess (2*nv, 2*nv)): the derivatives of
        delta -> cost(integrate(q, delta_q), qd + delta_v) at delta = 0 —
        the tangent-chart derivatives a manifold Newton/SQP consumes. The
        q-block uses the EXACT J_diff = dDifference(q_des, q, 'to'). The
        hessian is Gauss-Newton (J^T diag(Q) J) by default; gauss_newton=False
        adds the exact curvature from d2Integrate (full Newton): with
        e(delta) = difference(q_des, integrate(q, delta)) the chain is
        de/ddelta = M(e)^-1 N(delta), M = dIntegrate(q_des, e, 'v'),
        N = dIntegrate(q, delta, 'v') — differentiating BOTH factors at
        delta = 0 gives -J (dM) J (through e) plus J-composed dN (the
        d2Integrate-at-zero right-Jacobian slope; N(0) = I but dN(0) != 0).
        Fixed base reduces exactly to quadratic_state_cost (J = I,
        e = x - x_des).
        """
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        x_des = np.asarray(x_des, dtype=np.float64).reshape(-1)
        Q = np.asarray(Q, dtype=np.float64).reshape(-1)
        q, qd = x[:nq], x[nq:nq + nv]
        q_des, qd_des = x_des[:nq], x_des[nq:nq + nv]
        Qq, Qv = Q[:nv], Q[nv:]
        eq = self.difference(q_des, q)
        ev = qd - qd_des
        value = 0.5 * float(np.sum(Qq * eq * eq) + np.sum(Qv * ev * ev))
        Jq = self.dDifference(q_des, q, "to")
        grad_q = Jq.T @ (Qq * eq)
        grad = np.concatenate([grad_q, Qv * ev])
        hess = np.zeros((2 * nv, 2 * nv))
        hess[:nv, :nv] = Jq.T @ (Qq[:, None] * Jq)
        hess[nv:, nv:] = np.diag(Qv)
        if not gauss_newton:
            T_des = self.d2Integrate(q_des, eq, "v", "v")
            T0 = self.d2Integrate(q, np.zeros(nv), "v", "v")
            hess[:nv, :nv] += -np.einsum("b,bcm,ci,mk->ik", grad_q, T_des, Jq, Jq)
            hess[:nv, :nv] += np.einsum("b,bik->ik", grad_q, T0)
        return value, grad, hess

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

    def gravity_torque(self, q, GRAVITY=-9.81):
        """g(q) = RNEA(q, 0, 0), the generalized gravity (size nv); the oracle of
        grid_plant::gravity_torque."""
        nv = self.robot.get_num_vel()
        return np.asarray(self.inverse_dynamics(q, np.zeros(nv), np.zeros(nv), GRAVITY=GRAVITY)[0], dtype=np.float64).reshape(-1)

    # ----- cost menu: activations, activated terms, linear feature, named frames -----

    ACTIVATION_QUADRATIC = 0
    ACTIVATION_PSEUDO_HUBER = 1

    ACTIVATION_RELAXED_LOG_BARRIER = 2
    ACTIVATION_QUADRATIC_BARRIER = 3

    @staticmethod
    def _relaxed_log_barrier_1d(z, mu, delta):
        """Grandia et al. relaxed log barrier on z >= 0: -mu log z above delta, the
        quadratic extension below it (value and slope continuous at delta).
        Returns (value, d/dz, d2/dz2) arrays. z may be +inf (no bound): zero."""
        z = np.asarray(z, dtype=np.float64)
        val = np.zeros_like(z); d1 = np.zeros_like(z); d2 = np.zeros_like(z)
        finite = np.isfinite(z)
        above = finite & (z > delta)
        below = finite & ~(z > delta)
        val[above] = -mu * np.log(z[above])
        d1[above] = -mu / z[above]
        d2[above] = mu / (z[above] ** 2)
        t = (z[below] - 2.0 * delta) / delta
        val[below] = mu * (0.5 * t * t - 0.5 - np.log(delta))
        d1[below] = mu * t / delta
        d2[below] = mu / (delta * delta) * np.ones_like(t)
        return val, d1, d2

    @staticmethod
    def _activation(kind, W, r, eps, v=None, lo=None, hi=None):
        """Per-entry activation a(r; w, eps) with its first and second derivatives.

        kind 0 (quadratic): a = 1/2 w r^2, a' = w r, a'' = w.
        kind 1 (pseudo-Huber): a = w (sqrt(r^2 + eps^2) - eps), a' = w r / s,
        a'' = w eps^2 / s^3 with s = sqrt(r^2 + eps^2); eps must be positive.
        kind 2 (relaxed log barrier): on the feature value v in [lo, hi], with
        w = mu and eps = delta: B(v - lo) + B(hi - v) where B is -mu log z above
        delta and its quadratic extension below; an infinite bound is zero.
        kind 3 (quadratic barrier): 1/2 w (max(0, lo - v)^2 + max(0, v - hi)^2).
        The barrier kinds read v, lo, hi; r is the residual the other kinds use.
        Returns (value (M,), slope (M,), curvature (M,)), the derivatives with
        respect to r (equal to those with respect to v: v = r + target).
        """
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        if kind == _PlantMixin.ACTIVATION_QUADRATIC:
            return 0.5 * W * r * r, W * r, W.copy()
        if kind == _PlantMixin.ACTIVATION_PSEUDO_HUBER:
            eps = float(eps)
            if not eps > 0.0:
                raise ValueError("the pseudo-Huber activation needs eps > 0")
            s = np.sqrt(r * r + eps * eps)
            return W * (s - eps), W * r / s, W * eps * eps / (s * s * s)
        if kind in (_PlantMixin.ACTIVATION_RELAXED_LOG_BARRIER, _PlantMixin.ACTIVATION_QUADRATIC_BARRIER):
            if v is None or lo is None or hi is None:
                raise ValueError("barrier activations need the feature value v and the bounds lo, hi")
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            lo = np.asarray(lo, dtype=np.float64).reshape(-1)
            hi = np.asarray(hi, dtype=np.float64).reshape(-1)
            if kind == _PlantMixin.ACTIVATION_QUADRATIC_BARRIER:
                below = np.where(np.isfinite(lo), np.maximum(0.0, lo - v), 0.0)
                above = np.where(np.isfinite(hi), np.maximum(0.0, v - hi), 0.0)
                value = 0.5 * W * (below * below + above * above)
                slope = W * (above - below)
                curv = W * ((below > 0) | (above > 0)).astype(np.float64)
                return value, slope, curv
            delta = float(eps)
            if not delta > 0.0:
                raise ValueError("the relaxed log barrier needs delta = eps > 0")
            value = np.zeros_like(r); slope = np.zeros_like(r); curv = np.zeros_like(r)
            for k in range(r.size):
                mu = W[k]
                if np.isfinite(lo[k]):
                    val, d1, d2 = _PlantMixin._relaxed_log_barrier_1d(np.array([v[k] - lo[k]]), mu, delta)
                    value[k] += val[0]; slope[k] += d1[0]; curv[k] += d2[0]
                if np.isfinite(hi[k]):
                    val, d1, d2 = _PlantMixin._relaxed_log_barrier_1d(np.array([hi[k] - v[k]]), mu, delta)
                    value[k] += val[0]; slope[k] -= d1[0]; curv[k] += d2[0]
            return value, slope, curv
        raise ValueError(f"unknown activation kind {kind!r}")

    @staticmethod
    def _skew3(v):
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @staticmethod
    def _so3_log(R):
        """phi = log(R) on the canonical branch |phi| <= pi (3-vector)."""
        R = np.asarray(R, dtype=np.float64)
        c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(c)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if theta < 1e-6:
            return 0.5 * w
        if theta < np.pi - 1e-4:
            return theta / (2.0 * np.sin(theta)) * w
        # near pi: axis from the symmetric part (R + I)/2 = a a^T, sign from w
        S = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(S)))
        a = S[:, k] / np.sqrt(max(S[k, k], 1e-18))
        a = a / np.linalg.norm(a)
        if np.dot(a, w) < 0.0:
            a = -a
        return theta * a

    @staticmethod
    def _so3_right_jacobian_inv(phi):
        """Jr(phi)^-1 = I + 1/2 [phi]x + (1/theta^2 - (1 + cos theta)/(2 theta sin theta)) [phi]x^2."""
        phi = np.asarray(phi, dtype=np.float64)
        theta = np.linalg.norm(phi)
        S = _PlantMixin._skew3(phi)
        if theta < 1e-5:
            coef = 1.0 / 12.0
        else:
            coef = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        return np.eye(3) + 0.5 * S + coef * (S @ S)

    def _frame_placement(self, q, frame_name):
        """(R_f, p_f, target_id) of a named articulated or fixed joint frame, the
        fixed joint's offset included (the frame_jacobian resolution)."""
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_off = self._resolve_frame_joint(frame_name)
        if target_id == -1:
            X = X_off
        else:
            X = Xw[target_id] @ X_off
        return X[:3, :3], X[:3, 3], target_id

    def frame_position_cost(self, q, frame_name, p_des, W, activation=0, eps=0.0, lo=None, hi=None):
        """Position residual r = p_f(q) - p_des of a named frame under an activation.
        Returns (value, grad_x (nx,) = [J_p^T a'(r); 0], GN hess_x (nx, nx))."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        R_f, p_f, _ = self._frame_placement(q, frame_name)
        Jp = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL_WORLD_ALIGNED"))[:3, :]
        value, slope, curv = self._activation(activation, W, p_f - np.asarray(p_des, dtype=np.float64).reshape(-1), eps, p_f, lo, hi)
        grad_x = np.zeros(nx); grad_x[:nv] = Jp.T @ slope
        hess_x = np.zeros((nx, nx)); hess_x[:nv, :nv] = Jp.T @ (curv[:, None] * Jp)
        return float(np.sum(value)), grad_x, hess_x

    def frame_orientation_cost(self, q, frame_name, R_des, W, activation=0, eps=0.0):
        """Log-map orientation residual e = log(R_des^T R_f(q)) (3-vector) under an
        activation. With the frame's body-axes angular Jacobian J_w (LOCAL rows 3..5)
        the residual Jacobian is J_e = Jr(e)^-1 J_w. Returns (value, grad_x, GN hess_x)."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        R_f, _, _ = self._frame_placement(q, frame_name)
        R_des = np.asarray(R_des, dtype=np.float64).reshape(3, 3, order="F")
        e = self._so3_log(R_des.T @ R_f)
        Jw = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL"))[3:, :]
        Je = self._so3_right_jacobian_inv(e) @ Jw
        value, slope, curv = self._activation(activation, W, e, eps)
        grad_x = np.zeros(nx); grad_x[:nv] = Je.T @ slope
        hess_x = np.zeros((nx, nx)); hess_x[:nv, :nv] = Je.T @ (curv[:, None] * Je)
        return float(np.sum(value)), grad_x, hess_x

    def _frame_velocity_q_derivative(self, q, qd, frame_name):
        """d(J(q) qd)/dq for the LOCAL_WORLD_ALIGNED frame Jacobian (6 x nv), from
        the same world-axis column construction frame_jacobian uses. Fixed base."""
        if self.robot.floating_base:
            raise NotImplementedError("frame velocity residual: fixed base only in this release")
        nv = self.robot.get_num_vel()
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_off = self._resolve_frame_joint(frame_name)
        p_f = (Xw[target_id] @ X_off)[:3, 3]
        chain = sorted(self.robot.get_ancestors_by_id(target_id)) + [target_id]

        def column_info(j):
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            R_j = Xw[j][:3, :3]; p_j = Xw[j][:3, 3]
            inds = self.robot.get_joint_index_v(j)
            inds = list(inds) if isinstance(inds, (list, tuple, np.ndarray)) else [inds]
            joint = self.robot.get_joint_by_id(j)
            scale = joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0
            out = []
            for c in range(S.shape[1]):
                vi = inds[c] if c < len(inds) else inds[-1]
                rot = np.linalg.norm(S[:3, c]) > 0.5
                axis = R_j @ (S[:3, c] if rot else S[3:6, c])
                out.append((vi, rot, axis, p_j, scale))
            return out

        cols = {j: column_info(j) for j in chain}
        dJqd = np.zeros((6, nv))
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        # Column j is a_j x (p_f - p_j) (revolute) or l_j (prismatic) over its axis
        # a_j / l_j and origin p_j. A joint m moves, rigidly, everything downstream
        # of it: for m an ancestor-or-self of j that is the axis, the origin and the
        # frame point; for m downstream of j only the frame point p_f moves.
        for j in chain:
            for (vi, rot, a_j, p_j, s_j) in cols[j]:
                w = s_j * qd[vi]
                for m in chain:
                    for (vm, rot_m, a_m, p_m, s_m) in cols[m]:
                        if m <= j:
                            if rot and rot_m:
                                d_axis = np.cross(a_m, a_j)
                                d_lin = np.cross(d_axis, p_f - p_j) + np.cross(a_j, np.cross(a_m, p_f - p_j))
                                dJqd[:3, vm] += s_m * w * d_lin
                                dJqd[3:, vm] += s_m * w * d_axis
                            elif (not rot) and rot_m:
                                dJqd[:3, vm] += s_m * w * np.cross(a_m, a_j)
                            # a prismatic m shifts the frame and the joint origins together: no change
                        elif rot:
                            # m downstream of j: only p_f moves, by a_m x (p_f - p_m) or l_m
                            d_pf = np.cross(a_m, p_f - p_m) if rot_m else a_m
                            dJqd[:3, vm] += s_m * w * np.cross(a_j, d_pf)
        return dJqd

    def frame_velocity_cost(self, q, qd, frame_name, v_des, W, activation=0, eps=0.0, lo=None, hi=None):
        """Frame velocity residual r = J(q) qd - v_des over the LOCAL_WORLD_ALIGNED
        twist [linear; angular] (6) under an activation. The residual Jacobian over
        x = [q; qd] is [d(J qd)/dq | J]. Returns (value, grad_x (nx,), GN hess_x)."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        J = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL_WORLD_ALIGNED"))
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        r = J @ qd - np.asarray(v_des, dtype=np.float64).reshape(-1)
        Jx = np.hstack([self._frame_velocity_q_derivative(q, qd, frame_name), J])
        value, slope, curv = self._activation(activation, W, r, eps, J @ qd, lo, hi)
        return float(np.sum(value)), Jx.T @ slope, Jx.T @ (curv[:, None] * Jx)

    @classmethod
    def _activated_cost(cls, var, des, W, activation, eps, lo=None, hi=None):
        var = np.asarray(var, dtype=np.float64).reshape(-1)
        des = np.asarray(des, dtype=np.float64).reshape(-1)
        value, slope, curv = cls._activation(activation, W, var - des, eps, var, lo, hi)
        return float(np.sum(value)), slope, np.diag(curv)

    def state_cost(self, x, x_des, Q, activation=0, eps=0.0, lo=None, hi=None):
        """sum_i a(x_i - x_des_i; Q_i, eps) over x = [q; qd]. The quadratic
        activation is quadratic_state_cost; lo / hi are the barrier bounds on x.
        Returns (value, grad (nx,), GN hess (nx, nx))."""
        return self._activated_cost(x, x_des, Q, activation, eps, lo, hi)

    def input_cost(self, u, u_des, R, activation=0, eps=0.0, lo=None, hi=None):
        """sum_i a(u_i - u_des_i; R_i, eps). Returns (value, grad (nu,), GN hess (nu, nu))."""
        return self._activated_cost(u, u_des, R, activation, eps, lo, hi)

    def linear_feature_cost(self, v, A, b, target, W, activation=0, eps=0.0, lo=None, hi=None):
        """sum_m a(r_m; W_m, eps) with r = A v + b - target, A of shape (M, N); lo / hi
        are barrier bounds on the feature value A v + b.

        Returns (value, grad (N,) = A^T a'(r), GN hess (N, N) = A^T diag(a''(r)) A).
        """
        A = np.asarray(A, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        f = A @ v + np.asarray(b, dtype=np.float64).reshape(-1)
        r = f - np.asarray(target, dtype=np.float64).reshape(-1)
        value, slope, curv = self._activation(activation, W, r, eps, f, lo, hi)
        return float(np.sum(value)), A.T @ slope, A.T @ (curv[:, None] * A)

    ACTIVATION_RELAXED_LOG_BARRIER = 2
    ACTIVATION_QUADRATIC_BARRIER = 3

    @staticmethod
    def _relaxed_log_barrier_1d(z, mu, delta):
        """Grandia et al. relaxed log barrier on z >= 0: -mu log z above delta, the
        quadratic extension below it (value and slope continuous at delta).
        Returns (value, d/dz, d2/dz2) arrays. z may be +inf (no bound): zero."""
        z = np.asarray(z, dtype=np.float64)
        val = np.zeros_like(z); d1 = np.zeros_like(z); d2 = np.zeros_like(z)
        finite = np.isfinite(z)
        above = finite & (z > delta)
        below = finite & ~(z > delta)
        val[above] = -mu * np.log(z[above])
        d1[above] = -mu / z[above]
        d2[above] = mu / (z[above] ** 2)
        t = (z[below] - 2.0 * delta) / delta
        val[below] = mu * (0.5 * t * t - 0.5 - np.log(delta))
        d1[below] = mu * t / delta
        d2[below] = mu / (delta * delta) * np.ones_like(t)
        return val, d1, d2

    @staticmethod
    def _activation(kind, W, r, eps, v=None, lo=None, hi=None):
        """Per-entry activation a(r; w, eps) with its first and second derivatives.

        kind 0 (quadratic): a = 1/2 w r^2, a' = w r, a'' = w.
        kind 1 (pseudo-Huber): a = w (sqrt(r^2 + eps^2) - eps), a' = w r / s,
        a'' = w eps^2 / s^3 with s = sqrt(r^2 + eps^2); eps must be positive.
        kind 2 (relaxed log barrier): on the feature value v in [lo, hi], with
        w = mu and eps = delta: B(v - lo) + B(hi - v) where B is -mu log z above
        delta and its quadratic extension below; an infinite bound is zero.
        kind 3 (quadratic barrier): 1/2 w (max(0, lo - v)^2 + max(0, v - hi)^2).
        The barrier kinds read v, lo, hi; r is the residual the other kinds use.
        Returns (value (M,), slope (M,), curvature (M,)), the derivatives with
        respect to r (equal to those with respect to v: v = r + target).
        """
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        if kind == _PlantMixin.ACTIVATION_QUADRATIC:
            return 0.5 * W * r * r, W * r, W.copy()
        if kind == _PlantMixin.ACTIVATION_PSEUDO_HUBER:
            eps = float(eps)
            if not eps > 0.0:
                raise ValueError("the pseudo-Huber activation needs eps > 0")
            s = np.sqrt(r * r + eps * eps)
            return W * (s - eps), W * r / s, W * eps * eps / (s * s * s)
        if kind in (_PlantMixin.ACTIVATION_RELAXED_LOG_BARRIER, _PlantMixin.ACTIVATION_QUADRATIC_BARRIER):
            if v is None or lo is None or hi is None:
                raise ValueError("barrier activations need the feature value v and the bounds lo, hi")
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            lo = np.asarray(lo, dtype=np.float64).reshape(-1)
            hi = np.asarray(hi, dtype=np.float64).reshape(-1)
            if kind == _PlantMixin.ACTIVATION_QUADRATIC_BARRIER:
                below = np.where(np.isfinite(lo), np.maximum(0.0, lo - v), 0.0)
                above = np.where(np.isfinite(hi), np.maximum(0.0, v - hi), 0.0)
                value = 0.5 * W * (below * below + above * above)
                slope = W * (above - below)
                curv = W * ((below > 0) | (above > 0)).astype(np.float64)
                return value, slope, curv
            delta = float(eps)
            if not delta > 0.0:
                raise ValueError("the relaxed log barrier needs delta = eps > 0")
            value = np.zeros_like(r); slope = np.zeros_like(r); curv = np.zeros_like(r)
            for k in range(r.size):
                mu = W[k]
                if np.isfinite(lo[k]):
                    val, d1, d2 = _PlantMixin._relaxed_log_barrier_1d(np.array([v[k] - lo[k]]), mu, delta)
                    value[k] += val[0]; slope[k] += d1[0]; curv[k] += d2[0]
                if np.isfinite(hi[k]):
                    val, d1, d2 = _PlantMixin._relaxed_log_barrier_1d(np.array([hi[k] - v[k]]), mu, delta)
                    value[k] += val[0]; slope[k] -= d1[0]; curv[k] += d2[0]
            return value, slope, curv
        raise ValueError(f"unknown activation kind {kind!r}")

    @staticmethod
    def _skew3(v):
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @staticmethod
    def _so3_log(R):
        """phi = log(R) on the canonical branch |phi| <= pi (3-vector)."""
        R = np.asarray(R, dtype=np.float64)
        c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(c)
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if theta < 1e-6:
            return 0.5 * w
        if theta < np.pi - 1e-4:
            return theta / (2.0 * np.sin(theta)) * w
        # near pi: axis from the symmetric part (R + I)/2 = a a^T, sign from w
        S = 0.5 * (R + np.eye(3))
        k = int(np.argmax(np.diag(S)))
        a = S[:, k] / np.sqrt(max(S[k, k], 1e-18))
        a = a / np.linalg.norm(a)
        if np.dot(a, w) < 0.0:
            a = -a
        return theta * a

    @staticmethod
    def _so3_right_jacobian_inv(phi):
        """Jr(phi)^-1 = I + 1/2 [phi]x + (1/theta^2 - (1 + cos theta)/(2 theta sin theta)) [phi]x^2."""
        phi = np.asarray(phi, dtype=np.float64)
        theta = np.linalg.norm(phi)
        S = _PlantMixin._skew3(phi)
        if theta < 1e-5:
            coef = 1.0 / 12.0
        else:
            coef = 1.0 / (theta * theta) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        return np.eye(3) + 0.5 * S + coef * (S @ S)

    def _frame_placement(self, q, frame_name):
        """(R_f, p_f, target_id) of a named articulated or fixed joint frame, the
        fixed joint's offset included (the frame_jacobian resolution)."""
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_off = self._resolve_frame_joint(frame_name)
        if target_id == -1:
            X = X_off
        else:
            X = Xw[target_id] @ X_off
        return X[:3, :3], X[:3, 3], target_id

    def frame_position_cost(self, q, frame_name, p_des, W, activation=0, eps=0.0, lo=None, hi=None):
        """Position residual r = p_f(q) - p_des of a named frame under an activation.
        Returns (value, grad_x (nx,) = [J_p^T a'(r); 0], GN hess_x (nx, nx))."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        R_f, p_f, _ = self._frame_placement(q, frame_name)
        Jp = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL_WORLD_ALIGNED"))[:3, :]
        value, slope, curv = self._activation(activation, W, p_f - np.asarray(p_des, dtype=np.float64).reshape(-1), eps, p_f, lo, hi)
        grad_x = np.zeros(nx); grad_x[:nv] = Jp.T @ slope
        hess_x = np.zeros((nx, nx)); hess_x[:nv, :nv] = Jp.T @ (curv[:, None] * Jp)
        return float(np.sum(value)), grad_x, hess_x

    def frame_orientation_cost(self, q, frame_name, R_des, W, activation=0, eps=0.0):
        """Log-map orientation residual e = log(R_des^T R_f(q)) (3-vector) under an
        activation. With the frame's body-axes angular Jacobian J_w (LOCAL rows 3..5)
        the residual Jacobian is J_e = Jr(e)^-1 J_w. Returns (value, grad_x, GN hess_x)."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        R_f, _, _ = self._frame_placement(q, frame_name)
        R_des = np.asarray(R_des, dtype=np.float64).reshape(3, 3, order="F")
        e = self._so3_log(R_des.T @ R_f)
        Jw = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL"))[3:, :]
        Je = self._so3_right_jacobian_inv(e) @ Jw
        value, slope, curv = self._activation(activation, W, e, eps)
        grad_x = np.zeros(nx); grad_x[:nv] = Je.T @ slope
        hess_x = np.zeros((nx, nx)); hess_x[:nv, :nv] = Je.T @ (curv[:, None] * Je)
        return float(np.sum(value)), grad_x, hess_x

    def _frame_velocity_q_derivative(self, q, qd, frame_name):
        """d(J(q) qd)/dq for the LOCAL_WORLD_ALIGNED frame Jacobian (6 x nv), from
        the same world-axis column construction frame_jacobian uses. Fixed base."""
        if self.robot.floating_base:
            raise NotImplementedError("frame velocity residual: fixed base only in this release")
        nv = self.robot.get_num_vel()
        Xw, _ = self._frame_world_placement_and_chain(q)
        target_id, X_off = self._resolve_frame_joint(frame_name)
        p_f = (Xw[target_id] @ X_off)[:3, 3]
        chain = sorted(self.robot.get_ancestors_by_id(target_id)) + [target_id]

        def column_info(j):
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            R_j = Xw[j][:3, :3]; p_j = Xw[j][:3, 3]
            inds = self.robot.get_joint_index_v(j)
            inds = list(inds) if isinstance(inds, (list, tuple, np.ndarray)) else [inds]
            joint = self.robot.get_joint_by_id(j)
            scale = joint.get_mimic_multiplier() if getattr(joint, "is_mimic", False) else 1.0
            out = []
            for c in range(S.shape[1]):
                vi = inds[c] if c < len(inds) else inds[-1]
                rot = np.linalg.norm(S[:3, c]) > 0.5
                axis = R_j @ (S[:3, c] if rot else S[3:6, c])
                out.append((vi, rot, axis, p_j, scale))
            return out

        cols = {j: column_info(j) for j in chain}
        dJqd = np.zeros((6, nv))
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        # Column j is a_j x (p_f - p_j) (revolute) or l_j (prismatic) over its axis
        # a_j / l_j and origin p_j. A joint m moves, rigidly, everything downstream
        # of it: for m an ancestor-or-self of j that is the axis, the origin and the
        # frame point; for m downstream of j only the frame point p_f moves.
        for j in chain:
            for (vi, rot, a_j, p_j, s_j) in cols[j]:
                w = s_j * qd[vi]
                for m in chain:
                    for (vm, rot_m, a_m, p_m, s_m) in cols[m]:
                        if m <= j:
                            if rot and rot_m:
                                d_axis = np.cross(a_m, a_j)
                                d_lin = np.cross(d_axis, p_f - p_j) + np.cross(a_j, np.cross(a_m, p_f - p_j))
                                dJqd[:3, vm] += s_m * w * d_lin
                                dJqd[3:, vm] += s_m * w * d_axis
                            elif (not rot) and rot_m:
                                dJqd[:3, vm] += s_m * w * np.cross(a_m, a_j)
                            # a prismatic m shifts the frame and the joint origins together: no change
                        elif rot:
                            # m downstream of j: only p_f moves, by a_m x (p_f - p_m) or l_m
                            d_pf = np.cross(a_m, p_f - p_m) if rot_m else a_m
                            dJqd[:3, vm] += s_m * w * np.cross(a_j, d_pf)
        return dJqd

    def frame_velocity_cost(self, q, qd, frame_name, v_des, W, activation=0, eps=0.0, lo=None, hi=None):
        """Frame velocity residual r = J(q) qd - v_des over the LOCAL_WORLD_ALIGNED
        twist [linear; angular] (6) under an activation. The residual Jacobian over
        x = [q; qd] is [d(J qd)/dq | J]. Returns (value, grad_x (nx,), GN hess_x)."""
        nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
        J = np.asarray(self.frame_jacobian(q, frame_name, "LOCAL_WORLD_ALIGNED"))
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        r = J @ qd - np.asarray(v_des, dtype=np.float64).reshape(-1)
        Jx = np.hstack([self._frame_velocity_q_derivative(q, qd, frame_name), J])
        value, slope, curv = self._activation(activation, W, r, eps, J @ qd, lo, hi)
        return float(np.sum(value)), Jx.T @ slope, Jx.T @ (curv[:, None] * Jx)

    @classmethod
    def _activated_cost(cls, var, des, W, activation, eps, lo=None, hi=None):
        var = np.asarray(var, dtype=np.float64).reshape(-1)
        des = np.asarray(des, dtype=np.float64).reshape(-1)
        value, slope, curv = cls._activation(activation, W, var - des, eps, var, lo, hi)
        return float(np.sum(value)), slope, np.diag(curv)

    def state_cost(self, x, x_des, Q, activation=0, eps=0.0, lo=None, hi=None):
        """sum_i a(x_i - x_des_i; Q_i, eps) over x = [q; qd]. The quadratic
        activation is quadratic_state_cost; lo / hi are the barrier bounds on x.
        Returns (value, grad (nx,), GN hess (nx, nx))."""
        return self._activated_cost(x, x_des, Q, activation, eps, lo, hi)

    def input_cost(self, u, u_des, R, activation=0, eps=0.0, lo=None, hi=None):
        """sum_i a(u_i - u_des_i; R_i, eps). Returns (value, grad (nu,), GN hess (nu, nu))."""
        return self._activated_cost(u, u_des, R, activation, eps, lo, hi)

    def linear_feature_cost(self, v, A, b, target, W, activation=0, eps=0.0, lo=None, hi=None):
        """sum_m a(r_m; W_m, eps) with r = A v + b - target, A of shape (M, N); lo / hi
        are barrier bounds on the feature value A v + b.

        Returns (value, grad (N,) = A^T a'(r), GN hess (N, N) = A^T diag(a''(r)) A).
        """
        A = np.asarray(A, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        f = A @ v + np.asarray(b, dtype=np.float64).reshape(-1)
        r = f - np.asarray(target, dtype=np.float64).reshape(-1)
        value, slope, curv = self._activation(activation, W, r, eps, f, lo, hi)
        return float(np.sum(value)), A.T @ slope, A.T @ (curv[:, None] * A)

    def ee_pos_cost_activated(self, q, p_des, W, activation=0, eps=0.0, ee=0):
        """End-effector position residual r = p(q) - p_des under an activation.

        Returns (value, grad_x (nx,) = [J_p^T a'(r); 0], GN hess_x (nx, nx) with the
        q-block J_p^T diag(a''(r)) J_p). The quadratic activation is ee_pos_cost.
        """
        nq = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        nx = nq + nv
        target = self._ee_target_name(ee)
        p_des = np.asarray(p_des, dtype=np.float64).reshape(-1)
        pose = np.asarray(self.end_effector_pose(q, ee_joint_names=target)[0], dtype=np.float64).reshape(-1)
        Jp = np.asarray(self.end_effector_pose_gradient(q, ee_joint_names=target)[0], dtype=np.float64)[:3, :]
        value, slope, curv = self._activation(activation, W, pose[:3] - p_des, eps)
        grad_x = np.zeros(nx, dtype=np.float64)
        grad_x[:nv] = Jp.T @ slope
        hess_x = np.zeros((nx, nx), dtype=np.float64)
        hess_x[:nv, :nv] = Jp.T @ (curv[:, None] * Jp)
        return float(np.sum(value)), grad_x, hess_x

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

        Uses the CMM and its configuration derivative in the Pinocchio
        tangent convention (local base velocity, world-aligned centroidal
        momentum ordered [linear; angular]):
          r       = h(q,qd) - h_des                        (6-vector)
          value   = 1/2 sum_r W[r] r[r]^2
          J       = [ (dA/dq) qd | A ]                    (6, 2*nv)
          grad_x  = J.T @ (W * r)                         (2*nv,)
          GN hess = J.T @ diag(W) @ J                     (2*nv, 2*nv)

        The gradient is exact; only the Hessian uses Gauss-Newton. Both q and
        qd blocks are present, including the cross blocks. Perturb q with
        self.integrate while holding the supplied local qd coordinates fixed.
        Outputs are tangent-state sized, NOT ambient nq+nv arrays with pads.
        The GN Hessian is positive semidefinite when W is nonnegative; it is
        not the exact cost Hessian at a general nonzero residual.
        """
        W = np.asarray(W, dtype=np.float64).reshape(-1)
        h_des = np.asarray(h_des, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)

        A, h = self.ccrba(q, qd)  # A (6, nv), h (6,)
        A = np.asarray(A, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64).reshape(-1)

        r = h - h_des
        value = 0.5 * float(np.sum(W * r * r))

        # dA[row, velocity_column, tangent_direction]; contract velocity_column.
        dA = np.asarray(self.dccrba(q), dtype=np.float64)
        Jq = np.einsum("rki,k->ri", dA, qd)
        J = np.hstack((Jq, A))
        grad_x = J.T @ (W * r)
        hess_x = J.T @ (W[:, None] * J)
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
