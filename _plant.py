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

        Pass-through to `self.integrator_grad`, mirroring CUDA
        `plant_step_gradient` (the s_dAB surface).
        """
        return np.asarray(
            self.integrator_grad(q, qd, u, dt, integrator_type=integrator_type),
            dtype=np.float64,
        )

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
