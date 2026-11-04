"""Constraint embedding of explicit couplings (2026-11-03, item E of 0.8).

A follower joint whose coordinate is an explicit function of a leader's
(a URDF mimic, GRiD's polynomial <mimic polycoef>) is a one-body cluster:
the full spanning tree, in which every joint owns a coordinate, is projected
onto the independent coordinates y through

    q_full = F(y),   qd_full = G(y) yd,   qdd_full = G(y) ydd + Gdot(y, yd) yd,

with G = dF/dy (a follower row holds alpha = dq_f/dq_l at its leader's slot)
and Gdot yd the row alpha' yd_l^2. Every reduced quantity is the projection
of the full tree's:

    tau_y = G^T tau_full(F(y), G yd, G ydd + Gdot yd)
    M_y   = G^T M_full G
    c_y   = G^T rnea_full(F(y), G yd, Gdot yd)          (bias, qdd = Gdot yd)
    ydd   = M_y^{-1} (tau_y - c_y)

and the first-order derivatives follow by the chain rule with the full
tree's gradients (dG/dy carries alpha', d(Gdot yd)/dy carries alpha''). For a
linear coupling (alpha constant) this is the mimic fold the reduced
reference and the generated kernels already carry, so the two must agree to
round-off; for a polynomial coupling the projection is the oracle the
kernels are checked against, and it is itself checked against finite
differences through the chart.
"""
from __future__ import annotations

import numpy as np


class EmbeddedReference:
    """Projection of a full-tree ``RBDReference`` onto reduced coordinates."""

    def __init__(self, reduced_robot, full_reference, reduced=None):
        self.robot = reduced_robot
        self.full = full_reference
        # the reduced RBDReference (joint-dynamics settings: use_joint_dynamics,
        # friction_model, friction_width per reduced slot); None adds no bias
        self.reduced = reduced
        self.full_robot = full_reference.robot
        self.nv = reduced_robot.get_num_vel()
        self.nq = reduced_robot.get_num_pos()
        self.nv_full = self.full_robot.get_num_vel()
        self.nq_full = self.full_robot.get_num_pos()
        self._build_maps()

    # ── maps ──────────────────────────────────────────────────────────────
    def _build_maps(self):
        """Per full-tree joint: its q and v slots in the full tree, and either
        the reduced slots it owns (independent) or its coupling (follower)."""
        self.independent = []   # (full_q_idx list, full_v_idx list, red_q_idx list, red_v_idx list)
        self.followers = []     # (joint, full_q_idx, full_v_idx, leader_full_q_idx, leader_red_q_idx, leader_red_v_idx)
        by_name = {j.get_name(): j for j in self.robot.get_joints_ordered_by_id()}
        for fj in self.full_robot.get_joints_ordered_by_id():
            rj = by_name[fj.get_name()]
            fq = self._aslist(self.full_robot.get_joint_index_q(fj.get_id()))
            fv = self._aslist(self.full_robot.get_joint_index_v(fj.get_id()))
            if getattr(rj, "is_mimic", False):
                leader = self.robot.get_joint_by_id(rj.get_mimic_target_id())
                lf = self.full_robot.get_joint_by_name(leader.get_name())
                lfq = self._aslist(self.full_robot.get_joint_index_q(lf.get_id()))
                lrq = self._aslist(self.robot.get_joint_index_q(leader.get_id()))
                lrv = self._aslist(self.robot.get_joint_index_v(leader.get_id()))
                assert len(fq) == 1 and len(fv) == 1 and len(lfq) == 1, "couplings join single-DoF joints"
                self.followers.append((rj, fq[0], fv[0], lfq[0], lrq[0], lrv[0]))
            else:
                rq = self._aslist(self.robot.get_joint_index_q(rj.get_id()))
                rv = self._aslist(self.robot.get_joint_index_v(rj.get_id()))
                assert len(rq) == len(fq) and len(rv) == len(fv)
                self.independent.append((fq, fv, rq, rv))

    @staticmethod
    def _aslist(x):
        return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]

    # ── the map and its derivatives ───────────────────────────────────────
    def expand_q(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        q = np.zeros(self.nq_full)
        for fq, _, rq, _ in self.independent:
            q[fq] = y[rq]
        for joint, fqi, _, lfq, lrq, _ in self.followers:
            q[fqi] = joint.coupling_value(float(y[lrq]))
        return q

    def _alphas(self, y):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        out = []
        for joint, fqi, fvi, lfq, lrq, lrv in self.followers:
            ql = float(y[lrq])
            out.append((fvi, lrv, joint.coupling_alpha(ql), joint.coupling_alpha_d(ql), joint.coupling_alpha_dd(ql)))
        return out

    def G(self, y):
        G = np.zeros((self.nv_full, self.nv))
        for _, fv, _, rv in self.independent:
            for a, b in zip(fv, rv):
                G[a, b] = 1.0
        for fvi, lrv, al, _, _ in self._alphas(y):
            G[fvi, lrv] = al
        return G

    def dG_dy(self, y):
        """dG/dy_k as a list of (k, row, value) triples (only follower rows)."""
        return [(lrv, fvi, ald) for fvi, lrv, _, ald, _ in self._alphas(y)]

    def expand_qd(self, y, yd):
        return self.G(y) @ np.asarray(yd, dtype=np.float64).reshape(-1)

    def gdot_yd(self, y, yd):
        """Gdot(y, yd) yd: follower rows alpha' yd_l^2."""
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        out = np.zeros(self.nv_full)
        for fvi, lrv, _, ald, _ in self._alphas(y):
            out[fvi] = ald * yd[lrv] ** 2
        return out

    def expand_qdd(self, y, yd, ydd):
        return self.G(y) @ np.asarray(ydd, dtype=np.float64).reshape(-1) + self.gdot_yd(y, yd)

    def reduce(self, y, vec_full):
        return self.G(y).T @ np.asarray(vec_full, dtype=np.float64).reshape(-1)

    # ── dynamics ──────────────────────────────────────────────────────────
    def inverse_dynamics(self, y, yd, ydd=None, GRAVITY=-9.81, f_ext=None):
        q = self.expand_q(y)
        qd = self.expand_qd(y, yd)
        # ydd = 0 is not qdd_full = 0: the followers accelerate by Gdot yd.
        qdd = self.gdot_yd(y, yd) if ydd is None else self.expand_qdd(y, yd, ydd)
        c, v, a, f = self.full.inverse_dynamics(q, qd, qdd, GRAVITY, f_ext=f_ext)
        return (self.reduce(y, c) + self.joint_dynamics_bias(y, yd), v, a, f)

    def crba(self, y):
        G = self.G(y)
        return G.T @ np.asarray(self.full.crba(self.expand_q(y)), dtype=np.float64) @ G

    def minv(self, y, output_dense=True):
        return np.linalg.inv(self.crba(y))

    def bias(self, y, yd, GRAVITY=-9.81, f_ext=None):
        """c_y = G^T rnea_full(F(y), G yd, Gdot yd)."""
        q = self.expand_q(y)
        c, _, _, _ = self.full.inverse_dynamics(q, self.expand_qd(y, yd), self.gdot_yd(y, yd), GRAVITY, f_ext=f_ext)
        return self.reduce(y, c) + self.joint_dynamics_bias(y, yd)

    def coriolis_matrix(self, y, yd, GRAVITY=-9.81):
        """C_y = G^T C_f G + G^T M_f Gdot, so that C_y yd + g_y is the bias
        (the follower accelerates by Gdot yd at ydd = 0) and
        C_y + C_y^T = d/dt M_y (the Pinocchio convention of C_f is kept)."""
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        q = self.expand_q(y)
        G = self.G(y)
        C = np.asarray(self.full.coriolis_matrix(q, G @ yd, GRAVITY), dtype=np.float64)
        M = np.asarray(self.full.crba(q), dtype=np.float64)
        Gdot = np.zeros((self.nv_full, self.nv))
        for fvi, lrv, _, ald, _ in self._alphas(y):
            Gdot[fvi, lrv] = ald * yd[lrv]
        return G.T @ C @ G + G.T @ M @ Gdot

    def forward_dynamics(self, y, yd, u, f_ext=None, GRAVITY=-9.81):
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        return np.linalg.solve(self.crba(y), u - self.bias(y, yd, GRAVITY, f_ext))

    def aba(self, y, yd, tau, f_ext=None, GRAVITY=-9.81):
        return self.forward_dynamics(y, yd, tau, f_ext=f_ext, GRAVITY=GRAVITY)

    def inverse_dynamics_gradient(self, y, yd, ydd=None, GRAVITY=-9.81, f_ext=None):
        """[dtau_y/dy | dtau_y/dyd] (nv x 2nv) by the chain rule through the
        full tree's [dtau/dq | dtau/dqd] and M_full = dtau/dqdd."""
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        ydd = np.zeros(self.nv) if ydd is None else np.asarray(ydd, dtype=np.float64).reshape(-1)
        q = self.expand_q(y)
        qd = self.expand_qd(y, yd)
        qdd = self.expand_qdd(y, yd, ydd)
        G = self.G(y)
        c, _, _, _ = self.full.inverse_dynamics(q, qd, qdd, GRAVITY, f_ext=f_ext)
        dc = np.asarray(self.full.inverse_dynamics_gradient(q, qd, qdd, GRAVITY, f_ext=f_ext), dtype=np.float64)
        dc_dq, dc_dqd = dc[:, :self.nv_full], dc[:, self.nv_full:]
        M = np.asarray(self.full.crba(q), dtype=np.float64)
        # d q_full / d y = G (the follower's q is F and dF/dy = alpha at the leader slot)
        dq_dy = G
        dqd_dy = np.zeros((self.nv_full, self.nv))
        dqdd_dy = np.zeros((self.nv_full, self.nv))
        dqdd_dyd = G * 0.0
        dG_T_c = np.zeros((self.nv, self.nv))
        for fvi, lrv, al, ald, aldd in self._alphas(y):
            dqd_dy[fvi, lrv] = ald * yd[lrv]
            dqdd_dy[fvi, lrv] = ald * ydd[lrv] + aldd * yd[lrv] ** 2
            dqdd_dyd[fvi, lrv] = 2.0 * ald * yd[lrv]
            dG_T_c[lrv, lrv] += ald * c[fvi]
        dtau_dy = G.T @ (dc_dq @ dq_dy + dc_dqd @ dqd_dy + M @ dqdd_dy) + dG_T_c
        dtau_dyd = G.T @ (dc_dqd @ G + M @ dqdd_dyd)
        jd_dy, jd_dyd = self.joint_dynamics_gradient(y, yd)
        return np.hstack([dtau_dy + jd_dy, dtau_dyd + jd_dyd])

    # ── joint dynamics (damping, Coulomb friction) in the reduced coordinates ──
    def _joint_dynamics_terms(self, y):
        """(reduced v-slot, alpha, alpha', b, fr) per joint of the reduced
        robot that declares damping or friction: an independent joint has
        alpha = 1, a follower alpha(y_l) at its leader's slot. The floating
        root carries none."""
        red = self.reduced
        if red is None or not red.use_joint_dynamics:
            return []
        robot = self.robot
        has_b = robot.robot_has_joint_damping()
        has_f = robot.robot_has_joint_friction()
        if not (has_b or has_f):
            return []
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        by_id = {}
        for joint, fqi, fvi, lfq, lrq, lrv in self.followers:
            ql = float(y[lrq])
            by_id[joint.get_id()] = (lrv, joint.coupling_alpha(ql), joint.coupling_alpha_d(ql))
        out = []
        for joint in robot.get_joints_ordered_by_id():
            jid = joint.get_id()
            if robot.floating_base and robot.get_parent_id(jid) == -1:
                continue
            b = float(robot.get_damping_by_id(jid)) if has_b else 0.0
            fr = float(robot.get_friction_by_id(jid)) if has_f else 0.0
            if b == 0.0 and fr == 0.0:
                continue
            if jid in by_id:
                lrv, al, ald = by_id[jid]
                out.append((lrv, al, ald, b, fr))
            else:
                for k in self._aslist(robot.get_joint_index_v(jid)):
                    out.append((int(k), 1.0, 0.0, b, fr))
        return out

    def _friction_shape_and_slope(self, k, qd_k):
        """(s(qd), ds/dqd) of the reduced slot's friction shape: sign (slope 0)
        or tanh(qd / w_k) with slope sech^2 / w_k."""
        red = self.reduced
        if red.friction_model == "tanh":
            w = float(red.friction_width[k])
            th = np.tanh(qd_k / w)
            return th, (1.0 - th * th) / w
        return float(np.sign(qd_k)), 0.0

    def joint_dynamics_bias(self, y, yd):
        """tau_y += alpha^2 b yd_l + |alpha| fr s(yd_l) per damped joint."""
        out = np.zeros(self.nv)
        terms = self._joint_dynamics_terms(y)
        if not terms:
            return out
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        for k, al, _, b, fr in terms:
            s, _ = self._friction_shape_and_slope(k, float(yd[k]))
            out[k] += al * al * b * yd[k] + abs(al) * fr * s
        return out

    def joint_dynamics_gradient(self, y, yd):
        """(d/dy, d/dyd) of joint_dynamics_bias, both nv x nv: the dyd
        diagonal alpha^2 b + |alpha| fr s'(yd_l); the dy diagonal (the fold's
        own dependence on the leader) 2 alpha alpha' b yd_l +
        sign(alpha) alpha' fr s(yd_l)."""
        d_y = np.zeros((self.nv, self.nv))
        d_yd = np.zeros((self.nv, self.nv))
        terms = self._joint_dynamics_terms(y)
        if not terms:
            return d_y, d_yd
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        for k, al, ald, b, fr in terms:
            s, sl = self._friction_shape_and_slope(k, float(yd[k]))
            d_yd[k, k] += al * al * b + abs(al) * fr * sl
            d_y[k, k] += 2.0 * al * ald * b * yd[k] + np.sign(al) * ald * fr * s
        return d_y, d_yd

    def forward_dynamics_gradient(self, y, yd, u, f_ext=None, GRAVITY=-9.81):
        ydd = self.forward_dynamics(y, yd, u, f_ext=f_ext, GRAVITY=GRAVITY)
        dc = self.inverse_dynamics_gradient(y, yd, ydd, GRAVITY, f_ext=f_ext)
        dc_dy, dc_dyd = np.hsplit(dc, [self.nv])
        minv = self.minv(y)
        return (-minv @ dc_dy, -minv @ dc_dyd)

    # ── kinematics ────────────────────────────────────────────────────────
    def end_effector_pose(self, y, ee_joint_names=None, ee_offsets=None):
        return self.full.end_effector_pose(self.expand_q(y), ee_joint_names, ee_offsets)

    def end_effector_pose_gradient(self, y, ee_joint_names=None, ee_offsets=None):
        """The full tree's pose gradient right-multiplied by G (chain rule
        through q = F(y)); the trailing axis is the velocity index."""
        grad = np.asarray(self.full.end_effector_pose_gradient(self.expand_q(y), ee_joint_names, ee_offsets), dtype=np.float64)
        return grad @ self.G(y)

    def frame_jacobian(self, y, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        return np.asarray(self.full.frame_jacobian(self.expand_q(y), frame_name, reference_frame)) @ self.G(y)

    def frame_velocity(self, y, yd, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        return self.full.frame_velocity(self.expand_q(y), self.expand_qd(y, yd), frame_name, reference_frame)

    def frame_acceleration(self, y, yd, ydd, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        return self.full.frame_acceleration(self.expand_q(y), self.expand_qd(y, yd), self.expand_qdd(y, yd, ydd),
                                            frame_name, reference_frame)

    # chain-rule blocks: d(G yd)/dy, d(G ydd + Gdot yd)/dy and /dyd (follower rows only)
    def _dqd_dy(self, y, yd):
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        out = np.zeros((self.nv_full, self.nv))
        for fvi, lrv, _, ald, _ in self._alphas(y):
            out[fvi, lrv] = ald * yd[lrv]
        return out

    def _dqdd_dy(self, y, yd, ydd):
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        ydd = np.asarray(ydd, dtype=np.float64).reshape(-1)
        out = np.zeros((self.nv_full, self.nv))
        for fvi, lrv, _, ald, aldd in self._alphas(y):
            out[fvi, lrv] = ald * ydd[lrv] + aldd * yd[lrv] ** 2
        return out

    def _dqdd_dyd(self, y, yd):
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        out = np.zeros((self.nv_full, self.nv))
        for fvi, lrv, _, ald, _ in self._alphas(y):
            out[fvi, lrv] = 2.0 * ald * yd[lrv]
        return out

    def frame_jacobian_dot(self, y, yd, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        """d/dt of J_y = J_f(F(y)) G(y): Jdot_f G + J_f Gdot."""
        q = self.expand_q(y)
        qd = self.expand_qd(y, yd)
        Jd = np.asarray(self.full.frame_jacobian_dot(q, qd, frame_name, reference_frame), dtype=np.float64)
        J = np.asarray(self.full.frame_jacobian(q, frame_name, reference_frame), dtype=np.float64)
        return Jd @ self.G(y) + J @ self._dqd_dy(y, yd)

    def frame_velocity_derivatives(self, y, yd, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        q = self.expand_q(y)
        qd = self.expand_qd(y, yd)
        dv_dq, dv_dqd = (np.asarray(m, dtype=np.float64) for m in
                         self.full.frame_velocity_derivatives(q, qd, frame_name, reference_frame))
        G = self.G(y)
        return dv_dq @ G + dv_dqd @ self._dqd_dy(y, yd), dv_dqd @ G

    def frame_acceleration_derivatives(self, y, yd, ydd, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        q = self.expand_q(y)
        qd = self.expand_qd(y, yd)
        qdd = self.expand_qdd(y, yd, ydd)
        dv_dq, da_dq, da_dqd, da_dqdd = (np.asarray(m, dtype=np.float64) for m in
                                         self.full.frame_acceleration_derivatives(q, qd, qdd, frame_name, reference_frame))
        G = self.G(y)
        dqd_dy = self._dqd_dy(y, yd)
        # dv/dy = dv/dq_f G + J_f d(G yd)/dy, with J_f = dv/dqd_f = da/dqdd_f
        return (dv_dq @ G + da_dqdd @ dqd_dy,
                da_dq @ G + da_dqd @ dqd_dy + da_dqdd @ self._dqdd_dy(y, yd, ydd),
                da_dqd @ G + da_dqdd @ self._dqdd_dyd(y, yd),
                da_dqdd @ G)

    # ── regressors and centroidal quantities ───────────────────────────────
    def inverse_dynamics_regressor(self, y, yd, ydd, GRAVITY=-9.81):
        """Y_y = G^T Y_f(F(y), G yd, G ydd + Gdot yd): tau_y = Y_y pi."""
        Y = np.asarray(self.full.inverse_dynamics_regressor(self.expand_q(y), self.expand_qd(y, yd),
                                                            self.expand_qdd(y, yd, ydd), GRAVITY=GRAVITY), dtype=np.float64)
        return self.G(y).T @ Y

    def kinetic_energy_regressor(self, y, yd):
        return self.full.kinetic_energy_regressor(self.expand_q(y), self.expand_qd(y, yd))

    def potential_energy_regressor(self, y, GRAVITY=-9.81):
        return self.full.potential_energy_regressor(self.expand_q(y), GRAVITY=GRAVITY)

    def com(self, y):
        return self.full.com(self.expand_q(y))

    def jacobian_com(self, y):
        return np.asarray(self.full.jacobian_com(self.expand_q(y)), dtype=np.float64) @ self.G(y)

    def ccrba(self, y, yd):
        A, h = self.full.ccrba(self.expand_q(y), self.expand_qd(y, yd))
        return np.asarray(A, dtype=np.float64) @ self.G(y), h

    def _Gdot(self, y, yd):
        yd = np.asarray(yd, dtype=np.float64).reshape(-1)
        Gdot = np.zeros((self.nv_full, self.nv))
        for fvi, lrv, _, ald, _ in self._alphas(y):
            Gdot[fvi, lrv] = ald * yd[lrv]
        return Gdot

    def dccrba(self, y):
        """dA_y[:, k, i] of A_y = A_f(F(y)) G(y): the full tensor contracted
        with G on both the column and the tangent, plus A_f[:, fv] alpha' at
        the leader's diagonal cell of every follower."""
        q = self.expand_q(y)
        G = self.G(y)
        dA = np.asarray(self.full.dccrba(q), dtype=np.float64)
        out = np.einsum("ajm,jk,mi->aki", dA, G, G)
        A = np.asarray(self.full.ccrba(q, np.zeros(self.nv_full))[0], dtype=np.float64)
        for fvi, lrv, _, ald, _ in self._alphas(y):
            out[:, lrv, lrv] += A[:, fvi] * ald
        return out

    def cmm_time_variation(self, y, yd):
        """Adot_y = Adot_f G + A_f Gdot."""
        q = self.expand_q(y)
        G = self.G(y)
        qd = G @ np.asarray(yd, dtype=np.float64).reshape(-1)
        Adot = np.asarray(self.full.cmm_time_variation(q, qd), dtype=np.float64)
        A = np.asarray(self.full.ccrba(q, qd)[0], dtype=np.float64)
        return Adot @ G + A @ self._Gdot(y, yd)

    def centroidal_momentum_time_variation(self, y, yd, ydd):
        """hdot = A_y ydd + Adot_y yd, both terms analytic (the full tree's
        hdot sources its bias from finite differences)."""
        ydd = np.asarray(ydd, dtype=np.float64).reshape(-1)
        A, _ = self.ccrba(y, yd)
        return np.asarray(A, dtype=np.float64) @ ydd + self.cmm_time_variation(y, yd) @ np.asarray(yd, dtype=np.float64).reshape(-1)

    def generalized_gravity(self, y, GRAVITY=-9.81):
        return self.reduce(y, self.full.generalized_gravity(self.expand_q(y), GRAVITY=GRAVITY))

    def nonlinear_effects(self, y, yd, GRAVITY=-9.81):
        return self.bias(y, yd, GRAVITY)

    def kinetic_energy(self, y, yd):
        return self.full.kinetic_energy(self.expand_q(y), self.expand_qd(y, yd))

    def potential_energy(self, y, GRAVITY=-9.81):
        return self.full.potential_energy(self.expand_q(y), GRAVITY=GRAVITY)

    def mechanical_energy(self, y, yd, GRAVITY=-9.81):
        return self.full.mechanical_energy(self.expand_q(y), self.expand_qd(y, yd), GRAVITY=GRAVITY)

    def osc_inertia(self, y, frame_name=None, reference_frame="LOCAL_WORLD_ALIGNED"):
        J = np.asarray(self.frame_jacobian(y, frame_name, reference_frame), dtype=np.float64)
        return np.linalg.inv(J @ self.minv(y) @ J.T)
