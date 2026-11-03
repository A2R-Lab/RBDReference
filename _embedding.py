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

    def __init__(self, reduced_robot, full_reference):
        self.robot = reduced_robot
        self.full = full_reference
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
        return (self.reduce(y, c), v, a, f)

    def crba(self, y):
        G = self.G(y)
        return G.T @ np.asarray(self.full.crba(self.expand_q(y)), dtype=np.float64) @ G

    def minv(self, y, output_dense=True):
        return np.linalg.inv(self.crba(y))

    def bias(self, y, yd, GRAVITY=-9.81, f_ext=None):
        """c_y = G^T rnea_full(F(y), G yd, Gdot yd)."""
        q = self.expand_q(y)
        c, _, _, _ = self.full.inverse_dynamics(q, self.expand_qd(y, yd), self.gdot_yd(y, yd), GRAVITY, f_ext=f_ext)
        return self.reduce(y, c)

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
        return np.hstack([dtau_dy, dtau_dyd])

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
