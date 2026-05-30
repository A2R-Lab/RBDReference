"""CoM + centroidal numpy reference (R2/R3 of the centroidal-quickwins plan).

Pure compositions of the existing `RBDReference` kinematics (the per-joint world
homogeneous transforms and motion subspaces). Validated against Pinocchio:
  - com                 vs pin.centerOfMass            (data.com[0])
  - jacobian_com        vs pin.jacobianCenterOfMass    (data.Jcom)
  - ccrba               vs pin.ccrba                   (data.Ag, data.hg)
  - centroidal_momentum vs pin.computeCentroidalMomentum

Frame convention (matches Pinocchio's `data.Ag` / `data.hg`): the centroidal
momentum is expressed at the CoM in a WORLD-ALIGNED frame, ordered
`[linear (3); angular (3)]`. The CMM `A` is 6 x nv with `h = A @ qd`; its linear
rows equal `M_total * jacobian_com`.

Construction: `h = sum_i  cX_i^*  I_i^world  v_i^world`, where for each link i
  - `I_i^world` is the link's spatial inertia rotated/translated to the world,
  - `v_i^world = J_i qd` is the link's spatial velocity (J_i its body Jacobian),
  - `cX_i^*` shifts the spatial force/momentum from the world origin to the CoM
    (world-aligned), i.e. the dual translation by -p_com.
So `A = sum_i cX0^* * I_i^world * J_i^world`, assembled column by column. This is
exactly Pinocchio's centroidal map definition, so it agrees to float64 rounding.
"""

import numpy as np


def _skew(p):
    return np.array(
        [[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]],
        dtype=np.float64,
    )


class _CentroidalMixin:
    """CoM / CoM-Jacobian / centroidal-momentum-matrix numpy reference."""

    # ---- shared kinematics: per-joint world homogeneous transforms ----

    def _world_transforms(self, q):
        """4x4 world homogeneous transform of every joint, plus a vinds helper.

        Mirrors the forward-kinematics pass in `end_effector_pose_gradient`.
        Returns (Xw list indexed by joint id).
        """
        q = self._normalize_kinematics_q(q)
        n_joints = self.robot.get_num_joints()
        Xw = [None] * n_joints
        for j in range(n_joints):
            q_arg = self.robot.q_for_joint(j, q)
            X_local = np.asarray(
                self.robot.get_Xmat_hom_Func_by_id(j)(q_arg), dtype=np.float64
            )
            par = self.robot.get_parent_id(j)
            Xw[j] = X_local if par == -1 else (Xw[par] @ X_local)
        return Xw

    def _link_world_spatial_inertia(self, jid, Xw):
        """Spatial inertia of link `jid` expressed in the WORLD frame.

        The body-frame 6x6 is `Ib = [[I_O, m*cx],[m*cx^T, m*I3]]` (the parser's
        `get_Imat_by_id`). With world transform (R, p) of the body frame, the
        world spatial inertia (Featherstone, [angular; linear] rows) is
            Iw = AdT^{-T} Ib AdT^{-1}
        which we form directly: rotate the inertia, then translate by p.
        """
        Ib = np.asarray(self.robot.get_Imat_by_id(jid), dtype=np.float64)
        R = Xw[jid][:3, :3]
        p = Xw[jid][:3, 3]
        m = float(Ib[5, 5])
        # body-frame angular inertia about body origin and first moment m*c:
        I_O = Ib[:3, :3]
        mc_skew = Ib[:3, 3:6]  # = m * skew(c)
        mc = np.array([mc_skew[2, 1], mc_skew[0, 2], mc_skew[1, 0]], dtype=np.float64)
        # Rotate to world axes:
        I_O_w = R @ I_O @ R.T
        mc_w = R @ mc
        # Build world spatial inertia at the world ORIGIN by translating from the
        # body origin (at world position p) using the parallel-axis on the 6x6:
        #   Iw = [[ I_O_w + (px*mc_w^T + mc_w*px^T)*... ]]  -> use the standard
        # spatial transform. Simplest exact form: assemble at body origin then
        # apply the dual translation X*^{-1} for offset p.
        S_mc = _skew(mc_w)
        S_p = _skew(p)
        # spatial inertia at world origin (angular-first 6x6):
        # top-left  = I_O_w + S_p (m*S_c_w) ... build via momentum map directly.
        # Use the closed form for translating a spatial inertia by p (Featherstone
        # 2.63): with h = mc_w (first moment), m mass:
        #   Iw_ang_ang = I_O_w - S_p S_mc - S_mc S_p ... careful; assemble via Ad.
        # Adjoint dual transform for pure translation p (no rotation), motion
        # [ang; lin]:  X = [[I,0],[S_p, I]];  Iw = X^{-T} I_body_world X^{-1}
        X = np.zeros((6, 6), dtype=np.float64)
        X[:3, :3] = np.eye(3)
        X[3:, 3:] = np.eye(3)
        X[3:, :3] = S_p
        Xinv = np.linalg.inv(X)
        Ibw = np.zeros((6, 6), dtype=np.float64)
        Ibw[:3, :3] = I_O_w
        Ibw[:3, 3:] = S_mc
        Ibw[3:, :3] = S_mc.T
        Ibw[3:, 3:] = m * np.eye(3)
        Iw = Xinv.T @ Ibw @ Xinv
        return Iw, m, R, p

    def _body_spatial_jacobian_world(self, jid, Xw, nv):
        """6 x nv world spatial Jacobian of link `jid`, ordered [angular; linear]
        (so J @ qd is the link's world spatial velocity in Featherstone order)."""
        J = np.zeros((6, nv), dtype=np.float64)
        chain = sorted(self.robot.get_ancestors_by_id(jid)) + [jid]
        for j in chain:
            S = np.asarray(self.robot.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            R_j = Xw[j][:3, :3]
            p_j = Xw[j][:3, 3]
            scale = self._mimic_multiplier(j)
            vinds = self.robot.get_joint_index_v(j)
            if not isinstance(vinds, (list, tuple, np.ndarray)):
                vinds = [vinds]
            else:
                vinds = list(vinds)
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang_local = S[:3, c]
                lin_local = S[3:6, c]
                # world spatial velocity contribution of this DOF (angular; linear
                # at the world origin):
                aw = R_j @ ang_local
                lw = R_j @ lin_local
                # spatial motion of a screw at joint j expressed at world origin:
                # angular = aw; linear_at_origin = lw + p_j x aw
                J[:3, vi] += scale * aw
                J[3:, vi] += scale * (lw + np.cross(p_j, aw))
        return J

    # ---- CoM ----

    def _total_mass_and_com(self, q):
        """(M_total, p_com_world). p_com = sum_i m_i (R_i c_i + p_i) / M_total."""
        Xw = self._world_transforms(q)
        NB = self.robot.get_num_bodies()
        m_total = 0.0
        first_moment = np.zeros(3, dtype=np.float64)
        for jid in range(NB):
            Ib = np.asarray(self.robot.get_Imat_by_id(jid), dtype=np.float64)
            m = float(Ib[5, 5])
            mc_skew = Ib[:3, 3:6]
            mc = np.array(
                [mc_skew[2, 1], mc_skew[0, 2], mc_skew[1, 0]], dtype=np.float64
            )
            R = Xw[jid][:3, :3]
            p = Xw[jid][:3, 3]
            # world first moment of this link = R @ (m*c) + m*p
            first_moment += R @ mc + m * p
            m_total += m
        com = first_moment / m_total if m_total != 0.0 else first_moment
        return m_total, com

    def com(self, q):
        """Center-of-mass world position (3-vector)."""
        _m, com = self._total_mass_and_com(q)
        return com

    def jacobian_com(self, q):
        """CoM Jacobian J_com (3 x nv): d(p_com)/dv.

        J_com = (1 / M_total) * sum_i m_i J_v,i, with J_v,i the linear part of
        link i's body Jacobian taken at the link's CoM. Equivalently the linear
        rows of the CMM A divided by M_total (used here, since A already carries
        the mass-weighted linear momentum)."""
        A, _h, m_total = self._ccrba_core(q, np.zeros(self.robot.get_num_vel()))
        return A[:3, :] / m_total

    # ---- centroidal momentum matrix ----

    def _ccrba_core(self, q, qd):
        """Return (A (6 x nv), h (6,), M_total). A and h are in the
        Pinocchio centroidal convention: expressed at the CoM, world-aligned,
        ordered [linear (3); angular (3)]."""
        nv = self.robot.get_num_vel()
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        Xw = self._world_transforms(q)
        m_total, com = self._total_mass_and_com(q)
        NB = self.robot.get_num_bodies()

        # Featherstone-ordered (angular; linear) momentum map at the world
        # origin: A0 = sum_i Iw_i J_i.
        A0 = np.zeros((6, nv), dtype=np.float64)
        for jid in range(NB):
            Iw, _m, _R, _p = self._link_world_spatial_inertia(jid, Xw)
            Ji = self._body_spatial_jacobian_world(jid, Xw, nv)
            A0 += Iw @ Ji

        # Shift the force/momentum from the world origin to the CoM (world
        # aligned). Dual translation by +com on a force [n (angular); f (linear)]:
        #   n_com = n - com x f ;  f_com = f
        # Build the 6x6 dual-shift in Featherstone (angular;linear) order:
        S_com = _skew(com)
        Xstar = np.eye(6, dtype=np.float64)
        Xstar[:3, 3:] = -S_com  # n_com = n - com x f
        A_fs = Xstar @ A0  # (angular; linear) at the CoM

        # Reorder to Pinocchio's [linear; angular] convention.
        A = np.vstack([A_fs[3:, :], A_fs[:3, :]])
        h = A @ qd
        return A, h, m_total

    def ccrba(self, q, qd):
        """Centroidal momentum matrix A (6 x nv) and momentum h = A qd (6,).

        Matches `pin.ccrba` (data.Ag, data.hg), ordered [linear; angular] at the
        CoM in a world-aligned frame."""
        A, h, _m = self._ccrba_core(q, qd)
        return A, h

    def centroidal_momentum(self, q, qd):
        """Centroidal momentum h = A(q) qd (6-vector, [linear; angular])."""
        _A, h, _m = self._ccrba_core(q, qd)
        return h
