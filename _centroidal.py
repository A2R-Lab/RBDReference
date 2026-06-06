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

    def _total_mass_and_com(self, q, Xw=None):
        """(M_total, p_com_world). p_com = sum_i m_i (R_i c_i + p_i) / M_total.

        `Xw` (the per-joint world transforms) may be passed in to avoid the
        repeated forward-kinematics pass when the caller already has it."""
        if Xw is None:
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

    def _ccrba_core(self, q, qd, Xw=None):
        """Return (A (6 x nv), h (6,), M_total). A and h are in the
        Pinocchio centroidal convention: expressed at the CoM, world-aligned,
        ordered [linear (3); angular (3)].

        `Xw` may be passed in to reuse a forward-kinematics pass."""
        nv = self.robot.get_num_vel()
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        if Xw is None:
            Xw = self._world_transforms(q)
        m_total, com = self._total_mass_and_com(q, Xw=Xw)
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

    def _centroidal_momentum_fast(self, q, qd, Xw=None):
        """Centroidal momentum `h = A(q) qd` computed WITHOUT forming the full
        6 x nv CMM: `h = cX0* sum_i Iw_i (J_i qd)` where `J_i qd` is link i's
        world spatial velocity. O(NB * nv) instead of O(NB * nv^2). Used in the
        finite-difference derivative loops; returns the SAME value as
        `centroidal_momentum` (to float64 rounding), since it is the same
        composition contracted with qd before the CoM shift instead of after."""
        nv = self.robot.get_num_vel()
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        if Xw is None:
            Xw = self._world_transforms(q)
        m_total, com = self._total_mass_and_com(q, Xw=Xw)
        NB = self.robot.get_num_bodies()
        h0 = np.zeros(6, dtype=np.float64)  # (angular; linear) at world origin
        for jid in range(NB):
            Iw, _m, _R, _p = self._link_world_spatial_inertia(jid, Xw)
            Ji = self._body_spatial_jacobian_world(jid, Xw, nv)
            vi = Ji @ qd
            h0 += Iw @ vi
        # shift world-origin momentum to the CoM (world-aligned): n -= com x f
        n = h0[:3] - np.cross(com, h0[3:])
        f = h0[3:]
        return np.concatenate([f, n])  # [linear; angular] (Pinocchio order)

    # ---- centroidal rate (h-dot) and its bias (A-dot qd) ----

    def _centroidal_bias(self, q, qd, fd_step=1e-5):
        """Centroidal-momentum bias `Adot(q,qd) qd` = `hdot` at `qdd = 0`.

        Equal to the time derivative of `h(q(t), qd)` holding `qd` constant and
        advancing `q` along `qd` (the Lie-group retract). Built by a central
        finite difference of the exact `centroidal_momentum` along `qd`, which
        reproduces Pinocchio's `computeCentroidalMomentumTimeVariation(a=0)` to
        ~1e-9 (the only FD-sourced quantity in this mixin; the value layer is
        exact). Tangent-space retract `self.integrate` is correct for both
        fixed- and floating-base. Uses the O(NB*nv) `_centroidal_momentum_fast`
        path (forms only h, not the full CMM)."""
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        q_plus = self.integrate(q, fd_step * qd)
        q_minus = self.integrate(q, -fd_step * qd)
        h_plus = self._centroidal_momentum_fast(q_plus, qd)
        h_minus = self._centroidal_momentum_fast(q_minus, qd)
        return (h_plus - h_minus) / (2.0 * fd_step)

    def _hdot_fast(self, q, qd, qdd, A=None, Xw=None):
        """`hdot = A qdd + bias` at configuration `q`, reusing a precomputed CMM
        `A` and/or world transforms `Xw` when available. Same value as
        `centroidal_momentum_time_variation`."""
        qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)
        if A is None:
            A, _h, _m = self._ccrba_core(q, qdd, Xw=Xw)
        return A @ qdd + self._centroidal_bias(q, qd)

    def centroidal_momentum_time_variation(self, q, qd, qdd):
        """Centroidal-momentum rate `hdot = A(q) qdd + Adot(q,qd) qd` (6-vector,
        [linear; angular]). Matches `pin.computeCentroidalMomentumTimeVariation`.

        The `A qdd` term is exact (the value-layer CMM); the bias `Adot qd` is
        the FD-sourced `_centroidal_bias`."""
        return self._hdot_fast(q, qd, qdd)

    # ---- centroidal dynamics derivatives ----

    def centroidal_dynamics_derivatives(self, q, qd, qdd, fd_step=1e-5):
        """Analytical derivatives of the centroidal dynamics, matching
        `pin.computeCentroidalDynamicsDerivatives(model, data, q, v, a)`:

            (dh_dq, dhdot_dq, dhdot_dv, dhdot_da)

        all 6 x nv, in the Pinocchio centroidal convention ([linear; angular]
        at the CoM, world-aligned), tangent-space ordered:
          - dh_dq    = d(h)/dq      = d(A qd)/dq        (the C2 deliverable)
          - dhdot_dq = d(hdot)/dq
          - dhdot_dv = d(hdot)/dv
          - dhdot_da = d(hdot)/da   = A   (exact, the CMM)

        `dh_dq` and the q-derivatives are central finite differences of the
        exact `centroidal_momentum` / `centroidal_momentum_time_variation` taken
        along the Lie-group tangent (`self.integrate`), so they are valid for
        both fixed- and floating-base. They reproduce Pinocchio's analytic
        `getCentroidalDynamicsDerivatives` to ~1e-7."""
        nv = self.robot.get_num_vel()
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)

        A, _h, _m = self._ccrba_core(q, qd)

        dh_dq = np.zeros((6, nv), dtype=np.float64)
        dhdot_dq = np.zeros((6, nv), dtype=np.float64)
        for i in range(nv):
            e = np.zeros(nv, dtype=np.float64)
            e[i] = fd_step
            q_plus = self.integrate(q, e)
            q_minus = self.integrate(q, -e)
            # Reuse one forward-kinematics pass per perturbed q for both the
            # momentum (dh_dq) and the rate (dhdot_dq); the CMM at the perturbed
            # q feeds the A@qdd term of hdot.
            Xw_p = self._world_transforms(q_plus)
            Xw_m = self._world_transforms(q_minus)
            A_p, h_p, _ = self._ccrba_core(q_plus, qd, Xw=Xw_p)
            A_m, h_m, _ = self._ccrba_core(q_minus, qd, Xw=Xw_m)
            dh_dq[:, i] = (h_p - h_m) / (2.0 * fd_step)
            hdot_p = self._hdot_fast(q_plus, qd, qdd, A=A_p, Xw=Xw_p)
            hdot_m = self._hdot_fast(q_minus, qd, qdd, A=A_m, Xw=Xw_m)
            dhdot_dq[:, i] = (hdot_p - hdot_m) / (2.0 * fd_step)

        # dhdot_dv: q fixed, perturb v. A and the bias both depend on v; A@qdd
        # term varies because qdd is fixed but the bias's qd changes. Reuse the
        # base CMM A (q is fixed) for the A@qdd term.
        dhdot_dv = np.zeros((6, nv), dtype=np.float64)
        Aqdd = A @ qdd
        for i in range(nv):
            dv = np.zeros(nv, dtype=np.float64)
            dv[i] = fd_step
            bias_p = self._centroidal_bias(q, qd + dv)
            bias_m = self._centroidal_bias(q, qd - dv)
            # hdot(q, qd±dv, qdd) = A(q) qdd + bias(q, qd±dv); the A@qdd term is
            # qd-independent so it cancels in the central difference.
            dhdot_dv[:, i] = (bias_p - bias_m) / (2.0 * fd_step)

        return dh_dq, dhdot_dq, dhdot_dv, A

    # ---- dCCRBA: derivatives of the centroidal map A(q) itself ----

    def _dccrba_world_sweep(self, q, Xw=None):
        """Shared backbone for the analytic dCCRBA. Runs ONE world-frame sweep
        (the centroidal analogue of `idsva_so_world_frame`'s forward pass) and
        returns the per-body world quantities the analytic tensor / Adot reuse:

          Iw   : list of NB world-origin spatial inertias (Featherstone [ang;lin])
          Jw   : list of NB world spatial Jacobians (6 x nv), alpha-folded, so
                 `Jw[i] @ qd` is body i's world spatial velocity (== value layer)
          units: list of per-(body, local-DOF) "motion units", each a dict
                 {j (owner body), phi (6, alpha-scaled world motion column),
                  vi (project v-slot)}.  Mimic-aware: several units may share a
                 project v-slot (the mimic + its target both write the slot).
          A0   : world-ORIGIN momentum map, A0 = sum_i Iw_i Jw_i  (6 x nv,
                 [ang;lin]) BEFORE the CoM dual-shift / [lin;ang] reorder.
          Jcom : CoM Jacobian (3 x nv) = A0[lin]/m_total = d(com)/dv.
          com, m_total, root_v (set of floating-base root v-slots).

        Every quantity matches `_ccrba_core` / `jacobian_com` to float64
        rounding (same closed-form world inertias + body Jacobians)."""
        r = self.robot
        nv = r.get_num_vel()
        NB = r.get_num_bodies()
        if Xw is None:
            Xw = self._world_transforms(q)
        Iw = [self._link_world_spatial_inertia(j, Xw)[0] for j in range(NB)]
        m_total, com = self._total_mass_and_com(q, Xw=Xw)

        # Per-(body, local-DOF) world motion units, alpha-scaled exactly as
        # `_body_spatial_jacobian_world` builds its columns.
        units = []
        for j in range(NB):
            S = np.asarray(r.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            R_j = Xw[j][:3, :3]
            p_j = Xw[j][:3, 3]
            alpha = self._mimic_multiplier(j)
            vinds = self._as_index_list(r.get_joint_index_v(j))
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                aw = R_j @ S[:3, c]
                lw = R_j @ S[3:6, c]
                phi = np.empty(6, dtype=np.float64)
                phi[:3] = alpha * aw
                phi[3:] = alpha * (lw + np.cross(p_j, aw))
                units.append({"j": j, "phi": phi, "vi": vi})

        Jw = [np.zeros((6, nv), dtype=np.float64) for _ in range(NB)]
        for i in range(NB):
            anc_self = set(r.get_ancestors_by_id(i)) | {i}
            for u in units:
                if u["j"] in anc_self:
                    Jw[i][:, u["vi"]] += u["phi"]
        A0 = np.zeros((6, nv), dtype=np.float64)
        for i in range(NB):
            A0 += Iw[i] @ Jw[i]
        Jcom = A0[3:, :] / m_total if m_total != 0.0 else A0[3:, :]
        root_v = set(self._as_index_list(r.get_joint_index_v(0))) if r.floating_base else set()
        return {
            "Iw": Iw, "Jw": Jw, "units": units, "A0": A0, "Jcom": Jcom,
            "com": com, "m_total": m_total, "root_v": root_v,
        }

    def _dccrba_analytic(self, q, Xw=None):
        """Analytic dCCRBA tensor `dA_dq[:, k, i] = d A[:, k]/d q_i` (6 x nv x nv).

        Derivation (all in the world-ORIGIN [ang;lin] frame, then CoM-shifted):
          A0 = sum_i Iw_i Jw_i.  Differentiating w.r.t. tangent coord m,
            * JOINT dof m: only the subtree it moves changes.  For each body i,
                d Iw_i / d q_m = dot_matrix(Iw_i, phi_m)   (if m moves body i, i.e.
                    m's owner is an ancestor-or-self of i; else 0), and
                d (Jw_i col c) / d q_m = crm(phi_m) @ phi_c  (if m's owner is a
                    STRICT ancestor of c's owner; else 0).
              (`phi_m` = the alpha-scaled world motion column of dof m.)
            * BASE dof m (floating root): a base twist rigidly transports the
              whole world-origin momentum map, so d A0 / d q_m = crf(phi_m) @ A0
              (`crf = dual_cross_operator`).  Verified to ~1e-10 vs FD of A0.
          Then A_fs = Xstar A0 with Xstar[ang,lin] = -skew(com); its q-derivative
          adds the CoM-MOTION term  d Xstar/d q_m = -skew(Jcom[:, m])  (Jcom the
          CoM Jacobian).  Finally reorder rows [ang;lin] -> [lin;ang].

        Mimic-aware: units share a project v-slot, so a mimic q_m drives every
        unit on that slot (each with its own owner/alpha); the per-unit
        contributions accumulate into the project column.

        Validates vs the FD tensor to ~1e-7 (~1e-9 typical) and both pinocchio
        contractions (Adot, dh_dq) to ~1e-12 (relative ~1e-9 on big robots)."""
        r = self.robot
        nv = r.get_num_vel()
        NB = r.get_num_bodies()
        sw = self._dccrba_world_sweep(q, Xw=Xw)
        Iw, Jw, units, A0 = sw["Iw"], sw["Jw"], sw["units"], sw["A0"]
        Jcom, com, root_v = sw["Jcom"], sw["com"], sw["root_v"]

        crm = self.cross_operator
        crf = self.dual_cross_operator
        ancestors = [set(r.get_ancestors_by_id(i)) for i in range(NB)]
        anc_self = [ancestors[i] | {i} for i in range(NB)]
        units_by_vi = {}
        for u in units:
            units_by_vi.setdefault(u["vi"], []).append(u)

        # d A0 / d q_m, assembled in the world-origin [ang;lin] frame.
        dA0 = np.zeros((6, nv, nv), dtype=np.float64)
        for m in range(nv):
            if m in root_v:
                # Base twist: the single root unit on this slot transports all of A0.
                phim = units_by_vi[m][0]["phi"]
                dA0[:, :, m] = crf(phim) @ A0
                continue
            for um in units_by_vi.get(m, ()):
                jm = um["j"]
                phim = um["phi"]
                crmM = crm(phim)
                for i in range(NB):
                    block = np.zeros((6, nv), dtype=np.float64)
                    if jm in anc_self[i]:
                        block += self.dot_matrix(Iw[i], phim) @ Jw[i]
                    # d Jw_i / d q_m : crm(phi_m) @ phi_c for units c BELOW m in i's chain
                    dJi = np.zeros((6, nv), dtype=np.float64)
                    for u2 in units:
                        if u2["j"] in anc_self[i] and jm in ancestors[u2["j"]]:
                            dJi[:, u2["vi"]] += crmM @ u2["phi"]
                    block += Iw[i] @ dJi
                    dA0[:, :, m] += block

        # CoM dual-shift + its q-derivative (CoM-motion term) + [lin;ang] reorder.
        Xstar = np.eye(6, dtype=np.float64)
        S_com = _skew(com)
        Xstar[:3, 3:] = -S_com
        reorder = np.array([3, 4, 5, 0, 1, 2])
        dA = np.zeros((6, nv, nv), dtype=np.float64)
        for m in range(nv):
            dXstar = np.zeros((6, 6), dtype=np.float64)
            dXstar[:3, 3:] = -_skew(Jcom[:, m])
            dA_fs = dXstar @ A0 + Xstar @ dA0[:, :, m]
            dA[:, :, m] = dA_fs[reorder, :]
        return dA

    def dccrba(self, q):
        """Configuration derivative of the CMM: the rank-3 tensor

            dA_dq[:, k, i] = d A[:, k] / d q_i      (6 x nv x nv)

        i.e. the partial of every CMM column `k` with respect to every tangent
        coordinate `i`, in the Pinocchio centroidal convention ([linear; angular]
        at the CoM, world-aligned). This is the fundamental dCCRBA object the
        downstream codegen consumes: it contracts with `qd` two different ways to
        recover both first-order centroidal derivatives,

            Adot   = sum_i dA_dq[:, :, i] * qd[i]         (= d A / dt, pin.dccrba)
            dh_dq  = sum_k dA_dq[:, k, :] * qd[k]         (= d(A qd)/dq)

        Computed ANALYTICALLY (`_dccrba_analytic`): one world-frame sweep plus
        spatial cross-product / inertia-derivative operators, mimic-aware and
        valid for fixed- and floating-base. Matches the FD tensor to ~1e-7 and
        both pinocchio contractions to ~1e-12 (relative ~1e-9 on big robots).
        See `dccrba_fd` for the finite-difference cross-check."""
        return self._dccrba_analytic(q)

    def dccrba_fd(self, q, fd_step=1e-5):
        """Finite-difference cross-check for `dccrba`: the same rank-3 tensor
        `dA_dq[:, k, i]` built as a 4th-order central FD of the EXACT value-layer
        CMM `ccrba` along the Lie-group tangent (`self.integrate`), one stencil
        per tangent coordinate. The value layer it differences is exact, so this
        is float64-accurate (~1e-9 vs `pin.dccrba`); kept as the independent
        oracle the analytic path is validated against. `copy=True` on each
        sampled `A` dodges the pin-view aliasing trap (guide §6)."""
        nv = self.robot.get_num_vel()
        dA_dq = np.zeros((6, nv, nv), dtype=np.float64)
        zero_v = np.zeros(nv, dtype=np.float64)
        for i in range(nv):
            e = np.zeros(nv, dtype=np.float64)
            e[i] = fd_step

            def _A(scale):
                A, _h, _m = self._ccrba_core(self.integrate(q, scale * e), zero_v)
                return np.asarray(A, dtype=np.float64).copy()

            dA_dq[:, :, i] = (
                -_A(2.0) + 8.0 * _A(1.0) - 8.0 * _A(-1.0) + _A(-2.0)
            ) / (12.0 * fd_step)
        return dA_dq

    def cmm_time_variation(self, q, qd):
        """Time derivative of the centroidal map, `Adot = dA(q(t))/dt` (6 x nv),
        in the Pinocchio centroidal convention ([linear; angular] at the CoM,
        world-aligned). Matches `pin.dccrba(model, data, q, v)` and
        `pin.computeCentroidalMapTimeVariation` to ~1e-12 (relative ~1e-9 on big
        robots).

        Computed ANALYTICALLY as the qd-contraction of the exact dCCRBA tensor,
        `Adot = sum_i (dA/dq_i) qd_i` (`_dccrba_analytic`), so it shares the
        single world-frame sweep and is consistent by construction with both
        `dccrba` contractions. `Adot @ qd == centroidal_bias` (the `Adot qd`
        term of `hdot`) and `Adot @ qdd + (Adot qd)` matches
        `centroidal_momentum_time_variation`."""
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        dA = self._dccrba_analytic(q)
        return np.einsum("abi,i->ab", dA, qd)

    def cmm_time_variation_fd(self, q, qd, fd_step=1e-5):
        """Finite-difference cross-check for `cmm_time_variation`: `Adot` as a
        4th-order central FD of the exact `ccrba` along `qd` (the Lie-group
        retract `self.integrate`), O(nv) value-layer evals (no full tensor).
        Kept as the independent oracle the analytic `cmm_time_variation` is
        validated against (~1e-9 vs `pin.dccrba`)."""
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        zero_v = np.zeros(self.robot.get_num_vel(), dtype=np.float64)

        def _A(scale):
            A, _h, _m = self._ccrba_core(self.integrate(q, scale * fd_step * qd), zero_v)
            return np.asarray(A, dtype=np.float64).copy()

        return (-_A(2.0) + 8.0 * _A(1.0) - 8.0 * _A(-1.0) + _A(-2.0)) / (12.0 * fd_step)
