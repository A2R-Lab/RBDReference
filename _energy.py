"""Energy / gravity / Coriolis numpy reference (R1 of the centroidal-quickwins plan).

Pure compositions of the existing `RBDReference` RNEA / CRBA, plus the CoM
(`com`, from the `_centroidal` mixin) for potential energy. Validated against
Pinocchio:
  - kinetic_energy        vs pin.computeKineticEnergy
  - potential_energy      vs pin.computePotentialEnergy
  - mechanical_energy     vs pin.computeMechanicalEnergy
  - generalized_gravity   vs pin.computeGeneralizedGravity
  - nonlinear_effects     vs pin.nonLinearEffects
  - coriolis_matrix       vs pin.computeCoriolisMatrix

Sign convention: `GRAVITY` is the scalar gravity acceleration (default -9.81),
matching `RBDReference.inverse_dynamics`. PE uses the gravity vector g = [0,0,GRAVITY] so
that `PE = -M_total * g . p_com` reproduces Pinocchio's potential energy.
"""

import numpy as np


class _EnergyMixin:
    """Energy / generalized-gravity / Coriolis numpy reference."""

    def generalized_gravity(self, q, GRAVITY=-9.81):
        """g(q) = RNEA(q, 0, 0): the generalized gravity torque (size nv)."""
        n = self.robot.get_num_vel()
        zero = np.zeros(n, dtype=np.float64)
        c, _v, _a, _f = self.inverse_dynamics(q, zero, zero, GRAVITY=GRAVITY)
        return np.asarray(c, dtype=np.float64).reshape(-1)

    def nonlinear_effects(self, q, qd, GRAVITY=-9.81):
        """c(q, qd) = RNEA(q, qd, 0) = C(q,qd) qd + g(q) (size nv)."""
        n = self.robot.get_num_vel()
        zero = np.zeros(n, dtype=np.float64)
        c, _v, _a, _f = self.inverse_dynamics(q, qd, zero, GRAVITY=GRAVITY)
        return np.asarray(c, dtype=np.float64).reshape(-1)

    def kinetic_energy(self, q, qd):
        """KE = 1/2 qd^T M(q) qd (M from CRBA)."""
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        M = np.asarray(self.crba(q), dtype=np.float64)
        return 0.5 * float(qd @ M @ qd)

    def potential_energy(self, q, GRAVITY=-9.81):
        """PE = -M_total * g . p_com, with g = [0, 0, GRAVITY].

        Matches Pinocchio's `computePotentialEnergy` (PE increases with height
        for the physical g < 0)."""
        m_total, com = self._total_mass_and_com(q)
        g_vec = np.array([0.0, 0.0, GRAVITY], dtype=np.float64)
        return -float(m_total) * float(g_vec @ com)

    def mechanical_energy(self, q, qd, GRAVITY=-9.81):
        """KE + PE."""
        return self.kinetic_energy(q, qd) + self.potential_energy(q, GRAVITY=GRAVITY)

    def coriolis_matrix(self, q, qd, GRAVITY=-9.81):
        """Coriolis matrix C(q, qd) with C qd + g = nonlinear effects.

        A direct port of Pinocchio's ``computeCoriolisMatrix`` spatial recursion
        (rnea.hxx ``CoriolisMatrixForwardStep`` / ``CoriolisMatrixBackwardStep``),
        computed in the world frame. It matches Pinocchio's *algorithm-specific*
        convention entrywise — both the symmetric part (= 1/2 d/dt M) AND the
        skew part (Pinocchio's body-composite-bias ``Bcrb`` recursion) — for
        fixed AND floating base.

        Per-body world-frame quantities (forward pass):
          oY[i]   single-body spatial inertia in the world frame
          ov[i]   body spatial velocity in the world frame
          oh[i]   = oY[i] @ ov[i] (spatial momentum, world frame)
          Sw[i]   motion subspace columns expressed in the world frame
          dJ[i]   = ov[i] x Sw[i]   (motion cross of the world velocity)
          B[i]    = oY[i].variation(1/2 ov[i]) + forceCross(1/2 oh[i])
                  = (crf(1/2 ov) oY - oY crm(1/2 ov)) + icrf(1/2 oh)

        Backward pass accumulates the CRBA composite inertia oYc and composite
        bias Bc (the ``Bcrb``) toward the root, and fills C blockwise:
          dFdv[d] = oYc[d] dJ[d] + Bc[d] Sw[d]
          C[v_i, v_d] = Sw[i]^T dFdv[d]                  (d in subtree(i))
          C[v_i, v_j] = (oYc[i] Sw[i])^T dJ[j]
                        + (Sw[i]^T Bc[i]) Sw[j]           (j an ancestor of i)

        Mimic joints scale each column/row contribution by the joint multiplier
        (1.0 for non-mimic joints), matching the CRBA assembly above. The
        floating-base root uses its real motion subspace S (a [v_lin; omega] ->
        [omega; v_lin] permutation block), which already maps the user-facing
        root velocity ordering into the internal spatial ordering, so no manual
        root permutation is applied here (unlike CRBA, which assembles the root
        block with S = I and reorders by hand).
        """
        q = self._normalize_q_input(q)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()

        cross = self.cross_operator
        dcross = self.dual_cross_operator

        def _vinds(i):
            vi = self.robot.get_joint_index_v(i)
            if not isinstance(vi, (list, tuple, np.ndarray)):
                vi = [vi]
            return list(np.asarray(vi).reshape(-1))

        # ---- forward pass: world-frame per-body quantities ----
        iX0 = [None] * NB      # body<-world spatial transform (iX0)
        oXi = [None] * NB      # world<-body spatial transform (0Xi = inv(iX0))
        ov = [None] * NB       # world-frame body spatial velocity
        oY = [None] * NB       # world-frame single-body spatial inertia
        Sw = [None] * NB       # world-frame motion subspace (6 x nv_i)
        dJ = [None] * NB       # ov x Sw
        B = [None] * NB        # per-body bias matrix
        vinds = [None] * NB
        alpha = [None] * NB

        for i in range(NB):
            parent = self.robot.get_parent_id(i)
            _q = self.robot.q_for_joint(i, q)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(_q)   # iX_parent
            iX0[i] = Xmat if parent == -1 else np.matmul(Xmat, iX0[parent])
            oXi[i] = np.linalg.inv(iX0[i])

            S = np.asarray(self.robot.get_S_by_id(i), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(6, 1)
            vinds[i] = _vinds(i)
            alpha[i] = self._mimic_multiplier(i)
            _qd = alpha[i] * qd[vinds[i]]

            Sw[i] = np.matmul(oXi[i], S)
            vJ = np.matmul(Sw[i], np.asarray(_qd, dtype=np.float64).reshape(-1, 1)).reshape(-1)
            ov[i] = vJ if parent == -1 else (ov[parent] + vJ)

            Iloc = np.asarray(self.robot.get_Imat_by_id(i), dtype=np.float64)
            oY[i] = np.matmul(np.matmul(iX0[i].T, Iloc), iX0[i])

            v = ov[i]
            oh = np.matmul(oY[i], v)
            hv = 0.5 * v
            var = np.matmul(dcross(hv), oY[i]) - np.matmul(oY[i], cross(hv))
            B[i] = var + self.icrf(0.5 * oh)
            dJ[i] = np.matmul(cross(v), Sw[i])

        # ---- backward pass: composite inertia / bias + C assembly ----
        oYc = [oY[i].copy() for i in range(NB)]
        Bc = [B[i].copy() for i in range(NB)]
        # per-body dFdv depends on the *composite* inertia/bias, so compute it
        # after the composite has fully accumulated by walking root-ward and
        # storing each body's dFdv once its own composite is final.
        dFdv = [None] * NB
        subtree = {i: sorted(self.robot.get_subtree_by_id(i)) for i in range(NB)}

        def ancestors(i):
            res = []
            j = self.robot.get_parent_id(i)
            while j != -1:
                res.append(j)
                j = self.robot.get_parent_id(j)
            return res

        C = np.zeros((nv, nv), dtype=np.float64)

        for i in range(NB - 1, -1, -1):
            # finalize this body's composite-dependent dFdv now that oYc[i]/Bc[i]
            # have absorbed every child (children have larger ids, processed
            # earlier in this reverse loop).
            dFdv[i] = np.matmul(oYc[i], dJ[i]) + np.matmul(Bc[i], Sw[i])

            # C[v_i, v_d] for d in subtree(i): Sw[i]^T dFdv[d]
            for d in subtree[i]:
                block = np.matmul(Sw[i].T, dFdv[d])
                ai = alpha[i] * alpha[d]
                for a, ia in enumerate(vinds[i]):
                    for b, jb in enumerate(vinds[d]):
                        C[ia, jb] += ai * block[a, b]

            # C[v_i, v_j] for j an ancestor of i
            Ag = np.matmul(oYc[i], Sw[i])
            Mat_tmp = np.matmul(Sw[i].T, Bc[i])
            for j in ancestors(i):
                colblk = np.matmul(Ag.T, dJ[j]) + np.matmul(Mat_tmp, Sw[j])
                aj = alpha[i] * alpha[j]
                for a, ia in enumerate(vinds[i]):
                    for b, jb in enumerate(vinds[j]):
                        C[ia, jb] += aj * colblk[a, b]

            parent = self.robot.get_parent_id(i)
            if parent != -1:
                oYc[parent] = oYc[parent] + oYc[i]
                Bc[parent] = Bc[parent] + Bc[i]

        # The root joint's real motion subspace S (a [v_lin; omega] ->
        # [omega; v_lin] permutation block) already maps the user-facing root
        # velocity ordering into the internal spatial ordering, so unlike CRBA
        # (which assembles the root block with S = I and reorders by hand) no
        # manual root permutation is needed here. Under the pinocchio convention
        # _denormalize_qv_matrix_output is a no-op; it is kept for the legacy
        # convention's v-permutation.
        return self._denormalize_qv_matrix_output(C, row_space="v", col_space="v")
