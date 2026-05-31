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
matching `RBDReference.rnea`. PE uses the gravity vector g = [0,0,GRAVITY] so
that `PE = -M_total * g . p_com` reproduces Pinocchio's potential energy.
"""

import numpy as np


class _EnergyMixin:
    """Energy / generalized-gravity / Coriolis numpy reference."""

    def generalized_gravity(self, q, GRAVITY=-9.81):
        """g(q) = RNEA(q, 0, 0): the generalized gravity torque (size nv)."""
        n = self.robot.get_num_vel()
        zero = np.zeros(n, dtype=np.float64)
        c, _v, _a, _f = self.rnea(q, zero, zero, GRAVITY=GRAVITY)
        return np.asarray(c, dtype=np.float64).reshape(-1)

    def nonlinear_effects(self, q, qd, GRAVITY=-9.81):
        """c(q, qd) = RNEA(q, qd, 0) = C(q,qd) qd + g(q) (size nv)."""
        n = self.robot.get_num_vel()
        zero = np.zeros(n, dtype=np.float64)
        c, _v, _a, _f = self.rnea(q, qd, zero, GRAVITY=GRAVITY)
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

    def coriolis_matrix(self, q, qd, GRAVITY=-9.81, fd_step=1e-6):
        """Coriolis matrix C(q, qd) with C qd + g = nonlinear effects.

        Uses the Christoffel-symbol construction
            C[i,j] = 1/2 sum_k (dM[i,j,k] + dM[i,k,j] - dM[k,j,i]) qd[k]
        with dM[:,:,k] = d M / d q_k formed by a central difference of CRBA.
        This reproduces Pinocchio's `computeCoriolisMatrix` entrywise (verified)
        — not just the product C qd. Fixed-base only (the Christoffel q-tangent
        FD below uses scalar q perturbations).
        """
        if self.robot.floating_base:
            raise NotImplementedError(
                "coriolis_matrix reference is implemented for fixed-base only. "
                "Floating-base Coriolis is intentionally deferred: Pinocchio's "
                "computeCoriolisMatrix returns an algorithm-specific matrix whose "
                "symmetric part is exactly 1/2 d/dt M (reproducible by a "
                "Lie-tangent CRBA finite difference) but whose SKEW part follows "
                "Pinocchio's internal spatial Bcrb recursion and is NOT the "
                "tangent-space Christoffel skew (verified: the naive Christoffel "
                "construction mismatches by full magnitude on go2/iiwa14 floating). "
                "A matching reference therefore requires porting Pinocchio's exact "
                "body-frame Coriolis recursion, which is out of scope here; the "
                "floating-base Coriolis equivalence test stays skipped (the pin "
                "oracle is present for when the port lands)."
            )
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        nv = self.robot.get_num_vel()
        dM = np.zeros((nv, nv, nv), dtype=np.float64)
        for k in range(nv):
            dq = np.zeros(nv, dtype=np.float64)
            dq[k] = fd_step
            Mp = np.asarray(self.crba(q + dq), dtype=np.float64)
            Mm = np.asarray(self.crba(q - dq), dtype=np.float64)
            dM[:, :, k] = (Mp - Mm) / (2.0 * fd_step)
        C = np.zeros((nv, nv), dtype=np.float64)
        for i in range(nv):
            for j in range(nv):
                C[i, j] = 0.5 * float(
                    np.sum((dM[i, j, :] + dM[i, :, j] - dM[:, j, i]) * qd)
                )
        return C
