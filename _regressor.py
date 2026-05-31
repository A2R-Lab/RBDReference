"""Joint-torque regressor numpy reference (D.4 / G.x of the sysID plan).

`tau = Y(q, qd, qdd) . pi`, where `pi = [pi_1; ...; pi_NB]` stacks each link's
10 standard inertial parameters and `Y` is the joint-torque regressor
(nv x 10*NB). Inverse dynamics is exactly affine in each link's spatial inertia,
so `Y = d(tau)/d(pi)` is exact (no finite differences).

Parameter basis (per link, GRiD/URDF convention):
    pi_i = [ m, h(3)=m*c, I_O(6)=[Ixx, Ixy, Ixz, Iyy, Iyz, Izz] ]
with I_O the inertia about the link-frame ORIGIN (the parser's `topLeft`, the
top-left 3x3 of `get_Imat_by_id`). This is the same basis the D.4 runtime path
stores. Pinocchio's `toDynamicParameters` uses `[m, mc(3), [Ixx,Ixy,Iyy,Ixz,
Iyz,Izz]]` and `[linear; angular]` spatial order; the pinocchio backend applies
the constant 10x10 permutation `P` (a 6x6 swap on the inertia block) so the two
bases line up — see `pinocchio_backend.joint_torque_regressor`.

Spatial convention: internal `[angular; linear]` (Featherstone), matching
`RBDReference.rnea` / `cross_operator` / `dual_cross_operator`. The body
regressor `Y_body,i` (6x10) satisfies `f_i = Y_body,i . pi_i` with
`f_i = I_i a_i + v_i x* (I_i v_i)`; the joint regressor is the RNEA backward
force sweep run with a 6x10 right-hand side instead of a 6x1 force.
"""

import numpy as np


# The 10 basis spatial-inertia derivatives dI/dpi_k in GRiD [angular; linear]
# 6x6 order, for pi = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
#   I(pi) = [[ I_O,        skew(h) ],
#            [ skew(h)^T,  m * I3  ]]
# with skew(h) = [[0,-hz,hy],[hz,0,-hx],[-hy,hx,0]].
def _basis_spatial_inertias():
    bases = []

    # m: lower-right 3x3 = I3
    dm = np.zeros((6, 6))
    dm[3, 3] = dm[4, 4] = dm[5, 5] = 1.0
    bases.append(dm)

    # hx, hy, hz: top-right skew(h) and its transpose (bottom-left)
    def h_basis(axis):
        d = np.zeros((6, 6))
        S = np.zeros((3, 3))
        if axis == 0:  # hx -> skew entries at (1,2)=-1,(2,1)=1
            S[1, 2] = -1.0
            S[2, 1] = 1.0
        elif axis == 1:  # hy -> (0,2)=1,(2,0)=-1
            S[0, 2] = 1.0
            S[2, 0] = -1.0
        else:  # hz -> (0,1)=-1,(1,0)=1
            S[0, 1] = -1.0
            S[1, 0] = 1.0
        d[:3, 3:] = S
        d[3:, :3] = S.T
        return d

    bases.append(h_basis(0))
    bases.append(h_basis(1))
    bases.append(h_basis(2))

    # I_O entries [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] -> symmetric top-left 3x3
    def I_basis(r, c):
        d = np.zeros((6, 6))
        d[r, c] = 1.0
        if r != c:
            d[c, r] = 1.0
        return d

    bases.append(I_basis(0, 0))  # Ixx
    bases.append(I_basis(0, 1))  # Ixy
    bases.append(I_basis(0, 2))  # Ixz
    bases.append(I_basis(1, 1))  # Iyy
    bases.append(I_basis(1, 2))  # Iyz
    bases.append(I_basis(2, 2))  # Izz
    return bases


_BASIS_I = _basis_spatial_inertias()


class _RegressorMixin:
    """Joint-torque / body regressor numpy reference."""

    def body_regressor(self, v, a):
        """6x10 body regressor Y_body with f = Y_body . pi for a single link.

        f = I a + v x* (I v), linear in I, so column k is
            dI_k a + crf(v) dI_k v
        with dI_k the k-th basis spatial inertia and crf = dual_cross_operator.
        Spatial order is internal [angular; linear]; param basis is
        [m, h(3), I_O(6)] (URDF symmetric ordering).
        """
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        crf = self.dual_cross_operator(v)  # 6x6
        Y = np.zeros((6, 10), dtype=np.float64)
        for k, dI in enumerate(_BASIS_I):
            Y[:, k] = dI @ a + crf @ (dI @ v)
        return Y

    def joint_torque_regressor(self, q, qd, qdd, GRAVITY=-9.81):
        """Joint-torque regressor Y (nv x 10*NB) with tau = Y . pi.

        Runs the RNEA forward sweep to get each link's (v_i, a_i), builds each
        link's 6x10 body regressor, and back-propagates the 6x10 blocks up the
        tree exactly like the RNEA force backward pass (X^T accumulation +
        S^T projection onto each ancestor DOF). The link param blocks are
        ordered by body id (column block i = link i's 10 params).
        """
        q = self._normalize_q_input(q)
        qd = self._normalize_v_input(qd)
        qdd = self._normalize_v_input(qdd)

        NB = self.robot.get_num_bodies()
        nv = self.robot.get_num_vel()

        # Forward pass: per-link spatial velocity v_i and acceleration a_i.
        v, a, _f = self.rnea_fpass(q, qd, qdd, GRAVITY)

        # Backward force sweep with a 6 x (10*NB) right-hand side per link.
        # `Fblk[i]` carries the spatial-force regressor (wrt EVERY link's param
        # block) accumulated at link i — its own 6x10 block lives in columns
        # [10*i:10*i+10], descendants are propagated up via X^T.
        Fblk = [np.zeros((6, 10 * NB), dtype=np.float64) for _ in range(NB)]
        for i in range(NB):
            Fblk[i][:, 10 * i:10 * i + 10] = self.body_regressor(v[:, i], a[:, i])
        Y = np.zeros((nv, 10 * NB), dtype=np.float64)

        for curr_id in range(NB - 1, -1, -1):
            parent_id = self.robot.get_parent_id(curr_id)
            S = np.asarray(self.robot.get_S_by_id(curr_id), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            inds_f = self.robot.get_joint_index_f(curr_id)
            if not isinstance(inds_f, (list, tuple, np.ndarray)):
                inds_f = [inds_f]
            else:
                inds_f = list(inds_f)
            mimic_scale = self._mimic_multiplier(curr_id)

            # Project the accumulated 6 x (10*NB) onto this link's DOF rows.
            proj = mimic_scale * (S.T @ Fblk[curr_id])  # (ndof_curr, 10*NB)
            for r, fidx in enumerate(inds_f):
                Y[fidx, :] += proj[r, :]

            if parent_id != -1:
                _q = self.robot.q_for_joint(curr_id, q)
                Xmat = np.asarray(
                    self.robot.get_Xmat_Func_by_id(curr_id)(_q), dtype=np.float64
                )
                Fblk[parent_id] = Fblk[parent_id] + Xmat.T @ Fblk[curr_id]

        return self._denormalize_reduced_q_matrix_output(Y, row_space="v")

    def fd_parameter_gradient(self, q, qd, u, GRAVITY=-9.81):
        """Forward-dynamics gradient w.r.t. the inertial params: ∂q̈/∂π.

        From `M(π)·q̈ + c(q,q̇,π) = u` with `u` fixed, differentiating in π gives
        `∂q̈/∂π = − M⁻¹ · Y(q, q̇, q̈_actual)` because `ID(q,q̇,q̈,π) = M q̈ + c`
        is affine in π with Jacobian `Y` at the *actual* acceleration. So:

          1. q̈_actual = forward_dynamics(q, q̇, u)
          2. Y = joint_torque_regressor(q, q̇, q̈_actual)   (nv x 10*NB)
          3. ∂q̈/∂π = − minv(q) · Y                          (nv x 10*NB)

        reuses the existing minv + regressor; no new factorization (mirrors the
        CUDA emit `fd_parameter_gradient` = −Minv·Y). Result is nv x 10*NB.
        """
        qdd = self.forward_dynamics(q, qd, u)
        Y = np.asarray(
            self.joint_torque_regressor(q, qd, qdd, GRAVITY=GRAVITY), dtype=np.float64
        )
        Minv = np.asarray(self.minv(q), dtype=np.float64)
        return -(Minv @ Y)
