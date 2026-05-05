import numpy as np
import copy
np.set_printoptions(precision=4, suppress=True, linewidth=100)

class RBDReference:
    def __init__(self, robotObj):
        """Initialize RBDReference with a robot object.

        Parameters
        ----------
        robotObj : URDFparser
            An instance of the URDFparser class.

        Returns
        -------
        None : None
            None
        """
        self.robot = robotObj # instance of Robot Object class created by URDFparser

    def _normalize_q_input(self, q):
        return self.robot.normalize_floating_base_q_input(q)

    def _normalize_v_input(self, vec):
        return self.robot.normalize_floating_base_v_input(vec)

    def _denormalize_v_output(self, vec):
        return self.robot.denormalize_floating_base_v_output(vec)

    def _permute_matrix_prefix(self, matrix, permutation, axis):
        if permutation is None:
            return matrix
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        prefix = len(permutation)
        if axis == 0:
            matrix[:prefix, :] = matrix[permutation, :]
        else:
            matrix[:, :prefix] = matrix[:, permutation]
        return matrix

    def _denormalize_qv_matrix_output(self, matrix, row_space=None, col_space=None):
        matrix = np.asarray(matrix, dtype=np.float64).copy()
        if row_space == "v":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_v_permutation_from_internal(),
                axis=0,
            )
        elif row_space == "q":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_q_output_permutation_from_internal(),
                axis=0,
            )
        if col_space == "v":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_v_permutation_from_internal(),
                axis=1,
            )
        elif col_space == "q":
            matrix = self._permute_matrix_prefix(
                matrix,
                self.robot.get_floating_base_q_output_permutation_from_internal(),
                axis=1,
            )
        return matrix

    def _denormalize_reduced_q_matrix_output(self, matrix, row_space=None):
        return self._denormalize_qv_matrix_output(matrix, row_space=row_space, col_space=None)

    def _denormalize_rnea_grad_output(self, dc_dq, dc_dqd):
        return np.hstack(
            (
                self._denormalize_reduced_q_matrix_output(dc_dq, row_space="v"),
                self._denormalize_qv_matrix_output(dc_dqd, row_space="v", col_space="v"),
            )
        )

    def cross_operator(self, v):
        """Compute the 6x6 spatial cross product matrix for a velocity vector.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_cross : numpy.ndarray
            6x6 spatial cross product matrix.
        """
        # for any vector v, computes the operator v x 
        # vec x = [wx   0]
        #         [vox wx]
        #(crm in spatial_v2_extended)
        v_cross = np.array([0, -v[2], v[1], 0, 0, 0,
                            v[2], 0, -v[0], 0, 0, 0,
                            -v[1], v[0], 0, 0, 0, 0,
                            0, -v[5], v[4], 0, -v[2], v[1], 
                            v[5], 0, -v[3], v[2], 0, -v[0],
                            -v[4], v[3], 0, -v[1], v[0], 0]
                          ).reshape(6,6)
        return(v_cross)
    
    def dual_cross_operator(self, v):
        """Compute the 6x6 spatial dual cross product matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_dual_cross : numpy.ndarray
            6x6 spatial dual cross product matrix.
        """
        #(crf in in spatial_v2_extended)
        return(-1 * self.cross_operator(v).T)
    
    def dot_matrix(self, I, v):
        """Compute the time derivative of the spatial inertia matrix.

        Parameters
        ----------
        I : numpy.ndarray
            6x6 spatial inertia matrix.
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        I_dot : numpy.ndarray
            6x6 time derivative of spatial inertia.
        """
        A =  self.dual_cross_operator(v) @ I - I @ self.cross_operator(v)
        scale_factor = 10**-15
        A = A / scale_factor
        return self.dual_cross_operator(v) @ I - I @ self.cross_operator(v)
    
    def icrf(self, v):
        """Compute the inverse of the force (dual) cross operator.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        res : numpy.ndarray
            6x6 inverse force cross operator matrix.
        """
        #helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
        # inverse of the force(dual) cross operator
        # v crf f = f icrf v
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asmatrix(res)
    
    def factor_functions(self, I, v, number=3):
        """Helper functions for factorization in IDSVA and RNEA gradient.

        Parameters
        ----------
        I : numpy.ndarray
            6x6 spatial inertia matrix.
        v : numpy.ndarray
            6D spatial velocity vector.
        number : int
            Type of factorization to perform.

        Returns
        -------
        B : numpy.ndarray
            The resulting factorized matrix.
        """
        # helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
        if number == 1:
            B = self.dual_cross_operator(v) * I
        elif number == 2:
            B = self.icrf(np.matmul(I,v)) - I * self.cross_operator(v)
        else:
            B = 1/2 * (np.matmul(self.dual_cross_operator(v),I) + self.icrf(np.matmul(I,v)) - np.matmul(I, self.cross_operator(v)))

        return B

    def _mxS(self, S, vec, alpha=1.0):
        """Compute the product of a cross operator matrix and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        vS : numpy.ndarray
            The 6xN matrix product.
        """
        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        return np.squeeze(np.array((alpha * np.dot(self.cross_operator(vec), S)))) # added np.squeeze and np.array

    def mxS(self, S, vec):
        """Compute the cross product of a spatial vector and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        vS : numpy.ndarray
            The 6xN matrix product.
        """
        result = np.zeros((6))
        if not S[0] == 0:
            result += self.mx1(vec, S[0])
        if not S[1] == 0:
            result += self.mx2(vec, S[1])
        if not S[2] == 0:
            result += self.mx3(vec, S[2])
        if not S[3] == 0:
            result += self.mx4(vec, S[3])
        if not S[4] == 0:
            result += self.mx5(vec, S[4])
        if not S[5] == 0:
            result += self.mx6(vec, S[5])
        return result

    def mx1(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 1).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[1] = vec[2] * alpha
            vecX[2] = -vec[1] * alpha
            vecX[4] = vec[5] * alpha
            vecX[5] = -vec[4] * alpha
        except:
            vecX[1] = vec[0, 2] * alpha
            vecX[2] = -vec[0, 1] * alpha
            vecX[4] = vec[0, 5] * alpha
            vecX[5] = -vec[0, 4] * alpha
        return vecX

    def mx2(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 2).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[0] = -vec[2] * alpha
            vecX[2] = vec[0] * alpha
            vecX[3] = -vec[5] * alpha
            vecX[5] = vec[3] * alpha
        except:
            vecX[0] = -vec[0, 2] * alpha
            vecX[2] = vec[0, 0] * alpha
            vecX[3] = -vec[0, 5] * alpha
            vecX[5] = vec[0, 3] * alpha
        return vecX

    def mx3(self, vec, alpha=1.0):
        """Compute product of cross operator and a vector (Variant 3).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        vc : numpy.ndarray
            6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[0] = vec[1] * alpha
            vecX[1] = -vec[0] * alpha
            vecX[3] = vec[4] * alpha
            vecX[4] = -vec[3] * alpha
        except:
            vecX[0] = vec[0, 1] * alpha
            vecX[1] = -vec[0, 0] * alpha
            vecX[3] = vec[0, 4] * alpha
            vecX[4] = -vec[0, 3] * alpha
        return vecX

    def mx4(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 4).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[4] = vec[2] * alpha
            vecX[5] = -vec[1] * alpha
        except:
            vecX[4] = vec[0, 2] * alpha
            vecX[5] = -vec[0, 1] * alpha
        return vecX

    def mx5(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 5).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[3] = -vec[2] * alpha
            vecX[5] = vec[0] * alpha
        except:
            vecX[3] = -vec[0, 2] * alpha
            vecX[5] = vec[0, 0] * alpha
        return vecX

    def mx6(self, vec, alpha=1.0):
        """Compute product of dual cross operator and a force vector (Variant 6).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        f : numpy.ndarray
            6D spatial force vector.

        Returns
        -------
        res : numpy.ndarray
            Resulting 6D spatial force vector.
        """
        vecX = np.zeros((6))
        try:
            vecX[3] = vec[1] * alpha
            vecX[4] = -vec[0] * alpha
        except:
            vecX[3] = vec[0, 1] * alpha
            vecX[4] = -vec[0, 0] * alpha
        return vecX

    def fxv(self, fxVec, timesVec):
        """Compute the spatial cross product between two spatial vectors.

        Parameters
        ----------
        v1 : numpy.ndarray
            First 6D spatial vector.
        v2 : numpy.ndarray
            Second 6D spatial vector.

        Returns
        -------
        res : numpy.ndarray
            6D spatial vector cross product.
        """
        # Fx(fxVec)*timesVec
        #   0  -v(2)  v(1)    0  -v(5)  v(4)
        # v(2)    0  -v(0)  v(5)    0  -v(3)
        # -v(1)  v(0)    0  -v(4)  v(3)    0
        #   0     0     0     0  -v(2)  v(1)
        #   0     0     0   v(2)    0  -v(0)
        #   0     0     0  -v(1)  v(0)    0
        result = np.zeros((6))
        result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
        result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
        result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
        result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
        result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
        result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
        return result

    def fxS(self, S, vec, alpha=1.0):
        """Compute the spatial cross product between a vector and a motion subspace matrix.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        S : numpy.ndarray
            6xN motion subspace matrix.

        Returns
        -------
        res : numpy.ndarray
            6xN spatial cross product matrix.
        """
        # force spatial cross product with motion subspace
        return np.squeeze(
            np.array(alpha * np.matmul(self.dual_cross_operator(S), vec))
        )

    def vxIv(self, vec, Imat):
        """Compute the spatial force vector v x (I * v).

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.
        I : numpy.ndarray
            6x6 spatial inertia matrix.

        Returns
        -------
        res : numpy.ndarray
            6D spatial force vector.
        """
        # necessary component in differentiating Iv (product rule).
        # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
        # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
        temp = np.matmul(Imat, vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec
    

    """
    End Effector Joint Selector

    Helper function to select specific end-effector joints for the end-effector position and gradient functions. If no joints specified then defaults to all leaf joints.
    """
    def select_end_effector_joints(self, ee_joint_names):
        """Identify the joint indices along the chain to the specified end effector.

        Parameters
        ----------
        ee_id : int
            Index of the end effector.

        Returns
        -------
        q_inds : list
            List of joint indices from root to end effector.

        Raises
        ------
        ValueError if ee_id is not a valid end-effector index.
        """
        # deterimine the target joints for the kinematic calcs
        ee_jids = []
        fixed_jids = []
        # if no joints specified then do all leaf joints
        if ee_joint_names is None:
            ee_jids = self.robot.get_leaf_nodes()
        # else search for specific end-effector joints
        else:
            if isinstance(ee_joint_names, str):
                ee_joint_names = [ee_joint_names]
            ee_jids = []
            fixed_jids = []
            for name in ee_joint_names:
                joint = self.robot.get_joint_by_name(name)
                if joint is not None:
                    ee_jids.append(joint.get_id())
                else:
                    fjoint = self.robot.get_fixed_joint_by_name(name)
                    if fjoint is None:
                        raise ValueError("Could not find joint or fixed joint named: " + name)
                    fixed_jids.append(fjoint.get_id())
        return ee_jids, fixed_jids

    """
    End Effector Posiitons

    offests is an array of np matricies of the form (offset_x, offset_y, offset_z, 1)
    
    TODO: Add and test floating base support.
    """

    def end_effector_pose(self, q, ee_joint_names = None, ee_offsets = [np.matrix([[0,0,0,1]])]):
        """Compute the 4x4 homogeneous transformation matrix of the end effector.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        T : numpy.ndarray
            4x4 homogeneous transformation matrix.
        """
        # chain up the transforms (version 1 for starting from the root)
        def forwardChain(self, jid, q):
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            # then chain them up
            Xmat_hom = np.eye(4)
            for ind in jidChain:
                inds_q = self.robot.get_joint_index_q(ind)
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[inds_q])
                Xmat_hom = np.matmul(Xmat_hom,currX)
            return Xmat_hom

        # chain up the transforms (version 2 for starting from the leaf)
        def backwardChain(self, jid, q, finalXmat_hom = np.eye(4)):
            currId = jid
            Xmat_hom = finalXmat_hom
            while(currId != -1):
                inds_q = self.robot.get_joint_index_q(currId)
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[inds_q])
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)
            return Xmat_hom

        # Extract the end-effector position with the given offset(s)
        # TODO handle different offsets for different branches
        def eePos_from_Xmat_hom(Xmat_hom, ee_offsets):
            # xyz position is easy
            eePos_xyz1 = Xmat_hom * ee_offsets[0].transpose()

            # roll pitch yaw is a bit more difficult
            eePos_roll = np.arctan2(Xmat_hom[2,1],Xmat_hom[2,2])
            pitch_temp = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
            eePos_pitch = np.arctan2(-Xmat_hom[2,0],pitch_temp)
            eePos_yaw = np.arctan2(Xmat_hom[1,0],Xmat_hom[0,0])
            eePos_rpy = np.matrix([[eePos_roll,eePos_pitch,eePos_yaw]])

            # then stack it up!
            eePos = np.vstack((eePos_xyz1[:3,:],eePos_rpy.transpose()))
            return eePos

        # do the actual computations
        eePos_arr = []
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        for jid in ee_jids:
            # Xmat_hom = forwardChain(self, jid, q)                
            Xmat_hom = backwardChain(self, jid, q)
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offsets)
            eePos_arr.append(eePos)
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            if fj.parent_name == -1:
                Xmat_hom = fj.get_transformation_matrix_hom()
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                Xmat_hom = backwardChain(self, parent.get_id(), q, fj.get_transformation_matrix_hom())
            eePos = eePos_from_Xmat_hom(Xmat_hom, ee_offsets)
            eePos_arr.append(eePos)
        return eePos_arr

    """
    End Effectors Pose Gradients
    """
    def equals_or_hstack(self, obj, col):
        """Concatenate or assign vectors depending on initialization state.

        Parameters
        ----------
        target : numpy.ndarray or None
            Target array to concatenate to.
        source : numpy.ndarray
            Array to append.

        Returns
        -------
        res : numpy.ndarray
            The resulting concatenated array.
        """
        if obj is None:
            obj = col
        else:
            obj = np.hstack((obj,col))
        return obj

    def _normalize_kinematics_q(self, q):
        """Ensure the joint position vector corresponds to the robot degrees of freedom.

        Parameters
        ----------
        q : numpy.ndarray
            Vector of joint positions.

        Returns
        -------
        q : numpy.ndarray
            Normalized joint position vector.

        Raises
        ------
        ValueError if q length does not match robot model degrees of freedom.
        """
         # keep the user-facing floating-base convention in one place:
        # q = [x, y, z, qx, qy, qz, qw, ...]
        #
        # for the analytic kinematics helpers below we assume the quaternion
        # block is already normalized, so make that explicit here rather than
        # scattering the normalization logic through each helper.
        q = np.asarray(q, dtype=np.float64).copy()
        if self.robot.floating_base and self.robot.using_quaternion:
            quat = q[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm == 0.0:
                raise ValueError("Floating-base quaternion norm was zero during kinematics normalization.")
            q[3:7] = quat / quat_norm
        return q

    def end_effector_pose_gradient(self, q, ee_joint_names = None, ee_offsets = [np.matrix([[0,0,0,1]])]):
        """Compute the Jacobian (gradient) of the end-effector pose.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        J : numpy.ndarray
            Gradient of the end-effector pose.
        """
        q = self._normalize_kinematics_q(q)
        n = len(q)

        # local helper for handling scalar joints and the floating-base root
        def qinds_to_list(inds):
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def qeval_arg(jid, q):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            q_block = np.asarray(q[inds], dtype=np.float64)
            if q_block.size == 1:
                return float(q_block[0])
            return q_block

        def get_dlocal_info(jid, dind):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            if dind not in inds:
                return None
            return inds.index(dind)

        # chain up the transforms (version 1 for starting from the root)
        def dforward_chain(self, jidChain, dind, q):
            Xmat_hom = np.eye(4)
            dXmat_hom = np.eye(4)
            for ind in jidChain:
                dlocal_ind = get_dlocal_info(ind, dind)
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                if dlocal_ind is not None: # use derivative
                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                    dXmat_hom = np.matmul(dXmat_hom,dcurrX)
                else: # use normal transform
                    dXmat_hom = np.matmul(dXmat_hom,currX)
                Xmat_hom = np.matmul(Xmat_hom,currX)
            return Xmat_hom, dXmat_hom

        # chain up the transforms (version 2 for starting from the leaf)
        def dbackward_chain(self, jid, dind, q, finalXmat_hom = np.eye(4)):
            currId = jid
            Xmat_hom = finalXmat_hom
            dXmat_hom = finalXmat_hom
            while(currId != -1):
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                dcurrX = currX
                dlocal_ind = get_dlocal_info(currId, dind)
                if dlocal_ind is not None:
                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                dXmat_hom = np.matmul(dcurrX,dXmat_hom)
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)
            return Xmat_hom, dXmat_hom
        
        # Extract the end-effector position with the given offset(s)
        def deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets):
            # Then extract the end-effector position with the given offset(s)
            # TODO handle different offsets for different branches

            # xyz position is easy
            deePos_xyz1 = dXmat_hom * ee_offsets[0].transpose()

            # roll pitch yaw is a bit more difficult
            # note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
            def darctan2(y,x,y_prime,x_prime):
                return (-x_prime*y + x*y_prime)/(x*x + y*y)
            # in other words for each atan2 we are plugging in Xmat_hom and dXmat_hom at 
            # those indicies into the spots as specified by that equation.
            # also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
            deePos_roll = darctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom[2,1],dXmat_hom[2,2])
            pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
            dpitch_sqrt_term = (Xmat_hom[2,2]*dXmat_hom[2,2] + Xmat_hom[2,1]*dXmat_hom[2,1])/pitch_sqrt_term # note canceled out the 2 in the numer and denom
            deePos_pitch = darctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom[2,0],dpitch_sqrt_term)
            deePos_yaw = darctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom[1,0],dXmat_hom[0,0])
            deePos_rpy = np.matrix([[deePos_roll,deePos_pitch,deePos_yaw]])

            # then stack it up!
            deePos_col = np.vstack((deePos_xyz1[:3,:],deePos_rpy.transpose()))
            return deePos_col

        # Then compute the gradients for each end-effector requested
        # -> For each branch chain up the transformations across all possible derivatives
        deePos_arr = []
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        # First for the standard joints
        for jid in ee_jids:
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            jidChainQInds = []
            for ind in jidChain:
                jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))
            # then compute the gradients
            deePos = None
            for dind in range(n):
                # Note: if not in branch then 0
                if dind not in jidChainQInds:
                    deePos_col = np.zeros((6,1))
                    deePos = self.equals_or_hstack(deePos,deePos_col)
                else:
                    # chain up the transforms (2 options)
                    # Xmat_hom, dXmat_hom = dforward_chain(self, jidChain, dind, q)
                    Xmat_hom, dXmat_hom = dbackward_chain(self, jid, dind, q)
                    deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets)
                    deePos = self.equals_or_hstack(deePos,deePos_col)
            deePos_arr.append(deePos)
        # Then for the fixed joints
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            if fj.parent_name == -1:
                deePos = None
                Xmat_hom = fj.get_transformation_matrix_hom()
                for _dind in range(n):
                    deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, np.zeros((4,4)), ee_offsets)
                    deePos = self.equals_or_hstack(deePos, deePos_col)
            else:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                # first get the joints in the chain
                jidChain = sorted(self.robot.get_ancestors_by_id(parent.get_id()))
                jidChain.append(parent.get_id())
                jidChainQInds = []
                for ind in jidChain:
                    jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))
                # then compute the gradients
                deePos = None
                for dind in range(n):
                    # Note: if not in branch then 0
                    if dind not in jidChainQInds:
                        deePos_col = np.zeros((6,1))
                        deePos = self.equals_or_hstack(deePos,deePos_col)
                    else:
                        Xmat_hom, dXmat_hom = dbackward_chain(self, parent.get_id(), dind, q, fj.get_transformation_matrix_hom())
                        deePos_col = deePos_col_from_Xmat_hom(Xmat_hom, dXmat_hom, ee_offsets)
                        deePos = self.equals_or_hstack(deePos,deePos_col)
            deePos_arr.append(deePos)
        return deePos_arr

    """
    End Effector Pose Hessian
    """
    def end_effector_pose_hessian(self, q, offsets = [np.matrix([[0,0,0,1]])], ee_joint_names = None):
        """Compute the Hessian of the end-effector pose.

        Parameters
        ----------
        q : numpy.ndarray
            N-element vector of joint positions.
        ee_id : int
            Index of the end effector.

        Returns
        -------
        H : numpy.ndarray
            Hessian of the end-effector pose.
        """
        q = self._normalize_kinematics_q(q)
        n = len(q)

        # local helper for handling scalar joints and the floating-base root
        def qinds_to_list(inds):
            if isinstance(inds, (list, tuple, np.ndarray)):
                return list(inds)
            return [inds]

        def qeval_arg(jid, q):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            q_block = np.asarray(q[inds], dtype=np.float64)
            if q_block.size == 1:
                return float(q_block[0])
            return q_block

        def get_dlocal_info(jid, dind):
            inds = qinds_to_list(self.robot.get_joint_index_q(jid))
            if dind not in inds:
                return None
            return inds.index(dind)

        # Determine which targets to compute
        ee_jids, fixed_jids = self.select_end_effector_joints(ee_joint_names)
        if ee_joint_names is None:
            ee_jids = self.robot.get_leaf_nodes()
            fixed_jids = []

        # For each branch chain up the transformations across all possible derivatives
        # Note: if not in branch then 0
        d2eePos_arr = []
        for jid in ee_jids: # can be done in parallel
            
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            jidChainQInds = []
            for ind in jidChain:
                jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))

            # first chain up the 1st derivative transforms
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                # n+1 for standard as well (noting that n+1 will never trigger derivative)
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                for ind in jidChain:
                    currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                    dlocal_ind = get_dlocal_info(ind, dind)
                    if dlocal_ind is not None: # use derivative
                        dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],dcurrX)
                    else: # use normal transform
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],currX)

            # chain up the 1st derivative transforms (version 2 for starting from the leaf)
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                currId = jid
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                while(currId != -1):
                    dlocal_ind = get_dlocal_info(currId, dind)
                    if dlocal_ind is not None:
                        dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                    else:
                        dcurrX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                    dXmat_hom_arr[:,:,dind] = np.matmul(dcurrX,dXmat_hom_arr[:,:,dind])
                    currId = self.robot.get_parent_id(currId)

            # then chain up the 2nd derivative transforms
            d2Xmat_hom_arr = np.zeros((4,4,n,n))
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    for ind in jidChain:
                        dlocal_ind_i = get_dlocal_info(ind, dind_i)
                        dlocal_ind_j = get_dlocal_info(ind, dind_j)
                        if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                            if (dlocal_ind_i is not None) and (dlocal_ind_j is not None): # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(ind, dlocal_ind_i, dlocal_ind_j)(qeval_arg(ind, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],d2currX)
                            else: # use first derivative values
                                dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(ind, dlocal_ind)(qeval_arg(ind, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],dcurrX)
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(ind)(qeval_arg(ind, q))
                            d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],currX)

            # then chain up the 2nd derivative transforms (version 2 - backward)
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    currId = jid
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    while(currId != -1):
                        dlocal_ind_i = get_dlocal_info(currId, dind_i)
                        dlocal_ind_j = get_dlocal_info(currId, dind_j)
                        if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                            if (dlocal_ind_i is not None) and (dlocal_ind_j is not None): # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(currId, dlocal_ind_i, dlocal_ind_j)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            else: # use first derivative values
                                dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(dcurrX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                            d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                        currId = self.robot.get_parent_id(currId)

            # Then extract the end-effector position with the given offset(s)
            # TODO handle different offsets for different branches
            d2eePos = np.zeros((6,n,n))
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel

                    # point to the correct transforms
                    Xmat_hom = dXmat_hom_arr[:,:,n]
                    dXmat_hom_i = dXmat_hom_arr[:,:,dind_i]
                    dXmat_hom_j = dXmat_hom_arr[:,:,dind_j]
                    d2Xmat_hom = d2Xmat_hom_arr[:,:,dind_i,dind_j]

                    # Note: if not in branch then 0
                    if (dind_i not in jidChainQInds) or (dind_j not in jidChainQInds):
                        d2eePos[:,dind_i,dind_j] = np.zeros((6,))
                    
                    else:
                        # xyz position is easy
                        d2eePos_xyz1 = d2Xmat_hom * offsets[0].transpose()

                        # roll pitch yaw is a bit more difficult
                        # note: d/dz of arctan2(y(z),x(z)) = [-x'(z)y(z)+x(z)y'(z)]/[(x(z)^2 + y(z)^2)]
                        def darctan2(y,x,y_prime,x_prime):
                            return (-x_prime*y + x*y_prime)/(x*x + y*y)
                        # then another chain / quotient rule
                        def quotient_rule(top,bottom,dtop,dbottom):
                            return (bottom*dtop - top*dbottom) / (bottom*bottom)
                        def d2arctan2(y,x,y_prime_i,x_prime_i,y_prime_j,x_prime_j,y_prime_prime,x_prime_prime,i,j):
                            top = -x_prime_i*y + x*y_prime_i
                            dtop = -x_prime_prime*y + x*y_prime_prime
                            if (i != j):
                                dtop = dtop + (-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
                            bottom = x*x + y*y
                            dbottom = 2*x*x_prime_j + 2*y*y_prime_j
                            return quotient_rule(top,bottom,dtop,dbottom)
                        # in other words for each atan2 we are plugging in Xmat_hom and dXmat_hom at 
                        # those indicies into the spots as specified by that equation.
                        # also note that d/dz of sqrt(f(z)) = f'(z)/2sqrt(f(z))
                        d2eePos_roll = d2arctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom_i[2,1],dXmat_hom_i[2,2], \
                                        dXmat_hom_j[2,1],dXmat_hom_j[2,2],d2Xmat_hom[2,1],d2Xmat_hom[2,2],dind_i,dind_j)

                        pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
                        dpitch_sqrt_term_i_top = Xmat_hom[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,1]*dXmat_hom_i[2,1]
                        dpitch_sqrt_term_i = dpitch_sqrt_term_i_top/pitch_sqrt_term # note canceled out the 2 in the numer and denom
                        dpitch_sqrt_term_j_top = Xmat_hom[2,2]*dXmat_hom_j[2,2] + Xmat_hom[2,1]*dXmat_hom_j[2,1]
                        dpitch_sqrt_term_j = dpitch_sqrt_term_j_top/pitch_sqrt_term # note canceled out the 2 in the numer and denom
                        # d2pitch_sqrt_term is quotient rule of dpitch_sqrt_term_i
                        # top = dpitch_sqrt_term_i, bottom = pitch_sqrt_term, dtop = dpitch_sqrt_term_i_top_dj, dbottom = dpitch_sqrt_term_j
                        dpitch_sqrt_term_i_top_dj = dXmat_hom_j[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,2]*d2Xmat_hom[2,2] + \
                                                    dXmat_hom_j[2,1]*dXmat_hom_i[2,1] + Xmat_hom[2,1]*d2Xmat_hom[2,1]
                        d2pitch_sqrt_term = quotient_rule(dpitch_sqrt_term_i,pitch_sqrt_term,dpitch_sqrt_term_i_top_dj,dpitch_sqrt_term_j)
                        d2eePos_pitch = d2arctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom_i[2,0],dpitch_sqrt_term_i, \
                                         -dXmat_hom_j[2,0],dpitch_sqrt_term_j,-d2Xmat_hom[2,0],d2pitch_sqrt_term,dind_i,dind_j)
                        d2eePos_yaw = d2arctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom_i[1,0],dXmat_hom_i[0,0], \
                                        dXmat_hom_j[1,0],dXmat_hom_j[0,0],d2Xmat_hom[1,0],d2Xmat_hom[0,0],dind_i,dind_j)
                        d2eePos_rpy = np.matrix([[d2eePos_roll,d2eePos_pitch,d2eePos_yaw]])

                        # then stack it up!
                        d2eePos_col = np.vstack((d2eePos_xyz1[:3,:],d2eePos_rpy.transpose()))
                        d2eePos[:,dind_i,dind_j] = d2eePos_col.reshape((6,))

            d2eePos_arr.append(d2eePos)

        # Then for the fixed joints
        for fjid in fixed_jids:
            fj = self.robot.get_fixed_joint_by_id(fjid)
            d2eePos = np.zeros((6,n,n))
            if fj.parent_name != -1:
                parent = self.robot.get_joint_by_name(fj.parent_name)
                # first get the joints in the chain
                jidChain = sorted(self.robot.get_ancestors_by_id(parent.get_id()))
                jidChain.append(parent.get_id())
                jidChainQInds = []
                for ind in jidChain:
                    jidChainQInds += qinds_to_list(self.robot.get_joint_index_q(ind))

                # first chain up the 1st derivative transforms
                dXmat_hom_arr = np.zeros((4,4,n+1))
                for dind in range(n+1):
                    dXmat_hom_arr[:,:,dind] = fj.get_transformation_matrix_hom() if dind == n else fj.get_transformation_matrix_hom()
                    currId = parent.get_id()
                    while(currId != -1):
                        dlocal_ind = get_dlocal_info(currId, dind)
                        if dlocal_ind is not None:
                            dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                        else:
                            dcurrX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                        dXmat_hom_arr[:,:,dind] = np.matmul(dcurrX,dXmat_hom_arr[:,:,dind])
                        currId = self.robot.get_parent_id(currId)

                # then chain up the 2nd derivative transforms (version 2 - backward)
                d2Xmat_hom_arr = np.zeros((4,4,n,n))
                for dind_i in range(n):
                    for dind_j in range(n):
                        d2Xmat_hom_arr[:,:,dind_i,dind_j] = fj.get_transformation_matrix_hom()
                        currId = parent.get_id()
                        while(currId != -1):
                            dlocal_ind_i = get_dlocal_info(currId, dind_i)
                            dlocal_ind_j = get_dlocal_info(currId, dind_j)
                            if (dlocal_ind_i is not None) or (dlocal_ind_j is not None):
                                if (dlocal_ind_i is not None) and (dlocal_ind_j is not None):
                                    d2currX = self.robot.get_d2Xmat_hom_local_Func_by_id(currId, dlocal_ind_i, dlocal_ind_j)(qeval_arg(currId, q))
                                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                                else:
                                    dlocal_ind = dlocal_ind_i if dlocal_ind_i is not None else dlocal_ind_j
                                    dcurrX = self.robot.get_dXmat_hom_local_Func_by_id(currId, dlocal_ind)(qeval_arg(currId, q))
                                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(dcurrX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            else:
                                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(qeval_arg(currId, q))
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            currId = self.robot.get_parent_id(currId)

                for dind_i in range(n):
                    for dind_j in range(n):
                        Xmat_hom = dXmat_hom_arr[:,:,n]
                        dXmat_hom_i = dXmat_hom_arr[:,:,dind_i]
                        dXmat_hom_j = dXmat_hom_arr[:,:,dind_j]
                        d2Xmat_hom = d2Xmat_hom_arr[:,:,dind_i,dind_j]

                        if (dind_i not in jidChainQInds) or (dind_j not in jidChainQInds):
                            d2eePos[:,dind_i,dind_j] = np.zeros((6,))
                        else:
                            d2eePos_xyz1 = d2Xmat_hom * offsets[0].transpose()

                            def quotient_rule(top,bottom,dtop,dbottom):
                                return (bottom*dtop - top*dbottom) / (bottom*bottom)
                            def d2arctan2(y,x,y_prime_i,x_prime_i,y_prime_j,x_prime_j,y_prime_prime,x_prime_prime,i,j):
                                top = -x_prime_i*y + x*y_prime_i
                                dtop = -x_prime_prime*y + x*y_prime_prime
                                if (i != j):
                                    dtop = dtop + (-x_prime_i*y_prime_j + x_prime_j*y_prime_i)
                                bottom = x*x + y*y
                                dbottom = 2*x*x_prime_j + 2*y*y_prime_j
                                return quotient_rule(top,bottom,dtop,dbottom)

                            d2eePos_roll = d2arctan2(Xmat_hom[2,1],Xmat_hom[2,2],dXmat_hom_i[2,1],dXmat_hom_i[2,2], \
                                            dXmat_hom_j[2,1],dXmat_hom_j[2,2],d2Xmat_hom[2,1],d2Xmat_hom[2,2],dind_i,dind_j)

                            pitch_sqrt_term = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
                            dpitch_sqrt_term_i_top = Xmat_hom[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,1]*dXmat_hom_i[2,1]
                            dpitch_sqrt_term_i = dpitch_sqrt_term_i_top/pitch_sqrt_term
                            dpitch_sqrt_term_j_top = Xmat_hom[2,2]*dXmat_hom_j[2,2] + Xmat_hom[2,1]*dXmat_hom_j[2,1]
                            dpitch_sqrt_term_j = dpitch_sqrt_term_j_top/pitch_sqrt_term
                            dpitch_sqrt_term_i_top_dj = dXmat_hom_j[2,2]*dXmat_hom_i[2,2] + Xmat_hom[2,2]*d2Xmat_hom[2,2] + \
                                                        dXmat_hom_j[2,1]*dXmat_hom_i[2,1] + Xmat_hom[2,1]*d2Xmat_hom[2,1]
                            d2pitch_sqrt_term = quotient_rule(dpitch_sqrt_term_i,pitch_sqrt_term,dpitch_sqrt_term_i_top_dj,dpitch_sqrt_term_j)
                            d2eePos_pitch = d2arctan2(-Xmat_hom[2,0],pitch_sqrt_term,-dXmat_hom_i[2,0],dpitch_sqrt_term_i, \
                                             -dXmat_hom_j[2,0],dpitch_sqrt_term_j,-d2Xmat_hom[2,0],d2pitch_sqrt_term,dind_i,dind_j)
                            d2eePos_yaw = d2arctan2(Xmat_hom[1,0],Xmat_hom[0,0],dXmat_hom_i[1,0],dXmat_hom_i[0,0], \
                                            dXmat_hom_j[1,0],dXmat_hom_j[0,0],d2Xmat_hom[1,0],d2Xmat_hom[0,0],dind_i,dind_j)
                            d2eePos_rpy = np.matrix([[d2eePos_roll,d2eePos_pitch,d2eePos_yaw]])

                            d2eePos_col = np.vstack((d2eePos_xyz1[:3,:],d2eePos_rpy.transpose()))
                            d2eePos[:,dind_i,dind_j] = d2eePos_col.reshape((6,))

            d2eePos_arr.append(d2eePos)
        return d2eePos_arr
    
    def apply_external_forces(self, q, f_in, f_ext):
        """Distribute externally applied forces to the internal rigid body force array.

        Parameters
        ----------
        f_ext_total : numpy.ndarray
            Array of external forces applied to the robot.

        Returns
        -------
        f_ext : numpy.ndarray
            6N-element array of spatial forces per link.
        """
        f_out = f_in
        NB = self.robot.get_num_bodies()
        if len(f_ext) > 0:
            for curr_id in range(NB):
                parent_id = self.robot.get_parent_id(curr_id)
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                if parent_id == -1:
                    Xa = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                else:
                    Xa = np.matmul(self.robot.get_Xmat_Func_by_id(curr_id)(curr_id),Xa) 
                if len(f_ext[curr_id]) > 1:
                    f_out[curr_id] -= np.matmul(np.linalg.inv(Xa.T), f_ext[curr_id])
        return f_out

    def rnea_fpass(self, q, qd, qdd=None, GRAVITY=-9.81):
        """Perform the forward pass of the Recursive Newton-Euler Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (v, a, f) : tuple
            Spatial velocities, accelerations, and forces of each link.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        v = np.zeros((6, NB))
        a = np.zeros((6, NB))
        f = np.zeros((6, NB))
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY  # a_base is gravity vec

        # forward pass
        for curr_id in range(NB):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            inds_q = self.robot.get_joint_index_q(curr_id)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
            # compute v and a
            if parent_id == -1:  # parent is fixed base or world
                # v_base is zero so v[:,ind] remains 0
                if self.robot.floating_base:
                    a[:, curr_id] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, curr_id] = np.matmul(Xmat, gravity_vec)
            else:
                v[:, curr_id] = np.matmul(Xmat, v[:, parent_id])
                a[:, curr_id] = np.matmul(Xmat, a[:, parent_id])
            inds_v = self.robot.get_joint_index_v(curr_id)
            _qd = qd[inds_v]
            
            if self.robot.floating_base and curr_id == 0:
                vJ = np.matmul(S, np.transpose(np.matrix(_qd)))
            else: vJ = S * _qd
            v[:, curr_id] += np.squeeze(np.array(vJ))  # reduces shape to (6,) matching v[:,curr_id]
            a[:, curr_id] += self.mxS(vJ, v[:, curr_id])
            if qdd is not None:
                _qdd = qdd[inds_v]
                if self.robot.floating_base and curr_id == 0:
                    aJ = np.matmul(S, np.transpose(np.matrix(_qdd)))
                else: aJ = S * _qdd
                a[:, curr_id] += np.squeeze(np.array(aJ))  # reduces shape to (6,) matching a[:,curr_id]
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:, curr_id] = np.matmul(Imat, a[:, curr_id]) + self.vxIv(v[:, curr_id], Imat)

        return (v, a, f)

    def rnea_bpass(self, q, f):
        """Perform the backward pass of the Recursive Newton-Euler Algorithm.

        Parameters
        ----------
        f : numpy.ndarray
            6N-element internal spatial forces per link.

        Returns
        -------
        (c, f) : tuple
            Generalized forces and updated internal spatial forces.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        m = self.robot.get_num_vel()
        c = np.zeros(m)

        # backward pass
        for curr_id in range(NB - 1, -1, -1):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            inds_f = self.robot.get_joint_index_f(curr_id)
            # compute c
            c[inds_f] = np.matmul(np.transpose(S), f[:, curr_id])
            # update f if applicable
            if parent_id != -1:
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                temp = np.matmul(np.transpose(Xmat), f[:, curr_id])
                f[:, parent_id] = f[:, parent_id] + temp.flatten()

        return (c, f)

    def rnea(
        self,
        q,
        qd,
        qdd=None,
        GRAVITY=-9.81,
        f_ext=None,
        public_output=True,
        normalize_input=True,
    ):
        """Compute the generalized forces using Recursive Newton-Euler Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.
        f_ext : numpy.ndarray
            External forces.

        Returns
        -------
        (c, v, a, f) : tuple
            Generalized forces and intermediate link quantities.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            if qdd is not None:
                qdd = self._normalize_v_input(qdd)
        # forward pass
        (v, a, f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c, f) = self.rnea_bpass(q, f)
        if public_output:
            c = self._denormalize_v_output(c)
        return (c, v, a, f)

    def minv_bpass(self, q):
        """Backward pass for the Articulated-Body Algorithm to compute inverse inertia.

        Parameters
        ----------
        I_art : numpy.ndarray
            Articulated-body inertia matrices.

        Returns
        -------
        (Minv, F, U, Dinv) : tuple
            Matrices for inverse inertia composition.
        """
        # Allocate memory
        NB = self.robot.get_num_bodies()
        if self.robot.floating_base:
            n = NB + 5  # count fb_joint as 6 instead of 1 joint else set n = len(qd)
        else:
            n = self.robot.get_num_vel()
        Minv = np.zeros((n, n))
        F = np.zeros((n, 6, n))
        U = np.zeros((n, 6))
        Dinv = np.zeros(n)

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

        # # Backward pass
        for ind in range(NB - 1, -1, -1):
            subtreeInds = self.robot.get_subtree_by_id(ind)
            if self.robot.floating_base:
                matrix_ind = ind + 5  # use for Minv, F, U, Dinv
                adj_subtreeInds = list(
                    np.array(subtreeInds) + 5
                )  # adjusted for matrix calculation
            else:
                matrix_ind = ind
                adj_subtreeInds = subtreeInds
            parent_ind = self.robot.get_parent_id(ind)
            if (
                parent_ind == -1 and self.robot.floating_base
            ):  # floating base joint check
                # Compute U, D
                S = self.robot.get_S_by_id(ind)  # np.eye(6) for floating base
                U[ind : ind + 6, :] = np.matmul(IA[ind], S)
                fb_Dinv = np.linalg.inv(
                    np.matmul(S.transpose(), U[ind : ind + 6, :])
                )  # vectorized Dinv calc
                # Update Minv and subtrees - subtree calculation for Minv -= Dinv * S.T * F with clever indexing
                Minv[ind : ind + 6, ind : ind + 6] = Minv[ind, ind] + fb_Dinv
                Minv[ind : ind + 6, adj_subtreeInds] -= (
                    np.matmul(
                        np.matmul(fb_Dinv, S), F[ind : ind + 6, :, adj_subtreeInds]
                    )
                )[-1]
            else:
                # Compute U, D
                S = self.robot.get_S_by_id(
                    ind
                )  # NOTE Can S be an np.array not np.matrix? np.matrix outdated...
                U[matrix_ind, :] = np.matmul(IA[ind], S).reshape(6,)
                Dinv[matrix_ind] = np.matmul(S.transpose(), U[matrix_ind, :])
                # Update Minv and subtrees
                Minv[matrix_ind, matrix_ind] = 1 / Dinv[matrix_ind]
                # Deals with issue where result is np.matrix instead of np.array (can't shape np.matrix as 1 dimension)
                Minv[matrix_ind, adj_subtreeInds] -= np.squeeze(
                    np.array(
                        1
                        / (Dinv[matrix_ind])
                        * np.matmul(S.transpose(), F[matrix_ind, :, adj_subtreeInds].T)
                    )
                )
                # update parent if applicable
                parent_ind = self.robot.get_parent_id(ind)
                if parent_ind != -1:
                    if self.robot.floating_base:
                        matrix_parent_ind = parent_ind + 5
                    else:
                        matrix_parent_ind = parent_ind
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    # update F
                    for subInd in adj_subtreeInds:
                        F[matrix_ind, :, subInd] += (
                            U[matrix_ind, :] * Minv[matrix_ind, subInd]
                        )
                        F[matrix_parent_ind, :, subInd] += np.matmul(
                            np.transpose(Xmat), F[matrix_ind, :, subInd]
                        )
                    # update IA
                    Ia = IA[ind] - np.outer(
                        U[matrix_ind, :],
                        ((1 / Dinv[matrix_ind]) * np.transpose(U[matrix_ind, :])),
                    )  # replace 1/Dinv if using linalg.inv
                    IaParent = np.matmul(np.transpose(Xmat), np.matmul(Ia, Xmat))
                    IA[parent_ind] += IaParent

        return Minv, F, U, Dinv

    def minv_fpass(self, q, Minv, F, U, Dinv):
        """Forward pass for the Articulated-Body Algorithm to compute inverse inertia.

        Parameters
        ----------
        Minv : numpy.ndarray
            N-element vector or matrix for Minv.
        F : numpy.ndarray
            Intermediate matrix.
        U : numpy.ndarray
            Intermediate matrix.
        Dinv : numpy.ndarray
            Inverse of articulated inertia projected on subspace.

        Returns
        -------
        Minv : numpy.ndarray
            Inverse of the joint-space inertia matrix.
        """
        NB = self.robot.get_num_bodies()
        # # Forward pass
        for ind in range(NB):
            if self.robot.floating_base:
                matrix_ind = ind + 5
            else:
                matrix_ind = ind
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            parent_ind = self.robot.get_parent_id(ind)
            S = self.robot.get_S_by_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            if parent_ind != -1:
                Minv[matrix_ind, :] -= (1 / Dinv[matrix_ind]) * np.matmul(
                    np.matmul(U[matrix_ind].transpose(), Xmat), F[parent_ind]
                )
                F[ind] = np.matmul(Xmat, F[parent_ind]) + np.outer(
                    S, Minv[matrix_ind, :]
                )
            else:
                if self.robot.floating_base:
                    F[ind] = np.matmul(S, Minv[ind : ind + 6, ind:])
                else:
                    F[ind] = np.outer(S, Minv[ind, :])

        return Minv

    def minv(self, q, output_dense=True, public_output=True, normalize_input=True):
        """Compute the inverse of the joint-space inertia matrix.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.

        Returns
        -------
        Minv : numpy.ndarray
            N x N inverse joint-space inertia matrix.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        # backward pass
        (Minv, F, U, Dinv) = self.minv_bpass(q)

        # forward pass
        Minv = self.minv_fpass(q, Minv, F, U, Dinv)

        # fill in full matrix (currently only upper triangular)
        if output_dense:
            NB = self.robot.get_num_bodies()
            for col in range(NB):
                for row in range(NB):
                    if col < row:
                        Minv[row, col] = Minv[col, row]

        if public_output:
            return self._denormalize_qv_matrix_output(Minv, row_space="v", col_space="v")
        return Minv
    

    def crm(self,v):
        """Spatial velocity cross product operator.

        Parameters
        ----------
        v : numpy.ndarray
            6D spatial velocity vector.

        Returns
        -------
        v_cross : numpy.ndarray
            6x6 cross product matrix.
        """
        if len(v) == 6:
            vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
        else:
            vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
        return vcross


    def aba(self, q, qd, tau, f_ext=[], GRAVITY = -9.81, normalize_input=True):
        """Compute forward dynamics using the Articulated-Body Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.
        f_ext : numpy.ndarray
            External forces.

        Returns
        -------
        qdd : numpy.ndarray
            N-element joint accelerations.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            tau = self._normalize_v_input(tau)
        if self.robot.floating_base:
            # allocate memory TODO check NB vs. n
            n = len(qd)
            NB = self.robot.get_num_bodies()
            v = np.zeros((6,NB))
            c = np.zeros((6,NB))
            a = np.zeros((6,NB))
            IA = np.zeros((NB,6,6))
            pA = np.zeros((6,NB))
            # variables may require special indexing
            f = np.zeros((6,n))
            # d = np.zeros(n)
            d = {}
            U = np.zeros((6,n))
            u = np.zeros(n)
            qdd = np.zeros(n)

            gravity_vec = np.zeros((6))
            gravity_vec[5] = -GRAVITY  # a_base is gravity vec

            # Initial Forward Pass
            for ind in range(NB): # curr_id = ind for this loop
                parent_ind = self.robot.get_parent_id(ind)
                _q = q[self.robot.get_joint_index_q(ind)]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                S = self.robot.get_S_by_id(ind)
                inds_v = self.robot.get_joint_index_v(ind)

                if parent_ind == -1: # parent is base
                    if self.robot.floating_base:
                        v[:, ind] = np.matmul(S, qd[ind:ind+6])
                    else:
                        v[:, ind] = np.squeeze(np.array(S*qd[ind]))
                else:
                    v[:, ind] = np.matmul(Xmat, v[:, parent_ind]) 
                    vJ = np.squeeze(np.array(S * qd[inds_v])) # reduces shape to (6,) matching v[:,curr_id]
                    v[:, ind] += vJ
                    c[:, ind] = np.matmul(self.cross_operator(v[:, ind]), vJ)

                Imat = self.robot.get_Imat_by_id(ind)
                # print(f'Imat:{Imat.shape}\n {Imat}')
                # print(IA[:,:,ind].shape)
                IA[ind] = Imat

                vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
                [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
                [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
                [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
                [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
                [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

                crf = -np.transpose(vcross) 
                temp = np.matmul(crf, Imat)

                pA[:, ind] = np.matmul(temp, v[:, ind])

            # apply external forces
            pA = self.apply_external_forces(q, pA, f_ext)

            # Backward Pass
            for ind in range(NB-1, -1, -1): # ind != ind for bpass
                S = self.robot.get_S_by_id(ind)
                parent_ind = self.robot.get_parent_id(ind)
                inds_v = self.robot.get_joint_index_v(ind)
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)

                U[:, inds_v] = np.squeeze(np.matmul(IA[ind], S))
                d[ind] = np.matmul(np.transpose(S), U[:, inds_v])
                u[inds_v] = tau[inds_v] - (np.matmul(S.T, pA[:, ind])) - (np.matmul(U[:, inds_v].T, c[:, ind]))

                if parent_ind != -1:
                    U[:, inds_v] = np.matmul(Xmat.T, U[:, inds_v]) # spatial edit

                    rightSide = np.reshape(U[:, inds_v], (6,1)) @ np.reshape(U[:, inds_v], (6,1)).T / d[ind]
                    Ia = np.matmul(Xmat.T, np.matmul(IA[ind], Xmat)) - rightSide # spatial edit

                    pa = np.matmul(Xmat.T, pA[:, ind] + np.matmul(IA[ind], c[:, ind]))
                    pa = pa + np.reshape(U[:, inds_v], (6, 1)).flatten() * ((1 / d[ind]) * u[inds_v])
                    
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    temp = np.matmul(np.transpose(Xmat), Ia)

                    IA[parent_ind] = IA[parent_ind] + Ia # spatial edit

                    pA[:, parent_ind] = pA[:, parent_ind] + pa # spatial edit


            # Final Forward Pass
            for ind in range(NB): # ind != ind for bpass
                parent_ind = self.robot.get_parent_id(ind)
                inds_q = self.robot.get_joint_index_q(ind)
                inds_v = self.robot.get_joint_index_v(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)

                if parent_ind == -1: # parent is base
                    a[:, ind] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, ind] = a[:, parent_ind]
                
                S = self.robot.get_S_by_id(ind)
                temp = u[inds_v] - np.matmul(np.transpose(U[:, inds_v]), a[:, ind])

                if parent_ind == -1:
                    # qdd[inds_v] = np.matmul(np.linalg.inv(d[ind]), temp)
                    if self.robot.floating_base:
                        qdd[inds_v] = np.linalg.solve(d[ind], temp)
                        a[:, ind] = a[:, ind] + np.matmul(S, qdd[inds_v]) + c[:, ind]
                    else:
                        qdd[ind] = temp / d[ind]
                        a[:, ind] = np.matmul(Xmat, a[:, ind]) + qdd[ind]*S.T + c[:, ind]
                else:
                    # qdd[inds_v] = np.linalg.inv(d[ind]) * temp
                    qdd[inds_v] = temp / d[ind]
                    a[:, ind] = np.matmul(Xmat, a[:, ind]) + np.dot(S.T,qdd[inds_v]) + c[:, ind]
        else:
            n = len(qd)
            v = np.zeros((6,n))
            c = np.zeros((6,n))
            a = np.zeros((6,n))
            f = np.zeros((6,n))
            d = np.zeros(n)
            U = np.zeros((6,n))
            u = np.zeros(n)
            IA = np.zeros((6,6,n))
            pA = np.zeros((6,n))
            qdd = np.zeros(n)
            
            
            gravity_vec = np.zeros((6))
            gravity_vec[5] = -GRAVITY # a_base is gravity vec
                    
            for ind in range(n):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)

                if parent_ind == -1: # parent is base
                    v[:,ind] = np.squeeze(np.array(S*qd[ind]))
                    
                else:
                    v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
                    v[:,ind] += np.squeeze(np.array(S*qd[ind]))
                    c[:,ind] = self._mxS(S,v[:,ind],qd[ind])

                Imat = self.robot.get_Imat_by_id(ind)

                IA[:,:,ind] = Imat

                vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
                [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
                [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
                [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
                [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
                [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

                crf=-np.transpose(vcross)
                temp=np.matmul(crf,Imat)

                pA[:,ind] = np.matmul(temp, v[:,ind])
            
            for ind in range(n-1,-1,-1):
                S = self.robot.get_S_by_id(ind)
                parent_ind = self.robot.get_parent_id(ind)

                U[:,ind] = np.squeeze(np.array(np.matmul(IA[:,:,ind],S)))
                d[ind] = np.matmul(np.transpose(S),U[:,ind])
                u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

                if parent_ind != -1:

                    rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
                    Ia = IA[:,:,ind] - rightSide

                    pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind]

                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                    temp = np.matmul(np.transpose(Xmat), Ia)

                    IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)

                    temp = np.matmul(np.transpose(Xmat), pa)
                    pA[:,parent_ind]=pA[:,parent_ind] + temp.flatten()
                                                
            for ind in range(n):

                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

                if parent_ind == -1: # parent is base
                    a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
                else:
                    a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

                S = self.robot.get_S_by_id(ind)
                temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
                qdd[ind] = temp / d[ind]
                a[:,ind] = a[:,ind] + qdd[ind]*S.T
        
        return self._denormalize_v_output(qdd)




    def crba(self, q, normalize_input=True):
        """Compute the joint-space inertia matrix using the Composite Rigid Body Algorithm.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.

        Returns
        -------
        M : numpy.ndarray
            N x N joint-space inertia matrix.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
        if self.robot.floating_base:
            NB = self.robot.get_num_bodies()
            n = self.robot.get_num_vel()
            H = np.zeros((n, n))

            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            for ind in range(NB - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                matrix_ind = ind + 5
                if ind > 0:
                    _q = q[self.robot.get_joint_index_q(ind)]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )
                    fh = np.matmul(IC[ind], S)
                    H[matrix_ind, matrix_ind] = np.matmul(S.T, fh)
                    j = ind
                    while self.robot.get_parent_id(j) > 0:
                        Xmat = self.robot.get_Xmat_Func_by_id(j)(
                            q[self.robot.get_joint_index_q(j)]
                        )
                        fh = np.matmul(Xmat.T, fh)
                        j = self.robot.get_parent_id(j)
                        S = self.robot.get_S_by_id(j)
                        H[matrix_ind, j + 5] = np.matmul(fh.T, S)
                        H[j + 5, matrix_ind] = H[matrix_ind, j + 5]
                    # # treat floating base 6 dof joint
                    inds_q = self.robot.get_joint_index_q(j)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(_q)
                    S = np.eye(6)
                    fh = np.matmul(Xmat.T, fh)
                    H[matrix_ind, :6] = np.matmul(fh.T, S)
                    H[:6, matrix_ind] = H[matrix_ind, :6].T
                else:
                    ind = 0
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    parent_ind = self.robot.get_parent_id(ind)
                    fh = np.matmul(IC[ind], S)
                    H[:6, :6] = np.matmul(S.T, fh)

            # keep the user-facing floating-base convention in one place:
            # the root block already matches the Pinocchio-style ordering,
            # but the root-to-joint cross terms are still accumulated in the
            # internal spatial row order [wx, wy, wz, vx, vy, vz].
            root_order = [3, 4, 5, 0, 1, 2]
            root_cross = H[:6, 6:].copy()
            H[:6, 6:] = root_cross[root_order, :]
            H[6:, :6] = np.transpose(H[:6, 6:])
        else:
            # # Fixed base implmentation of CRBA
            n = len(q)
            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            for ind in range(n - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

                if parent_ind != -1:
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )

            H = np.zeros((n, n))

            for ind in range(n):
                S = self.robot.get_S_by_id(ind)
                fh = np.matmul(IC[ind], S)
                H[ind, ind] = np.matmul(S.T, fh)
                j = ind

                while self.robot.get_parent_id(j) > -1:
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                    
                    fh = np.matmul(Xmat.T, fh) # add an addition Xmat.T everytime
                    j = self.robot.get_parent_id(j)

                    
                    S = self.robot.get_S_by_id(j)
                    H[ind, j] = np.matmul(S.T, fh)
                    H[j, ind] = H[ind, j]

        return self._denormalize_qv_matrix_output(H, row_space="v", col_space="v")

    ##### Testing original RNEA_grad to help with CUDA 
    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        """Forward pass gradient with respect to joint positions for RNEA.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (dv_dq, da_dq, df_dq) : tuple
            Gradients of velocity, acceleration, and force wrt q.
        """
        
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        dv_dq = np.zeros((6,n,NB))  # each body has its own derivative matrix with a column for each position
        da_dq = np.zeros((6,n,NB))
        df_dq = np.zeros((6,n,NB))

        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            if self.robot.floating_base: 
                # dc_dqd gets idx
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else:
                idx = ind
                parent_idx = parent_ind
            # Xmat access sequence
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,idx,ind] += self._mxS(S,np.matmul(Xmat,v[:,parent_ind])) # replace with new mxS
                
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            for c in range(n):
                if parent_ind == -1 and self.robot.floating_base:
                    # dv_dq should be all 0s => this results in all 0s
                    for ii in range(len(idx)):
                        da_dq[:,c,ii] += self._mxS(S[ii],dv_dq[:,c,ii],qd[ii]) # dv/du x S*q
                else:
                    da_dq[:,c,ind] += self._mxS(S,dv_dq[:,c,ind],qd[idx]) # replace with new mxS
                    
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,idx,ind] += self._mxS(S,np.matmul(Xmat,a[:,parent_ind])) # replace with new mxS
            else:
                if self.robot.floating_base:
                    root_gravity = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    root_gravity = np.matmul(Xmat, gravity_vec)
                da_dq[:,idx,ind] += self._mxS(S,root_gravity) # replace with new mxS 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            
            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])# puts 0.0014 instead of -0.0014 in df_dq[2,7,7]
            Iv = np.matmul(Imat,v[:,ind])
       
            for c in range(n):
               
                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))
    
        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
        """Forward pass gradient with respect to joint velocities for RNEA.

        Parameters
        ----------
        qd : numpy.ndarray
            N-element joint velocities.

        Returns
        -------
        (dv_dqd, da_dqd, df_dqd) : tuple
            Gradients of velocity, acceleration, and force wrt qd.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = len(qd)
        dv_dqd = np.zeros((6,n,NB))
        da_dqd = np.zeros((6,n,NB))
        df_dqd = np.zeros((6,n,NB))

        # forward pass
        for ind in range(NB):
            parent_ind = self.robot.get_parent_id(ind)
            if self.robot.floating_base:
                # dc_dqd gets idx, special matrix indexing
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else: 
                idx = ind
                parent_idx = parent_ind
            # Xmat access sequence
            inds_v = self.robot.get_joint_index_v(ind) #joint index for all joints without quaternion (does special joint indexing by itself)
            inds_q = self.robot.get_joint_index_q(ind) #joint index for all joints
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
            dv_dqd[:,inds_v,ind] += np.squeeze(np.array(S)) # added squeeze and mxS
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
            for c in range(n): 
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        da_dqd[:,c,ind] += self._mxS(S[ii],dv_dqd[:,c,ind],qd[ii]) 
                else:
                    da_dqd[:,c,ind] += self._mxS(S,dv_dqd[:,c,ind],qd[idx]) 

            
            da_dqd[:,idx,ind] += self._mxS(S,v[:,ind]) 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            
            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                
                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))
        
        
        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
        """Backward pass gradient with respect to joint positions for RNEA.

        Parameters
        ----------
        df_dq : numpy.ndarray
            Gradient of spatial forces wrt q.

        Returns
        -------
        dc_dq : numpy.ndarray
            Gradient of generalized forces wrt q.
        """

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel() # assuming len(q) = len(qd)
        dc_dq = np.zeros((n,n))
        
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)

            if self.robot.floating_base:
                # dc_dqd gets idx
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else:
                idx = ind
                parent_idx = parent_ind
            
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            if parent_ind == -1 and self.robot.floating_base:
                dc_dq[idx,:] = np.matmul(np.transpose(S),df_dq[:,:,ind])
            else:
                dc_dq[idx,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            if parent_ind != -1:
                # Xmat access sequence
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                for entry in range(6):
                    df_dq[entry,idx,parent_ind] += delta_dq[entry]
                    
            
        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
        """Backward pass gradient with respect to joint velocities for RNEA.

        Parameters
        ----------
        df_dqd : numpy.ndarray
            Gradient of spatial forces wrt qd.

        Returns
        -------
        dc_dqd : numpy.ndarray
            Gradient of generalized forces wrt qd.
        """

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel() # len(qd) always
        dc_dqd = np.zeros((n,n))
        
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)

            if self.robot.floating_base:
                # dc_dqd gets idx, special matrix indexing
                if parent_ind != -1:
                    idx = ind + 5
                    parent_idx = parent_ind + 5
                else:
                    idx = [0,1,2,3,4,5]
            else: 
                idx = ind
                parent_idx = parent_ind
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            # if parent_ind == -1 and self.robot.floating_base:
            #     for ii in range(len(idx)):
            #         dc_dqd[ii,:] = np.matmul(np.transpose(S[ii]),df_dqd[:,:,ii])
            # else:
            dc_dqd[idx,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            if parent_ind != -1:
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

            
        # add in the damping and simplify this expression later
        # suggestion: have a getter function that automatically indexes and allocates for floating base functions
        if USE_VELOCITY_DAMPING:
            for ind in range(NB):
                if self.robot.floating_base and self.robot.get_parent_id(ind) == -1:
                    dc_dqd[ind:ind+5, ind:ind+5] += self.robot.get_damping_by_id(ind)
                else:
                    dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)
        
        return dc_dqd

    def rnea_grad(
        self,
        q,
        qd,
        qdd = None,
        GRAVITY = -9.81,
        USE_VELOCITY_DAMPING = False,
        public_output=True,
        normalize_input=True,
    ):
        """Compute the gradients of RNEA wrt joint positions and velocities.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (dc_dq, dc_dqd) : tuple
            Gradients of generalized forces wrt q and qd.
        """
        
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            if qdd is not None:
                qdd = self._normalize_v_input(qdd)
        (c, v, a, f) = self.rnea(
            q,
            qd,
            qdd,
            GRAVITY,
            public_output=False,
            normalize_input=False,
        )

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)
 
        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        if public_output:
            return self._denormalize_rnea_grad_output(dc_dq, dc_dqd)
        return np.hstack((dc_dq, dc_dqd))


    def forward_dynamics(self, q, qd, u, public_output=True, normalize_input=True):
        """Compute the joint accelerations for the given state and torques.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        qdd : numpy.ndarray
            N-element joint accelerations.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            u = self._normalize_v_input(u)
        (c,v,a,f) = self.rnea(q, qd, public_output=False, normalize_input=False)
        minv = self.minv(q, public_output=False, normalize_input=False)
        qdd = np.matmul(minv, u - c)
        if public_output:
            return self._denormalize_v_output(qdd)
        return qdd
    
    def forward_dynamics_grad(self, q, qd, u, normalize_input=True):
        """Compute the gradients of the forward dynamics.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        (qdd_dq, qdd_dqd) : tuple
            Gradients of joint accelerations wrt q and qd.
        """
        if normalize_input:
            q = self._normalize_q_input(q)
            qd = self._normalize_v_input(qd)
            u = self._normalize_v_input(u)
        qdd = self.forward_dynamics(q, qd, u, public_output=False, normalize_input=False)
        dc_du = self.rnea_grad(q, qd, qdd, public_output=False, normalize_input=False)
        dc_dq, dc_dqd = np.hsplit(dc_du, [len(qd)])

        minv = self.minv(q, public_output=False, normalize_input=False)
        qdd_dq = np.matmul(-minv, dc_dq)
        qdd_dqd = np.matmul(-minv, dc_dqd)
        return (
            self._denormalize_reduced_q_matrix_output(qdd_dq, row_space="v"),
            self._denormalize_qv_matrix_output(qdd_dqd, row_space="v", col_space="v"),
        )


    def idsva_so(self, q, qd, qdd, GRAVITY = -9.81):
        """Compute second-order derivatives of inverse dynamics via parallel IDSVA.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        qdd : numpy.ndarray
            N-element joint accelerations.

        Returns
        -------
        (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq) : tuple
            Second-order derivatives of torques and inertia matrix.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        v = np.zeros((6,NB))
        a = np.zeros((6,NB))
        f = np.zeros((6,NB))
        Xup0 =  [None] * NB #list of transformation matrices in the world frame
        Xdown0 = [None] * NB
        IC = [None] * NB
        BC = [None] * NB
        S = [None] * NB
        Sd = [None] * NB
        vJ = np.zeros((6,NB))
        aJ = np.zeros((6,NB))
        psid = [None] * NB
        psidd = [None] * NB
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        def inds_to_list(inds):
            if isinstance(inds, list):
                return inds
            if isinstance(inds, tuple):
                return list(inds)
            if isinstance(inds, np.ndarray):
                return list(inds.flatten())
            return [inds]

        body_v_inds = [inds_to_list(self.robot.get_joint_index_v(i)) for i in range(NB)]

        def subtree_vel_inds(subtree):
            st_inds = []
            for body_id in subtree:
                st_inds += body_v_inds[body_id]
            return st_inds

        # forward pass 
        modelNB = NB
        modelNV = n
        for i in range(modelNB):
            parent_i = self.robot.get_parent_id(i)
            inds_q = self.robot.get_joint_index_q(i)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(i)(_q)
          # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                if self.robot.floating_base:
                    a[:, i] = np.matmul(np.linalg.inv(Xmat), gravity_vec)
                else:
                    a[:, i] = gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i]) 
            S[i] = self.robot.get_S_by_id(i)
            if len(S[i].shape) == 1:
                S[i] = np.reshape(S[i], (6,1))
            S[i] = Xdown0[i] @ S[i]
            inds_v = self.robot.get_joint_index_v(i)
            _qd = np.atleast_1d(qd[inds_v])
            _qdd = np.atleast_1d(qdd[inds_v])
            vJ[:,i] = np.reshape(np.matmul(S[i], _qd), (6,))
            aJ[:,i] = self.cross_operator(v[:,i])@vJ[:,i] + np.reshape(np.matmul(S[i], _qdd), (6,))
            psid[i] = self.cross_operator(v[:,i])@S[i]
            psidd[i] = self.cross_operator(a[:,i])@S[i] + self.cross_operator(v[:,i])@psid[i]
            v[:,i] = v[:,i] + vJ[:,i]
            a[:,i] = a[:,i] + aJ[:,i]
            I = self.robot.get_Imat_by_id(i)
            IC[i] = np.array(Xup0[i]).T @ (I @ Xup0[i])
            Sd[i] = self.cross_operator(v[:,i]) @ S[i]
            BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
            f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @v[:,i] 

        #backward pass: Can be parallelized across all j,d
        for i in range(modelNB-1,-1,-1):
            pi = self.robot.get_parent_id(i)
            if pi >= 0:
                    IC[pi] = IC[pi] + IC[i]
                    BC[pi] = BC[pi] + BC[i]
                    f[:, pi] = f[:, pi] + f[:, i]
        
        T1 = np.zeros((6,n))
        T2 = np.zeros((6,n))
        T3 = np.zeros((6,n))
        T4 = np.zeros((6,n))
        D1 = np.zeros((36,n))
        D2 = np.zeros((36,n))
        D3 = np.zeros((36,n))
        D4 = np.zeros((36,n))
        
        for j in range(modelNB-1,-1,-1):      
            for d in range(S[j].shape[1]):
                S_d = S[j][:, d]
                Sd_d = Sd[j][:, d]
                psid_d = psid[j][:, d]
                psidd_d = psidd[j][:, d]


                Bic_phii1 =  self.dual_cross_operator(S_d)@IC[j] 
                Bic_phii2 = self.icrf(IC[j] @ S_d)
                Bic_phii3 = -IC[j] @ self.cross_operator(S_d)
                
                Bic_phii = Bic_phii1+Bic_phii2+Bic_phii3 # almost complete
               
                Bic_psii_dot = 2 * 0.5 * (self.dual_cross_operator(psid_d) @ IC[j] + self.icrf(IC[j] @ psid_d) - IC[j] @ self.cross_operator(psid_d))
                
                dd = body_v_inds[j][d]
                A1 = self.dot_matrix(IC[j], S_d) # crf(S_d) @ IC[j] - (IC @ crm(S_d))
                A2 = Bic_psii_dot + self.dot_matrix(BC[j], S_d) # crf(S_d) @ BC[j] - (BC[j] @ crm(S_d))
                A3 = self.icrf(IC[j].T @ S_d)
        

                T1[:, dd] = IC[j] @ S_d
                T2[:, dd] = -BC[j].T @ S_d
                T3[:, dd] = BC[j] @ psid_d + IC[j] @ psidd_d + self.icrf(f[:, j]) @ S_d
                T4[:, dd] = BC[j] @ S_d + IC[j] @ (psid_d + Sd_d)

                

                D1[:, dd] = A1.flatten()
                D2[:, dd] = A2.flatten(order='F')
                D3[:, dd] = Bic_phii.flatten(order='F')
                D4[:, dd] = A3.flatten(order='F')          

        dM_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dqd = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dvdq = np.zeros((modelNV,modelNV,modelNV))
        
        #backward pass: Can be parallelized over all j,d,k,c 
        for j in range(modelNB-1,-1,-1):
            st_j = self.robot.get_subtree_by_id(j) # Subtree of j
            st_j_inds = subtree_vel_inds(st_j)
            succ_j = [i for i in st_j if i != j] # Joint successors
            succ_j_inds = subtree_vel_inds(succ_j)
            for d in range(S[j].shape[1]):
                dd = body_v_inds[j][d]
                S_d = S[j][:, d]
                Sd_d = Sd[j][:, d]
                psid_d = psid[j][:, d]
                psidd_d = psidd[j][:, d]
                ancestor_j = self.robot.get_ancestors_by_id(j)
                ancestor_j.insert(0, j)
                ancestor_j = ancestor_j[::-1]
                for k in ancestor_j:  # Assuming model['ancestors'][j] provides a list of ancestor indices
                    for c in range(S[k].shape[1]):
                        cc = body_v_inds[k][c]
                        S_c = S[k][:, c]
                        Sd_c = Sd[k][:, c]
                        psid_c = psid[k][:, c]
                        psidd_c = psidd[k][:, c]

                        # Compute temporary vectors
                        t1 = np.outer(S_d, psid_c.transpose()).flatten(order='F')
                        t2 = np.outer(S_d, S_c.transpose()).flatten(order='F')
                        t3 = np.outer(psid_d, psid_c.transpose()).flatten(order='F')
                        t4 = np.outer(S_d, psidd_c.transpose()).flatten(order='F')
                        t5 = np.outer(S_d, Sd_c + psid_c.transpose()).flatten(order='F')
                        t8 = np.outer(S_c, S_d.transpose()).flatten(order='F')
                        
                        # Computing the cross products
                        p1 = self.cross_operator(psid_c) @ S_d
                        p2 = self.cross_operator(psidd_c) @ S_d
                        
                        # Updating the tensors based on the computed vectors and cross products
                        # Fixed-base/scalar-joint version:
                        # d2tau_dq[st_j, dd, cc] = -np.dot(t3, D3[:, st_j]) - np.dot(p1, T2[:, st_j]) + np.dot(p2, T1[:, st_j])
                        #
                        # This broadcast works when each body contributes exactly one
                        # velocity coordinate, so the subtree body list and subtree
                        # velocity-index list are effectively the same object. Once the
                        # floating root contributes a 6-column block, that implicit
                        # one-body/one-index assumption no longer holds, so use the
                        # expanded subtree velocity indices instead.
                        primary_d2tau_dq_vec = -np.dot(t3, D3[:, st_j_inds]) - np.dot(p1, T2[:, st_j_inds]) + np.dot(p2, T1[:, st_j_inds])
                        for st_ind, st_j_ind in enumerate(st_j_inds):
                            d2tau_dq[st_j_ind, dd, cc] = primary_d2tau_dq_vec[st_ind]
                        d2tau_dvdq[st_j_inds, dd, cc] = -np.dot(t1, D3[:, st_j_inds])

                        # st_j is list of all ancestors of j
                        if k < j:
                            t6 = np.outer(S_c, psid_d.transpose()).flatten(order='F')
                            t7 = np.outer(S_c, psidd_d.transpose()).flatten(order='F')
                            p3 = self.cross_operator(S_c) @ S_d
                            p4 = self.cross_operator(Sd_c + psid_c) @ S_d - 2 * self.cross_operator(psid_d) @ S_c
                            p5 = self.cross_operator(S_d) @ S_c
                            
                            # Fixed-base/scalar-joint version:
                            # d2tau_dq[st_j, cc, dd] = d2tau_dq[st_j, dd, cc]
                            #
                            # The symmetry itself still holds, but the index set has to
                            # live in velocity-coordinate space rather than body-id
                            # space once a joint can contribute multiple local columns.
                            for st_j_ind in st_j_inds:
                                d2tau_dq[st_j_ind, cc, dd] = d2tau_dq[st_j_ind, dd, cc]

                            
                            d2tau_dqd[st_j_inds, cc, dd] = -np.dot(t2.T, D3[:, st_j_inds])
                            d2tau_dqd[st_j_inds, dd, cc] = d2tau_dqd[st_j_inds, cc, dd]
                            
                            
                            d2tau_dvdq[st_j_inds, cc, dd] = -np.dot(t6, D3[:, st_j_inds]) - np.dot(p3, T2[:, st_j_inds]) + np.dot(p4, T1[:, st_j_inds])
                        
                            # Fixed-base/scalar-joint version:
                            # d2tau_dq[cc, st_j, dd] = np.dot(t6, D2[:, st_j]) + np.dot(t7, D1[:, st_j]) - np.dot(p5, T3[:, st_j])
                            #
                            # This was the old fixed-base subtree broadcast. It works
                            # when every successor body contributes one scalar
                            # coordinate. In floating-base mode the root contributes a
                            # multi-column block, so the same algebra has to be written
                            # against the expanded subtree velocity-index list.
                            transpose_d2tau_dq_vec = np.dot(t6, D2[:, st_j_inds]) + np.dot(t7, D1[:, st_j_inds]) - np.dot(p5, T3[:, st_j_inds])
                            for st_ind, st_j_ind in enumerate(st_j_inds):
                                d2tau_dq[cc, st_j_ind, dd] = transpose_d2tau_dq_vec[st_ind]
                            
                            d2tau_dvdq[cc, st_j_inds, dd] = np.dot(t6, D3[:, st_j_inds]) - np.dot(p5, T4[:, st_j_inds])             


                            # S_d @ IC[j] is just T1
                            # self.dual_cross_operator(S_d) @ IC[j] is first part of D1
                            # Reuse these in CUDA
                            d2tau_dqd[cc,dd,dd] = (S_d.T @ IC[j] @ self.cross_operator(S_c) + S_c.T @ self.dual_cross_operator(S_d) @ IC[j] )  @ S_d
                            
                            dM_dq[cc,st_j_inds,dd] = t8.T @ D4[:, st_j_inds]
                            dM_dq[st_j_inds,cc,dd] = dM_dq[cc,st_j_inds,dd]
                            
                            if succ_j_inds:
                                t9 = np.outer(S_c, Sd_d + psid_d) 
                                t9 = t9.flatten(order='F')

                                
                                d2tau_dqd[cc, succ_j_inds, dd] = np.dot(t8, D3[:, succ_j_inds])
                                d2tau_dqd[cc, dd, succ_j_inds] = d2tau_dqd[cc, succ_j_inds, dd]
                                
                                
                                d2tau_dvdq[cc, dd, succ_j_inds] = np.dot(t8, D2[:, succ_j_inds]) + np.dot(t9, D1[:, succ_j_inds])
                                
                                
                                
                                # Fixed-base/scalar-joint version:
                                # d2tau_dq[cc, dd, succ_j] = d2tau_dq[cc, succ_j, dd]
                                #
                                # The old reuse-by-body shortcut assumes each successor
                                # contributes one coordinate. Keep the same symmetry, but
                                # spell it in successor velocity-index space.
                                for succ_j_ind in succ_j_inds:
                                    d2tau_dq[cc, dd, succ_j_ind] = d2tau_dq[cc, succ_j_ind, dd]
                                
                        if succ_j_inds:
                            # Fixed-base/scalar-joint version:
                            # d2tau_dq[dd, cc, succ_j] = np.dot(t1, D2[:, succ_j]) + np.dot(t4, D1[:, succ_j])
                            #
                            # Again, the vector expression is the same, but the direct
                            # successor-body broadcast only stays valid when successor
                            # joints are all 1-DoF. Use the expanded successor
                            # velocity-index list for floating-base compatibility.
                            successor_d2tau_dq_vec = np.dot(t1, D2[:, succ_j_inds]) + np.dot(t4, D1[:, succ_j_inds])
                            for succ_ind, succ_j_ind in enumerate(succ_j_inds):
                                d2tau_dq[dd, cc, succ_j_ind] = successor_d2tau_dq_vec[succ_ind]
                            

                            d2tau_dqd[dd, cc, succ_j_inds] = np.dot(t2, D3[:, succ_j_inds])
                            d2tau_dqd[dd, succ_j_inds, cc] = d2tau_dqd[dd, cc, succ_j_inds]


                            d2tau_dvdq[dd, succ_j_inds, cc] = np.dot(t1, D3[:, succ_j_inds])

                            # Fixed-base/scalar-joint version:
                            # d2tau_dq[dd, succ_j, cc] = d2tau_dq[dd, cc, succ_j]
                            #
                            # Preserve the old symmetry relation, but apply it in the
                            # expanded successor velocity-index space.
                            for succ_j_ind in succ_j_inds:
                                d2tau_dq[dd, succ_j_ind, cc] = d2tau_dq[dd, cc, succ_j_ind]


                            d2tau_dvdq[dd, cc, succ_j_inds] = np.dot(t2, D2[:, succ_j_inds]) + np.dot(t5, D1[:, succ_j_inds])
                            
                            
                            dM_dq[cc, dd, succ_j_inds] = np.dot(t8, D1[:, succ_j_inds])
                            dM_dq[dd, cc, succ_j_inds] = dM_dq[cc, dd, succ_j_inds]
                        
                        if k == j: 
                            d2tau_dqd[st_j_inds, dd, cc] = -np.dot(t2, D1[:, st_j_inds])

        if self.robot.floating_base:
            # The block-aware extension above is enough for:
            #   - d2tau_dqd
            #   - d2tau_dvdq
            #   - dM_dq
            #
            # but the q-side second derivative tensor d2tau_dq still does not
            # carry over cleanly from the old scalar-joint derivation. Rather
            # than replacing the whole second-order path, patch only d2tau_dq
            # from the already-verified first-order floating rnea_grad path and
            # leave the rest of the second-order tensors analytic.
            q = np.asarray(q, dtype=np.float64).copy()
            if self.robot.using_quaternion:
                quat = q[3:7]
                quat_norm = np.linalg.norm(quat)
                if quat_norm == 0.0:
                    raise ValueError("Floating-base quaternion norm was zero during second-order dynamics normalization.")
                q[3:7] = quat / quat_norm
            qd = np.asarray(qd, dtype=np.float64)
            qdd = np.asarray(qdd, dtype=np.float64)
            step = 1e-6
            for dind in range(n):
                # Reduced floating q-side dynamics convention:
                #   [x, y, z, qx, qy, qz, joints...]
                # so the articulated coordinates are shifted by one slot in q
                # because qw is present in configuration space but not in nv.
                q_pos = q.copy()
                q_neg = q.copy()
                if dind < 6:
                    q_pos[dind] += step
                    q_neg[dind] -= step
                else:
                    q_pos[dind + 1] += step
                    q_neg[dind + 1] -= step
                if self.robot.using_quaternion:
                    q_pos[3:7] = q_pos[3:7] / np.linalg.norm(q_pos[3:7])
                    q_neg[3:7] = q_neg[3:7] / np.linalg.norm(q_neg[3:7])
                dc_dq_pos, _dc_dqd_pos = np.hsplit(self.rnea_grad(q_pos, qd, qdd, GRAVITY), [n])
                dc_dq_neg, _dc_dqd_neg = np.hsplit(self.rnea_grad(q_neg, qd, qdd, GRAVITY), [n])
                d2tau_dq[:, :, dind] = (dc_dq_pos - dc_dq_neg) / (2.0 * step)
        return d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq
    
    def fdsva_so(self, q, qd, u, GRAVITY = -9.81):
        """Compute second-order derivatives of forward dynamics.

        Parameters
        ----------
        q : numpy.ndarray
            N-element joint positions.
        qd : numpy.ndarray
            N-element joint velocities.
        tau : numpy.ndarray
            N-element joint torques.

        Returns
        -------
        (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq) : tuple
            Second-order gradients of forward dynamics.
        """
        Minv = self.minv(q)
        qdd = self.forward_dynamics(q, qd, u)
        di2_dq, di2_dqd, di2_dvdq, dm_dq = self.idsva_so(q, qd, qdd, GRAVITY)
        fd_dq, fd_dqd = self.forward_dynamics_grad(q, qd, u)

        daba_dqdq = -np.einsum('il,ljk->ijk', Minv, di2_dq + np.einsum('ilk,lj->ijk', dm_dq, fd_dq) + np.einsum('ilk,lj->ikj', dm_dq, fd_dq))
        daba_dvdq = -np.einsum('il,ljk->ijk', Minv, di2_dvdq + np.einsum('ilk,lj->ijk', dm_dq, fd_dqd))
        # daba_dqdv = -np.einsum('il,ljk->ijk', Minv, di2_dqd + np.einsum('ilk,lj->ikj', dm_dq, fd_dqd)) # Rotate second term
        daba_dvdv = -np.einsum('il,ljk->ijk', Minv, di2_dqd)
        daba_dtdq = -np.einsum('il,ljk->ijk', Minv, np.einsum('ilk,lj->ijk', dm_dq, Minv))

        if self.robot.floating_base:
            # The floating-base reduced-q convention is already verified in the
            # first-order forward_dynamics_grad path. Keep the old analytic
            # composition for the velocity-side and torque-side second-order
            # tensors, but patch daba_dqdq from the first-order dq gradient so
            # the floating completion mirrors the targeted d2tau_dq patch used
            # in idsva_so above.
            q = np.asarray(q, dtype=np.float64).copy()
            if self.robot.using_quaternion:
                quat = q[3:7]
                quat_norm = np.linalg.norm(quat)
                if quat_norm == 0.0:
                    raise ValueError("Floating-base quaternion norm was zero during second-order forward-dynamics normalization.")
                q[3:7] = quat / quat_norm
            qd = np.asarray(qd, dtype=np.float64)
            u = np.asarray(u, dtype=np.float64)
            step = 1e-6
            for dind in range(self.robot.get_num_vel()):
                q_pos = q.copy()
                q_neg = q.copy()
                if dind < 6:
                    q_pos[dind] += step
                    q_neg[dind] -= step
                else:
                    q_pos[dind + 1] += step
                    q_neg[dind + 1] -= step
                if self.robot.using_quaternion:
                    q_pos[3:7] = q_pos[3:7] / np.linalg.norm(q_pos[3:7])
                    q_neg[3:7] = q_neg[3:7] / np.linalg.norm(q_neg[3:7])
                fd_dq_pos, _fd_dqd_pos = self.forward_dynamics_grad(q_pos, qd, u)
                fd_dq_neg, _fd_dqd_neg = self.forward_dynamics_grad(q_neg, qd, u)
                daba_dqdq[:, :, dind] = (fd_dq_pos - fd_dq_neg) / (2.0 * step)

        return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq
