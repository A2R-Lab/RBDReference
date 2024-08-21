import numpy as np
import copy
np.set_printoptions(precision=4, suppress=True, linewidth=100)

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj

    def cross_operator(self, v):
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
        #(crf in in spatial_v2_extended)
        return(-1 * self.cross_operator(v).T)
    
    def icrf(self, v):
        #helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asmatrix(res)
    
    def factor_functions(self, I, v, number=3):
        # helper function defined in spatial_v2_extended library, called by idsva() and rnea_grad()
        if number == 1:
            B = self.dual_cross_operator(v) * I
        elif number == 2:
            B = self.icrf(np.matmul(I,v)) - I * self.cross_operator(v)
        else:
            B = 1/2 * (np.matmul(self.dual_cross_operator(v),I) + self.icrf(np.matmul(I,v)) - np.matmul(I, self.cross_operator(v)))

        return B

    def _mxS(self, S, vec, alpha=1.0):
        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        return np.squeeze(np.array((alpha * np.dot(self.cross_operator(vec), S)))) # added np.squeeze and np.array

    def mxS(self, S, vec):
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
        vecX = np.zeros((6))
        try:
            vecX[4] = vec[2] * alpha
            vecX[5] = -vec[1] * alpha
        except:
            vecX[4] = vec[0, 2] * alpha
            vecX[5] = -vec[0, 1] * alpha
        return vecX

    def mx5(self, vec, alpha=1.0):
        vecX = np.zeros((6))
        try:
            vecX[3] = -vec[2] * alpha
            vecX[5] = vec[0] * alpha
        except:
            vecX[3] = -vec[0, 2] * alpha
            vecX[5] = vec[0, 0] * alpha
        return vecX

    def mx6(self, vec, alpha=1.0):
        vecX = np.zeros((6))
        try:
            vecX[3] = vec[1] * alpha
            vecX[4] = -vec[0] * alpha
        except:
            vecX[3] = vec[0, 1] * alpha
            vecX[4] = -vec[0, 0] * alpha
        return vecX

    def fxv(self, fxVec, timesVec):
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
        return -self._mxS(S, vec, alpha) #changed to _mxS

    def vxIv(self, vec, Imat):
        temp = np.matmul(Imat, vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec
    
    def apply_external_forces(self, q, f_in, f_ext):
        """ Implementation based on spatial v2: https://github.com/ROAM-Lab-ND/spatial_v2_extended/blob/main/dynamics/apply_external_forces.m
        
        Subtracts external forces from input f_in. 

        Parameters:
        - f_in (numpy.ndarray): Initial forces applied to links. 
        - f_ext (numpy.ndarray): The external force.

        Returns:
        - f_out (numpy.ndarray): The updated force.
        TODO Check the correct way to index the forces!
        """
        f_out = f_in
        if len(f_ext) > 0:
            for curr_id in range(len(f_ext)):
                parent_id = self.robot.get_parent_id(curr_id)
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                if parent_id == -1:
                    Xa = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                else:
                    Xa = np.matmul(self.robot.get_Xmat_Func_by_id(curr_id)(curr_id),Xa) 
                if len(f_ext[:, curr_id]) > 0:
                    f_out[:, curr_id] -= np.matmul(np.linalg.inv(Xa.T), f_ext[:, curr_id])
        return f_out

    def rnea_fpass(self, q, qd, qdd=None, GRAVITY=-9.81):
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
                a[:, curr_id] = np.matmul(Xmat, gravity_vec)
            else:
                v[:, curr_id] = np.matmul(Xmat, v[:, parent_id])
                a[:, curr_id] = np.matmul(Xmat, a[:, parent_id])
            inds_v = self.robot.get_joint_index_v(curr_id)
            _qd = qd[inds_v]
            vJ = np.matmul(S, np.transpose(np.matrix(_qd)))
            v[:, curr_id] += np.squeeze(np.array(vJ))  # reduces shape to (6,) matching v[:,curr_id]
            a[:, curr_id] += self.mxS(vJ, v[:, curr_id])
            if qdd is not None:
                _qdd = qdd[inds_v]
                aJ = np.matmul(S, np.transpose(np.matrix(_qdd)))
                a[:, curr_id] += np.squeeze(np.array(aJ))  # reduces shape to (6,) matching a[:,curr_id]
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:, curr_id] = np.matmul(Imat, a[:, curr_id]) + self.vxIv(v[:, curr_id], Imat)

        return (v, a, f)

    def rnea_bpass(self, q, f):
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

    def rnea(self, q, qd, qdd=None, GRAVITY=-9.81, f_ext=None):
        # forward pass
        (v, a, f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c, f) = self.rnea_bpass(q, f)

        return (c, v, a, f)

    def rnea_grad_fpass(self, q, qd, qdd, GRAVITY=-9.81):
        """
        Computes bpass according to dq for the RNEA gradient algorithm.

        Parameters:
        - q: numpy.ndarray
            Joint positions.
        - qd: numpy.ndarray
            Joint velocities.
        - qdd: numpy.ndarray
            Joint accelerations.

        Returns:
        - f: numpy.ndarray
            Spatial force on each link.
        - IC: numpy.ndarray
            Composite spatial inertia of the subtree rooted at each joint.
        - BC: numpy.ndarray
            Bias force matrix.
        - S_: numpy.ndarray
            Motion subspace matrix. 
        - Sd: numpy.ndarray
            The spatial derivative of S_.
        - Sdd: numpy.ndarray
            The spatial jerk, or second derivative calculation of S_.
        - Sj: numpy.ndarray
            Another derivative of the motion subspace matrix S.
        """
        # allocate memory 
        NB = self.robot.get_num_bodies()
        v = np.zeros((6, NB))
        a = np.zeros((6, NB))
        f = np.zeros((6, NB))
        IC = np.zeros((NB,6,6))
        BC = np.zeros((NB,6,6))
        S_ = {} 
        Sd = {}
        Sdd = {}
        Sj = {}
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
                a[:, curr_id] = gravity_vec
                Xup0 = Xmat
            else:
                v[:, curr_id] = v[:, parent_id]
                a[:, curr_id] = a[:, parent_id]
                Xup0 = np.matmul(Xmat, Xup0)

            inds_v = self.robot.get_joint_index_v(curr_id)
            _qd = np.matrix(qd[inds_v])
            _qdd = np.matrix(qdd[inds_v])

            Xdown = np.linalg.inv(Xup0)
            S_[curr_id] = np.matmul(Xdown, S)

            if parent_id == -1:
                vJ = np.matmul(S_[curr_id], _qd.T)
                aJ = np.matmul(self.cross_operator(v[:, curr_id]), vJ) + np.matmul(S_[curr_id], _qdd.T)
            else:
                vJ = S_[curr_id] * _qd
                aJ = np.matmul(self.cross_operator(v[:, curr_id]), vJ) + S_[curr_id] * _qdd.T
            
            Sd[curr_id] = np.matmul(self.cross_operator(v[:, curr_id]), S_[curr_id])
            Sdd[curr_id] = np.matmul(self.cross_operator(a[:, curr_id]),S_[curr_id]) + np.matmul(self.cross_operator(v[:, curr_id]),Sd[curr_id])
            Sj[curr_id] = 2 * Sd[curr_id] + np.matmul(self.cross_operator(np.squeeze(np.array(vJ))),S_[curr_id])

            v[:, curr_id] += np.squeeze(np.array(vJ))
            a[:, curr_id] += np.squeeze(np.array(aJ))

            Imat = self.robot.get_Imat_by_id(curr_id)
            IC[curr_id] = np.matmul(Xup0.T,np.matmul(Imat,Xup0))
            BC[curr_id] = 2 * self.factor_functions(IC[curr_id], v[:,curr_id])
            f[:, curr_id] = np.matmul(IC[curr_id], a[:, curr_id]) + self.vxIv(v[:, curr_id], IC[curr_id])
    
        return (f, IC, BC, S_, Sd, Sdd, Sj)

    def rnea_grad_bpass(self, f, IC, BC, S_, Sd, Sdd, Sj):
        """
        Computes bpass according to dq for the RNEA gradient algorithm.

        Parameters:
        - f: numpy.ndarray
            Spatial force on each link.
        - IC: numpy.ndarray
            Composite spatial inertia of the subtree rooted at each joint.
        - BC: numpy.ndarray
            Bias force matrix.
        - S_: numpy.ndarray
            Motion subspace matrix. 
        - Sd: numpy.ndarray
            The spatial derivative of S_.
        - Sdd: numpy.ndarray
            The spatial jerk, or second derivative calculation of S_.
        - Sj: numpy.ndarray
            Another derivative of the motion subspace matrix.

        Returns:
        - dc_dq: numpy.ndarray
            The derivative of the bpass with respect to the generalized coordinates.
        - dc_dqd: numpy.ndarray
            The derivative of the bpass with respect to the generalized velocities.
        """
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()
        dc_dq = np.zeros((n, n))
        dc_dqd = np.zeros((n, n))
        tmp1 = np.zeros((6,n))
        tmp2 = np.zeros((6,n))
        tmp3 = np.zeros((6,n))
        tmp4 = np.zeros((6,n))

        for curr_id in range(NB-1, -1, -1): # try 0 instead of -1
            parent_id = self.robot.get_parent_id(curr_id)
            if self.robot.floating_base:
                if parent_id != -1:
                    ind = curr_id + 5 # special matrix indexing for floating base
                else: 
                    ind = [0,1,2,3,4,5] # first 6 indices for floating base
            else:
                ind = curr_id
            # chain rule temp vars for intermediate derivative calculations 
            tmp1[:,ind] = np.squeeze(np.array(np.matmul(IC[curr_id], S_[curr_id])))
            tmp2[:,ind] = np.squeeze(np.array(np.matmul(BC[curr_id], S_[curr_id]) + np.matmul(IC[curr_id], Sj[curr_id])))
            tmp3[:,ind] = np.squeeze(np.array(np.matmul(BC[curr_id], Sd[curr_id]) + np.matmul(IC[curr_id], Sdd[curr_id]) + np.matmul(self.icrf(f[:,curr_id]), S_[curr_id])))
            tmp4[:,ind] = np.squeeze(np.array(np.matmul(BC[curr_id].T, S_[curr_id])))

            # adjust subtree inds for modified FB joint indexing
            subtreeInds = self.robot.get_subtree_by_id(curr_id)
            if self.robot.floating_base:
                adj_subtreeInds = list(np.array(subtreeInds) + 5)
            else:
                adj_subtreeInds = subtreeInds

            if self.robot.floating_base and parent_id == -1:
                # dc_dq calculation
                dc_dq[ind,:] = np.matmul(S_[curr_id].T, tmp3[:,:])
                dc_dq[:,ind] = (np.matmul(tmp1[:,:].T, Sdd[curr_id])  + np.matmul(tmp4[:,:].T,Sd[curr_id]))
                
                # dc_dqd calculation 
                dc_dqd[ind,:] = np.matmul(S_[curr_id].T, tmp2[:,:])
                dc_dqd[:,ind] = (np.matmul(tmp1[:,:].T, Sj[curr_id]) + np.matmul(tmp4[:,:].T,S_[curr_id]))
            else:
                # dc_dq calculation 
                dc_dq[ind,adj_subtreeInds] = np.matmul(S_[curr_id].T, tmp3[:,adj_subtreeInds])
                dc_dq[adj_subtreeInds,ind] = np.squeeze(np.array((np.matmul(tmp1[:,adj_subtreeInds].T, Sdd[curr_id]) 
                                            + np.matmul(tmp4[:,adj_subtreeInds].T,Sd[curr_id]))))
                
                # dc_dqd calculation 
                dc_dqd[ind,adj_subtreeInds] = np.matmul(S_[curr_id].T, tmp2[:,adj_subtreeInds])
                dc_dqd[adj_subtreeInds,ind] = np.squeeze(np.array((np.matmul(tmp1[:,adj_subtreeInds].T, Sj[curr_id]) 
                                            + np.matmul(tmp4[:,adj_subtreeInds].T,S_[curr_id]))))
                
                # update forces 
                IC[parent_id] += IC[curr_id]
                BC[parent_id] += BC[curr_id]
                f[:, parent_id] += f[:, curr_id]    

        return dc_dq, dc_dqd
    
    def rnea_grad(self, q, qd, qdd=None, GRAVITY=-9.81):
        """
        Implementation based on spatial v2: https://github.com/ROAM-Lab-ND/spatial_v2_extended/blob/main/v3/derivatives/ID_derivatives.m
        Computes the gradient of the RNEA algorithm with respect to q, qd, and qdd.
        Outputs the gradient of RNEA torque tau (c) with respect to q and qd.
        """

        # Compute the forward pass, 
        (f, IC, BC, S_, Sd, Sdd, Sj) = self.rnea_grad_fpass(q, qd, qdd, GRAVITY)

        # Compute the bpass, dq, dqd
        (dc_dq, dc_dqd) = self.rnea_grad_bpass(f, IC, BC, S_, Sd, Sdd, Sj)
        
        # dc_du = np.hstack((dc_dq, dc_dqd))
        return dc_dq, dc_dqd

    def minv_bpass(self, q):
        """
        Performs the backward pass of the Minv algorithm.

        NOTE:
        If floating base, treat floating base joint as 6 joints (Px,Py,Pz,Rx,Ry,Rz) where P=prismatic R=Revolute.
        Thus, allocate memroy and assign a "matrix_ind" shifting indices to match 6 joint representation.
        At the end of bpass at floating_base joint, 6 loop pass treating floating base joint as 6 joints.

        Args:
            q (numpy.ndarray): The joint positions.

        Returns:
            tuple: A tuple containing the following arrays:
            - Minv (numpy.ndarray): Analytical inverse of the joint space inertia matrix.
            - F (numpy.ndarray): The joint forces.
            - U (numpy.ndarray): The joint velocities multiplied by the inverse mass matrix.
            - Dinv (numpy.ndarray): The inverse diagonal elements of the mass matrix.
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
        """
        Performs a forward pass to compute the inverse mass matrix Minv.

        NOTE:
        If Floating base, treat floating base joint as 6 joints (Px,Py,Pz,Rx,Ry,Rz) where P=prismatic R=Revolute.
        Thus, allocate memroy and assign a "matrix_ind" shifting indices to match 6 joint representation.
        See Spatial_v2_extended algorithm for alterations to fpass algorithm.
        Additionally, made convenient shift to F[i] accessing based on matrix structure in math.

        Args:
            q (numpy.ndarray): The joint positions.
            Minv (numpy.ndarray): The inverse mass matrix.
            F (numpy.ndarray): The spatial forces.
            U (numpy.ndarray): The joint velocity transformation matrix.
            Dinv (numpy.ndarray): The inverse diagonal inertia matrix.

        Returns:
            numpy.ndarray: The updated inverse mass matrix Minv.
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

    def minv(self, q, output_dense=True):
        # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        """Computes the analytical inverse of the joint space inertia matrix
        CRBA calculates the joint space inertia matrix H to represent the composite inertia
        This is used in the fundamental motion equation H qdd + C = Tau
        Forward dynamics roughly calculates acceleration as H_inv ( Tau - C); analytic inverse - benchmark against Matlab spatial v2
        """
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

        return Minv
    
    def aba(self, q, qd, tau, f_ext=None, GRAVITY = -9.81):
        """
        Compute the Articulated Body Algorithm (ABA) to calculate the joint accelerations.
        """
        # allocate memory TODO check NB vs. n
        n = len(qd)
        NB = self.robot.get_num_bodies()
        v = np.zeros((6,NB))
        c = np.zeros((6,NB))
        a = np.zeros((6,NB))
        IA = np.zeros((6,6,NB))
        pA = np.zeros((6,NB))
        # variables may require special indexing
        f = np.zeros((6,n))
        # d = np.zeros(n)
        d = {}
        U = np.zeros((6,n))
        u = np.zeros(n)
        qdd = np.zeros(n)

        gravity_vec = np.zeros((6))
        gravity_vec[5] = GRAVITY  # a_base is gravity vec

        # Initial Forward Pass
        for ind in range(NB): # curr_id = ind for this loop
            parent_ind = self.robot.get_parent_id(ind)
            _q = q[self.robot.get_joint_index_q(ind)]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)

            if parent_ind == -1: # parent is base
                v[:, ind] = np.matmul(S, qd[ind:ind+6])
            else:
                v[:, ind] = np.matmul(Xmat, v[:, parent_ind]) 
                vJ = np.squeeze(np.array(S * qd[ind])) # reduces shape to (6,) matching v[:,curr_id]
                v[:, ind] += vJ
                c[:, ind] = np.matmul(self.cross_operator(v[:, ind]), vJ)
            
            Imat = self.robot.get_Imat_by_id(ind)

            IA[:,:,ind] = Imat

            vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
            [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
            [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
            [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
            [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
            [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

            crf = -np.transpose(vcross) 
            temp = np.matmul(crf, Imat)

            pA[:, ind] = np.matmul(temp, v[:, ind])[0]

        # Backward Pass
        for curr_id in range(NB-1, -1, -1): # curr_id != ind for bpass
            S = self.robot.get_S_by_id(curr_id)
            parent_ind = self.robot.get_parent_id(curr_id)

            if self.robot.floating_base:
                if curr_id == 0:
                    ind = [0,1,2,3,4,5]
                else:
                    ind = curr_id + 5

            U[:, ind] = np.squeeze(np.matmul(IA[:,:,curr_id], S))
            # print(S.shape)
            # print(U[:, ind].shape)
            d[curr_id] = np.matmul(np.transpose(S), U[:, ind])
            u[ind] = tau[ind] - np.matmul(np.transpose(S), pA[:, curr_id])

            if parent_ind != -1:

                rightSide = np.reshape(U[:, ind], (6,1)) @ np.reshape(U[:, ind], (6,1)).T / d[curr_id]
                Ia = IA[:,:, curr_id] - rightSide
                
                pa = pA[:, curr_id] + np.matmul(Ia, c[:, curr_id]) + U[:, ind] * u[ind] / d[curr_id]

                _q = q[self.robot.get_joint_index_q(curr_id)]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                temp = np.matmul(np.transpose(Xmat), Ia)

                IA[:, :, parent_ind] = IA[:, :, parent_ind] + np.matmul(temp, Xmat)

                # temp = np.matmul(np.transpose(Xmat), pa)
                temp = np.matmul(Xmat, np.transpose(pa))
                pA[:, parent_ind] = pA[:, parent_ind] + temp.flatten()
        
        # Final Forward Pass
        for curr_id in range(NB): # curr_id != ind for bpass
            parent_ind = self.robot.get_parent_id(curr_id)
            _q = q[self.robot.get_joint_index_q(curr_id)]
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)

            if self.robot.floating_base:
                if curr_id == 0:
                    ind = [0,1,2,3,4,5]
                else:
                    ind = curr_id + 5

            if parent_ind == -1: # parent is base
                a[:, curr_id] = np.matmul(Xmat, gravity_vec) + c[:, curr_id]
            else:
                a[:, curr_id] = np.matmul(Xmat, a[:, parent_ind]) + c[:, curr_id]
            
            S = self.robot.get_S_by_id(curr_id)
            temp = u[ind] - np.matmul(np.transpose(U[:, ind]), a[:, curr_id])

            # TODO come back and check matlab indexing referencingt Xup and alter qdd *S to be S *qdd
            if parent_ind == -1:
                qdd[ind] = np.matmul(np.linalg.inv(d[curr_id]), temp) #TODO fix / check for singularities
                a[:, curr_id] = a[:, curr_id] + np.matmul(qdd[ind],S)
            else:
                qdd[ind] = 1 / d[curr_id] * temp
                a[:, curr_id] = a[:, curr_id] + np.matmul(qdd[ind],S)
        
        return qdd




    def crba(self, q, qd):
        """
        Computes the Composite Rigid Body Algorithm (CRBA) to calculate the joint-space inertia matrix.
        # Based on Featherstone implementation of CRBA p.182 in rigid body dynamics algorithms book.

        NOTE:
        If Floating base, treat floating base joint as 6 joints (Px,Py,Pz,Rx,Ry,Rz) where P=prismatic R=Revolute.
        Thus, allocate memroy and assign a "matrix_ind" shifting indices to match 6 joint representation.
        Propagate changes to any indexing of j, F, U, Dinv, Minv, etc. to match 6 joint representation.

        Parameters:
        - q (numpy.ndarray): Joint positions.
        - qd (numpy.ndarray): Joint velocities.

        Returns:
        - H (numpy.ndarray): Joint-space inertia matrix.
        """
        if self.robot.floating_base:
            NB = self.robot.get_num_bodies()
            n = len(qd)
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
                    H[ind:6, ind:6] = np.matmul(S.T, fh)
        else:
            # # Fixed base implmentation of CRBA
            n = len(qd)
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
                    fh = np.matmul(Xmat.T, fh)
                    j = self.robot.get_parent_id(j)
                    S = self.robot.get_S_by_id(j)
                    H[ind, j] = np.matmul(S.T, fh)
                    H[j, ind] = H[ind, j]

        return H

##### Testing original RNEA_grad to help with CUDA 
    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = len(qd)
        dv_dq = np.zeros((6,n,n))
        da_dq = np.zeros((6,n,n))
        df_dq = np.zeros((6,n,n))

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
                dv_dq[:,:,idx] = np.matmul(Xmat,dv_dq[:,:,parent_idx])
                dv_dq[:,idx,idx] += self._mxS(S,np.matmul(Xmat,v[:,parent_ind])) # replace with new mxS
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,idx] = np.matmul(Xmat,da_dq[:,:,parent_idx])
            for c in range(n):
                # TODO compress the if condition to handle 6x6 in one step
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        da_dq[:,c,ii] += self._mxS(S[ii],dv_dq[:,c,ii],qd[ind])
                else:
                    da_dq[:,c,idx] += self._mxS(S,dv_dq[:,c,idx],qd[ind]) # replace with new mxS
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,idx,idx] += self._mxS(S,np.matmul(Xmat,a[:,parent_ind])) # replace with new mxS
            else:
                da_dq[:,idx,idx] += self._mxS(S,np.matmul(Xmat,gravity_vec)) # replace with new mxS
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            #TODO compress if condition for loop 
            if parent_ind == -1 and self.robot.floating_base:
                for ii in range(len(idx)):
                    df_dq[:,:,ii] = np.matmul(Imat, da_dq[:,:,ii])
            else:
                df_dq[:,:,idx] = np.matmul(Imat,da_dq[:,:,idx])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        df_dq[:,c,ii] += self.fxv(dv_dq[:,c,ii],Iv)
                        df_dq[:,c,ii] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ii]))
                else:
                    df_dq[:,c,idx] += self.fxv(dv_dq[:,c,idx],Iv)
                    df_dq[:,c,idx] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,idx]))

        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
        """
        Performs the forward pass of the Recursive Newton-Euler Algorithm (RNEA) for gradient computation with respect to qd.

        Args:
            q (np.ndarray): The joint positions.
            qd (np.ndarray): The joint velocities.
            v (6,NB) (np.ndarray): The body spatial velocities.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the gradient of spatial acceleration (dv_dqd), 
            gradient of spatial force (da_dqd), and gradient of spatial force derivative (df_dqd) with respect to qd.
        """
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = len(qd)
        dv_dqd = np.zeros((6,n,n))
        da_dqd = np.zeros((6,n,n))
        df_dqd = np.zeros((6,n,n))

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
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,idx] = np.matmul(Xmat,dv_dqd[:,:,parent_idx])
            dv_dqd[:,idx,idx] += np.squeeze(np.array(S)) # added squeeze 
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,idx] = np.matmul(Xmat,da_dqd[:,:,parent_idx])
            for c in range(NB): # TODO check here and test
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        da_dqd[:,c,ii] += self._mxS(S[ii],dv_dqd[:,c,ii],qd[ind])
                else:
                    da_dqd[:,c,idx] += self._mxS(S,dv_dqd[:,c,idx],qd[ind]) 

            da_dqd[:,idx,idx] += self._mxS(S,v[:,ind]) 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            if parent_ind == -1 and self.robot.floating_base:
                for ii in range(len(idx)):
                    df_dqd[:,:,ii] = np.matmul(Imat, da_dqd[:,:,ii])
            else:
                df_dqd[:,:,idx] = np.matmul(Imat,da_dqd[:,:,idx])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(idx)):
                        df_dqd[:,c,ii] += self.fxv(dv_dqd[:,c,ii],Iv)
                        df_dqd[:,c,ii] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ii]))
                else:
                    df_dqd[:,c,idx] += self.fxv(dv_dqd[:,c,idx],Iv)
                    df_dqd[:,c,idx] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,idx]))

        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
        
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
            # TODO Can this calculation be done in one line?
            if parent_ind == -1 and self.robot.floating_base:
                for ii in range(len(idx)):
                    dc_dq[ii,:] = np.matmul(np.transpose(S[ii]),df_dq[:,:,ii])
            else:
                dc_dq[idx,:]  = np.matmul(np.transpose(S),df_dq[:,:,idx]) 
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            if parent_ind != -1:
                # Xmat access sequence
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dq[:,:,parent_idx] += np.matmul(np.transpose(Xmat),df_dq[:,:,idx])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                for entry in range(6):
                    df_dq[entry,idx,parent_idx] += delta_dq[entry]

        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
        
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
            if parent_ind == -1 and self.robot.floating_base:
                for ii in range(len(idx)):
                    dc_dqd[ii,:] = np.matmul(np.transpose(S[ii]),df_dqd[:,:,ii])
            else:
                dc_dqd[idx,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            if parent_ind != -1:
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dqd[:,:,parent_idx] += np.matmul(np.transpose(Xmat),df_dqd[:,:,idx]) 

        # add in the damping and simplify this expression later
        # suggestion: have a getter function that automatically indexes and allocates for floating base functions
        if USE_VELOCITY_DAMPING:
            for ind in range(NB):
                if self.robot.floating_base and self.robot.get_parent_id(ind) == -1:
                    dc_dqd[ind:ind+5, ind:ind+5] += self.robot.get_damping_by_id(ind)
                else:
                    dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)

        return dc_dqd

    def rnea_grad_original(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
        # instead of passing in trajectory, what if we want our planning algorithm to solve for the optimal trajectory?
        """
        The gradients of inverse dynamics can be very extremely useful inputs into trajectory optimization algorithmss.
        Input: trajectory, including position, velocity, and acceleration
        Output: Computes the gradient of joint forces with respect to the positions and velocities. 
        """ 
        
        (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)
        print(f"df/dq {df_dq.shape}")
        print(df_dq)
        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        dc_du = np.hstack((dc_dq,dc_dqd))
        return dc_du

        