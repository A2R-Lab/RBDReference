import numpy as np
import copy
np.set_printoptions(precision=4, suppress=True, linewidth = 100)

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj

    def mxS(self, S, vec):
        result = np.zeros((6))
        if not S[0] == 0:
            result += self.mx1(vec,S[0])
        if not S[1] == 0:
            result += self.mx2(vec,S[1])
        if not S[2] == 0:
            result += self.mx3(vec,S[2])
        if not S[3] == 0:
            result += self.mx4(vec,S[3])
        if not S[4] == 0:
            result += self.mx5(vec,S[4])
        if not S[5] == 0:
            result += self.mx6(vec,S[5])
        return result

    def mx1(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[1] = vec[2]*alpha
            vecX[2] = -vec[1]*alpha
            vecX[4] = vec[5]*alpha
            vecX[5] = -vec[4]*alpha
        except:
            vecX[1] = vec[0,2]*alpha
            vecX[2] = -vec[0,1]*alpha
            vecX[4] = vec[0,5]*alpha
            vecX[5] = -vec[0,4]*alpha
        return vecX

    def mx2(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[0] = -vec[2]*alpha
            vecX[2] = vec[0]*alpha
            vecX[3] = -vec[5]*alpha
            vecX[5] = vec[3]*alpha
        except:
            vecX[0] = -vec[0,2]*alpha
            vecX[2] = vec[0,0]*alpha
            vecX[3] = -vec[0,5]*alpha
            vecX[5] = vec[0,3]*alpha
        return vecX

    def mx3(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[0] = vec[1]*alpha
            vecX[1] = -vec[0]*alpha
            vecX[3] = vec[4]*alpha
            vecX[4] = -vec[3]*alpha
        except:
            vecX[0] = vec[0,1]*alpha
            vecX[1] = -vec[0,0]*alpha
            vecX[3] = vec[0,4]*alpha
            vecX[4] = -vec[0,3]*alpha
        return vecX

    def mx4(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[4] = vec[2]*alpha
            vecX[5] = -vec[1]*alpha
        except:
            vecX[4] = vec[0,2]*alpha
            vecX[5] = -vec[0,1]*alpha
        return vecX

    def mx5(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[3] = -vec[2]*alpha
            vecX[5] = vec[0]*alpha
        except:
            vecX[3] = -vec[0,2]*alpha
            vecX[5] = vec[0,0]*alpha
        return vecX

    def mx6(self, vec, alpha = 1.0):
        vecX = np.zeros((6))
        try:
            vecX[3] = vec[1]*alpha
            vecX[4] = -vec[0]*alpha
        except:
            vecX[3] = vec[0,1]*alpha
            vecX[4] = -vec[0,0]*alpha
        return vecX

    def fxv(self, fxVec, timesVec):
        # Fx(fxVec)*timesVec
        #   0  -v(2)  v(1)    0  -v(5)  v(4)
        # v(2)    0  -v(0)  v(5)    0  -v(3)
        #-v(1)  v(0)    0  -v(4)  v(3)    0
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

    def fxS(self, S, vec, alpha = 1.0):
        return -self.mxS(S, vec, alpha)

    def vxIv(self, vec, Imat):
        temp = np.matmul(Imat,vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec

    def rnea_fpass(self, q, qd, qdd = None, GRAVITY = -9.81):
        # allocate memory
        NB = self.robot.get_num_bodies()
        v = np.zeros((6,NB))
        a = np.zeros((6,NB))
        f = np.zeros((6,NB))
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        # forward pass
        for curr_id in range(NB):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            inds_q = self.robot.get_joint_index_q(curr_id)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
            # compute v and a
            if parent_id == -1: # parent is fixed base or world
                # v_base is zero so v[:,ind] remains 0
                a[:,curr_id] = np.matmul(Xmat,gravity_vec)
            else:
                v[:,curr_id] = np.matmul(Xmat,v[:,parent_id])
                a[:,curr_id] = np.matmul(Xmat,a[:,parent_id])
            inds_v = self.robot.get_joint_index_v(curr_id)
            _qd = qd[inds_v]
            vJ = np.matmul(S,np.transpose(np.matrix(_qd)))
            # TODO fix the weird numpy error that is stopping this from
            # ValueError: non-broadcastable output operand with shape (6,) doesn't match the broadcast shape (1,6)
            # just being v[:,curr_id] += vJ
            for j in range(6):
                v[j,curr_id] += vJ[j]
            a[:,curr_id] += self.mxS(vJ,v[:,curr_id])
            if qdd is not None:
                _qdd = qdd[inds_v]
                aJ = np.matmul(S, np.transpose(np.matrix(_qdd)))
                for j in range(6):
                    a[j, curr_id] += aJ[j]
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:,curr_id] = np.matmul(Imat,a[:,curr_id]) + self.vxIv(v[:,curr_id],Imat)

        return (v,a,f)

    def rnea_bpass(self, q, f):
        # allocate memory
        NB = self.robot.get_num_bodies()
        m = self.robot.get_num_vel()
        c = np.zeros(m)

        # backward pass
        for curr_id in range(NB-1,-1,-1):
            parent_id = self.robot.get_parent_id(curr_id)
            S = self.robot.get_S_by_id(curr_id)
            inds_f = self.robot.get_joint_index_f(curr_id)
            # compute c
            c[inds_f] = np.matmul(np.transpose(S),f[:,curr_id])
            # update f if applicable
            if parent_id != -1:
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                temp = np.matmul(np.transpose(Xmat),f[:,curr_id])
                f[:,parent_id] = f[:,parent_id] + temp.flatten()

        return (c,f)

    def rnea(self, q, qd, qdd = None, GRAVITY = -9.81):
        # forward pass
        (v,a,f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c,f) = self.rnea_bpass(q, f)

        return (c,v,a,f)

    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        
        # allocate memory
        n = len(qd)
        dv_dq = np.zeros((6,n,n))
        da_dq = np.zeros((6,n,n))
        df_dq = np.zeros((6,n,n))

        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,v[:,parent_ind]))
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            for c in range(n):
                da_dq[:,c,ind] += self.mxS(S,dv_dq[:,c,ind],qd[ind])
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,a[:,parent_ind]))
            else:
                da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,gravity_vec))
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))

        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
        
        # allocate memory
        n = len(qd)
        dv_dqd = np.zeros((6,n,n))
        da_dqd = np.zeros((6,n,n))
        df_dqd = np.zeros((6,n,n))

        # forward pass
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
            dv_dqd[:,ind,ind] += S
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
            for c in range(n):
                da_dqd[:,c,ind] += self.mxS(S,dv_dqd[:,c,ind],qd[ind])
            da_dqd[:,ind,ind] += self.mxS(S,v[:,ind])
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))

        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
        
        # allocate memory
        n = len(q) # assuming len(q) = len(qd)
        dc_dq = np.zeros((n,n))
        
        for ind in range(n-1,-1,-1):
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            dc_dq[ind,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                for entry in range(6):
                    df_dq[entry,ind,parent_ind] += delta_dq[entry]

        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd):
        
        # allocate memory
        n = len(q) # assuming len(q) = len(qd)
        dc_dqd = np.zeros((n,n))
        
        for ind in range(n-1,-1,-1):
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            dc_dqd[ind,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

        return dc_dqd

    def rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81):
        (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)

        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd)

        dc_du = np.hstack((dc_dq,dc_dqd))
        return dc_du

    def minv_bpass(self, q):

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()

        Minv = np.zeros((n,n))
        F = np.zeros((n,6,n))
        U = np.zeros((n,6))
        Dinv = {}

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

        # backward pass
        for curr_id in range(NB-1,-1,-1):
            # Compute U, D
            S = self.robot.get_S_by_id(curr_id)
            inds_v = self.robot.get_joint_index_v(curr_id)
            subtreeInds = self.robot.get_subtree_by_id(curr_id)
            temp = np.matmul(IA[curr_id],S)

            if curr_id == 0 and self.robot.floating_base:
                U[0:6,:] = temp
                Dinv[curr_id] = 1/np.matmul(S.transpose(),U[0:6,:])
                # Update Minv
                Minv[0:6,0:6] = Dinv[curr_id]
                # TODO WHAT IS UP WITH THE SIZING HERE?!?!?!
                for subInd in subtreeInds:
                    for row in range(S.shape[1]):
                        Minv[0:6,subInd] -= np.matmul(Dinv[curr_id],np.matmul(S.transpose()[row,:],F[0:6,:,subInd]))      
            else:
                # TODO again numpy?!?
                # ValueError: non-broadcastable output operand with shape (6,) doesn't match the broadcast shape (1,6)
                for j in range(6):
                    U[inds_v,j] = temp[j]
                Dinv[curr_id] = 1/np.matmul(S.transpose(),U[inds_v,:])
                # Update Minv
                Minv[inds_v] = Dinv[curr_id]
                for subInd in subtreeInds:
                    Minv[inds_v,subInd] -= Dinv[curr_id] * np.matmul(S.transpose(),F[inds_v,:,subInd])
                
            # update parent if applicable
            parent_id = self.robot.get_parent_id(curr_id)
            if parent_id != -1:
                parent_ind = self.robot.get_joint_index_v(parent_id)
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                # update F
                for subInd in subtreeInds:
                    F[inds_v,:,subInd] += U[inds_v,:]*Minv[inds_v,subInd]
                    F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[inds_v,:,subInd]) 
                # update IA
                Ia = IA[curr_id] - np.outer(U[inds_v,:],Dinv[curr_id]*U[inds_v,:])
                IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                IA[parent_id] += IaParent

        print(Minv)
        return (Minv, F, U, Dinv)

    def minv_fpass(self, q, Minv, F, U, Dinv):
        NB = self.robot.get_num_bodies()

        # forward pass
        for curr_id in range(NB):
            inds_v = self.robot.get_joint_index_v(curr_id)
            inds_q = self.robot.get_joint_index_q(curr_id)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
            S = self.robot.get_S_by_id(curr_id)
            parent_id = self.robot.get_parent_id(curr_id)
            parent_inds = self.robot.get_joint_index_v(parent_id)

            if parent_id != -1:
                UX = np.matmul(U[inds_v,:].transpose(),Xmat)
                if parent_id == 0 and self.robot.floating_base:
                    Minv[inds_v,:] = 0
                    for parent_ind in parent_inds:
                        temp = Dinv[curr_id]*np.matmul(UX,F[parent_ind,:,inds_v:])
                        # Minv[inds_v,inds_v:] += temp
                        for j in range(Minv.shape[0]-inds_v):
                            Minv[inds_v,inds_v+j] = temp[0,j]
                else:
                    # TODO Numpy again!!!!
                    temp = Dinv[curr_id]*np.matmul(UX,F[parent_inds,:,inds_v:])
                    # Minv[inds_v,inds_v:] += temp
                    for j in range(Minv.shape[0]-inds_v):
                        Minv[inds_v,inds_v+j] = temp[0,j]

            if curr_id == 0 and self.robot.floating_base:
                for j in range(6):
                    F[0:6,j,:] = Minv[0:6,:]
            else:
                F[inds_v,:,inds_v:] = np.outer(S,Minv[inds_v,inds_v:])
            
            if parent_id != -1:
                if parent_id == 0 and self.robot.floating_base:
                    for parent_ind in parent_inds:
                        F[inds_v,:,inds_v:] += np.matmul(Xmat,F[parent_ind,:,inds_v:])
                else:
                    F[inds_v,:,inds_v:] += np.matmul(Xmat,F[parent_inds,:,inds_v:])

        return Minv

    def minv(self, q, output_dense = True):
        # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        """ Computes the analytical inverse of the joint space inertia matrix
        CRBA calculates the joint space inertia matrix H to represent the composite inertia
        This is used in the fundamental motion equation H qdd + C = Tau
        Forward dynamics roughly calculates acceleration as H_inv ( Tau - C); analytic inverse - benchmark against Pinocchio
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
                        Minv[row,col] = Minv[col,row]

        return Minv
    
    def test_minv(self, q, output_dense = True):
        
        # # Backward Pass
        n = len(q)
        NB = self.robot.get_num_bodies()
        Minv = np.zeros((n,n))
        F = np.zeros((n,6,n))
        U = np.zeros((n,6))
        Dinv = np.zeros(n)

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())
        # backward pass
        for ind in range(n-1,-1,-1):
            # Compute U, D
            S = self.robot.get_S_by_id(ind) # NOTE What is S for floating base?
            subtreeInds = self.robot.get_subtree_by_id(ind)
            U[ind,:] = np.matmul(IA[ind],S)
            Dinv[ind] = 1/np.matmul(S.transpose(),U[ind,:])
            # Update Minv
            Minv[ind,ind] = Dinv[ind]
            for subInd in subtreeInds:
                Minv[ind,subInd] -= Dinv[ind] * np.matmul(S.transpose(),F[ind,:,subInd])
            # update parent if applicable
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                # update F
                for subInd in subtreeInds:
                    F[ind,:,subInd] += U[ind,:]*Minv[ind,subInd]
                    F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[ind,:,subInd]) 
                # update IA
                Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
                IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                IA[parent_ind] += IaParent

        # # Original backward pass
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())

        # backward pass
        for curr_id in range(NB-1,-1,-1):
            # Compute U, D
            S = self.robot.get_S_by_id(curr_id)
            inds_v = self.robot.get_joint_index_v(curr_id)
            subtreeInds = self.robot.get_subtree_by_id(curr_id)
            temp = np.matmul(IA[curr_id],S)

            if curr_id == 0 and self.robot.floating_base:
                U[0:6,:] = temp
                Dinv[curr_id] = 1/np.matmul(S.transpose(),U[0:6,:])
                # Update Minv
                Minv[0:6,0:6] = Dinv[curr_id]
                # TODO WHAT IS UP WITH THE SIZING HERE?!?!?!
                for subInd in subtreeInds:
                    for row in range(S.shape[1]):
                        Minv[0:6,subInd] -= np.matmul(Dinv[curr_id],np.matmul(S.transpose()[row,:],F[0:6,:,subInd]))      
            else:
                # TODO again numpy?!?
                # ValueError: non-broadcastable output operand with shape (6,) doesn't match the broadcast shape (1,6)
                for j in range(6):
                    U[inds_v,j] = temp[j]
                Dinv[curr_id] = 1/np.matmul(S.transpose(),U[inds_v,:])
                # Update Minv
                Minv[inds_v] = Dinv[curr_id]
                for subInd in subtreeInds:
                    Minv[inds_v,subInd] -= Dinv[curr_id] * np.matmul(S.transpose(),F[inds_v,:,subInd])
                
            # update parent if applicable
            parent_id = self.robot.get_parent_id(curr_id)
            if parent_id != -1:
                parent_ind = self.robot.get_joint_index_v(parent_id)
                inds_q = self.robot.get_joint_index_q(curr_id)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
                # update F
                for subInd in subtreeInds:
                    F[inds_v,:,subInd] += U[inds_v,:]*Minv[inds_v,subInd]
                    F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[inds_v,:,subInd]) 
                # update IA
                Ia = IA[curr_id] - np.outer(U[inds_v,:],Dinv[curr_id]*U[inds_v,:])
                IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                IA[parent_id] += IaParent

        # # Forward Pass

        # forward pass
        for curr_id in range(NB):
            inds_v = self.robot.get_joint_index_v(curr_id)
            inds_q = self.robot.get_joint_index_q(curr_id)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(curr_id)(_q)
            S = self.robot.get_S_by_id(curr_id)
            parent_id = self.robot.get_parent_id(curr_id)
            parent_inds = self.robot.get_joint_index_v(parent_id)

            if parent_id != -1:
                UX = np.matmul(U[inds_v,:].transpose(),Xmat)
                if parent_id == 0 and self.robot.floating_base:
                    Minv[inds_v,:] = 0
                    for parent_ind in parent_inds:
                        temp = Dinv[curr_id]*np.matmul(UX,F[parent_ind,:,inds_v:])
                        # Minv[inds_v,inds_v:] += temp
                        for j in range(Minv.shape[0]-inds_v):
                            Minv[inds_v,inds_v+j] = temp[0,j]
                else:
                    # TODO Numpy again!!!!
                    temp = Dinv[curr_id]*np.matmul(UX,F[parent_inds,:,inds_v:])
                    # Minv[inds_v,inds_v:] += temp
                    for j in range(Minv.shape[0]-inds_v):
                        Minv[inds_v,inds_v+j] = temp[0,j]

            if curr_id == 0 and self.robot.floating_base:
                for j in range(6):
                    F[0:6,j,:] = Minv[0:6,:]
            else:
                F[inds_v,:,inds_v:] = np.outer(S,Minv[inds_v,inds_v:])
            
            if parent_id != -1:
                if parent_id == 0 and self.robot.floating_base:
                    for parent_ind in parent_inds:
                        F[inds_v,:,inds_v:] += np.matmul(Xmat,F[parent_ind,:,inds_v:])
                else:
                    F[inds_v,:,inds_v:] += np.matmul(Xmat,F[parent_inds,:,inds_v:])

        return Minv