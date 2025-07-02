import numpy as np
import copy

np.set_printoptions(precision=4, suppress=True, linewidth=100)

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj # instance of Robot Object class created by URDFparser``

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
    
    def fxv_simple(self, m, f):
        # force spatial vector cross product. 
        return(np.dot(self.dual_cross_operator(m), f))

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
        # force spatial corss product with motion subspace
        return -self._mxS(S, vec, alpha) #changed from mxS to _mxS
      
    def vxIv(self, vec, Imat):
        # necessary component in differentiating Iv (product rule).
        # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
        # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
        temp = np.matmul(Imat,vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec
    
    """
    End Effector Posiitons

    offests is an array of np matricies of the form (offset_x, offset_y, offset_z, 1)

    TODO: Add and test floating base support.
    """

    def end_effector_pose(self, q, offsets = [np.matrix([[0,0,0,1]])]):

        # do for each branch in the chain
        eePos_arr = []
        for jid in self.robot.get_leaf_nodes():
            
            # chain up the transforms (version 1 for starting from the root)
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)
            # then chain them up
            Xmat_hom = np.eye(4)
            for ind in jidChain:
                currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                Xmat_hom = np.matmul(Xmat_hom,currX)

            # chain up the transforms (version 2 for starting from the leaf)
            currId = jid
            Xmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
            currId = self.robot.get_parent_id(currId)
            while(currId != -1):
                currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                Xmat_hom = np.matmul(currX,Xmat_hom)
                currId = self.robot.get_parent_id(currId)

            # Then extract the end-effector position with the given offset(s)
            # TODO update for multiple offsets

            # xyz position is easy
            eePos_xyz1 = Xmat_hom * offsets[0].transpose()

            # roll pitch yaw is a bit more difficult
            eePos_roll = np.arctan2(Xmat_hom[2,1],Xmat_hom[2,2])
            pitch_temp = np.sqrt(Xmat_hom[2,2]*Xmat_hom[2,2] + Xmat_hom[2,1]*Xmat_hom[2,1])
            eePos_pitch = np.arctan2(-Xmat_hom[2,0],pitch_temp)
            eePos_yaw = np.arctan2(Xmat_hom[1,0],Xmat_hom[0,0])
            eePos_rpy = np.matrix([[eePos_roll,eePos_pitch,eePos_yaw]])

            # then stack it up!
            eePos = np.vstack((eePos_xyz1[:3,:],eePos_rpy.transpose()))
            eePos_arr.append(eePos)
        return eePos_arr

    """
    End Effectors Position Gradients
    * TODO: Add and test Floating base support.
    """
    def equals_or_hstack(self, obj, col):
        if obj is None:
            obj = col
        else:
            obj = np.hstack((obj,col))
        return obj
    def end_effector_pose_gradient(self, q, offsets = [np.matrix([[0,0,0,1]])]):
        n = self.robot.get_num_joints()
        
        # For each branch chain up the transformations across all possible derivatives
        # Note: if not in branch then 0
        deePos_arr = []
        for jid in self.robot.get_leaf_nodes():
            
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)

            # then compute the gradients
            deePos = None
            for dind in range(n):

                # Note: if not in branch then 0
                if dind not in jidChain:
                    deePos_col = np.zeros((6,1))
                    deePos = self.equals_or_hstack(deePos,deePos_col)
                
                else:
                    # first chain up the transforms
                    Xmat_hom = np.eye(4)
                    dXmat_hom = np.eye(4)
                    for ind in jidChain:
                        dcurrX = self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind])
                        currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                        if ind == dind: # use derivative
                            dXmat_hom = np.matmul(dXmat_hom,dcurrX)
                        else: # use normal transform
                            dXmat_hom = np.matmul(dXmat_hom,currX)
                        Xmat_hom = np.matmul(Xmat_hom,currX)

                    # chain up the transforms (version 2 for starting from the leaf)
                    currId = jid
                    if currId == dind:
                        dXmat_hom = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
                    else:
                        dXmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                    Xmat_hom = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                    currId = self.robot.get_parent_id(currId)
                    while(currId != -1):
                        currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                        dcurrX = currX
                        if currId == dind:
                            dcurrX = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
                        dXmat_hom = np.matmul(dcurrX,dXmat_hom)
                        Xmat_hom = np.matmul(currX,Xmat_hom)
                        currId = self.robot.get_parent_id(currId)
                    
                    # Then extract the end-effector position with the given offset(s)
                    # TODO handle different offsets for different branches

                    # xyz position is easy
                    deePos_xyz1 = dXmat_hom * offsets[0].transpose()

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
                    deePos = self.equals_or_hstack(deePos,deePos_col)

            deePos_arr.append(deePos)
        return deePos_arr

    """
    End Effector Hessian
    * TODO: Add and test Floating base support.
    """
    def end_effector_pose_hessian(self, q, offsets = [np.matrix([[0,0,0,1]])]):
        n = self.robot.get_num_joints()
        
        # For each branch chain up the transformations across all possible derivatives
        # Note: if not in branch then 0
        d2eePos_arr = []
        for jid in self.robot.get_leaf_nodes(): # can be done in parallel
            
            # first get the joints in the chain
            jidChain = sorted(self.robot.get_ancestors_by_id(jid))
            jidChain.append(jid)

            # first chain up the 1st derivative transforms
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                # n+1 for standard as well (noting that n+1 will never trigger derivative)
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                for ind in jidChain:
                    if ind == dind: # use derivative
                        dcurrX = self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind])
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],dcurrX)
                    else: # use normal transform
                        currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                        dXmat_hom_arr[:,:,dind] = np.matmul(dXmat_hom_arr[:,:,dind],currX)

            # chain up the 1st derivative transforms (version 2 for starting from the leaf)
            dXmat_hom_arr = np.zeros((4,4,n+1))
            for dind in range(n+1): # can be done in parallel
                currId = jid
                dXmat_hom_arr[:,:,dind] = np.eye(4)
                while(currId != -1):
                    if currId == dind:
                        dcurrX = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
                    else:
                        dcurrX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
                    dXmat_hom_arr[:,:,dind] = np.matmul(dcurrX,dXmat_hom_arr[:,:,dind])
                    currId = self.robot.get_parent_id(currId)

            # then chain up the 2nd derivative transforms
            d2Xmat_hom_arr = np.zeros((4,4,n,n))
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    for ind in jidChain:
                        if (ind == dind_i) or (ind == dind_j):
                            if dind_i == dind_j: # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_Func_by_id(ind)(q[ind])
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],d2currX)
                            else: # use first derivative values
                                dcurrX = self.robot.get_dXmat_hom_Func_by_id(ind)(q[ind])
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],dcurrX)
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(ind)(q[ind])
                            d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2Xmat_hom_arr[:,:,dind_i,dind_j],currX)

            # then chain up the 2nd derivative transforms (version 2 - backward)
            for dind_i in range(n): # can be done in parallel
                for dind_j in range(n): # can be done in parallel
                    currId = jid
                    d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.eye(4)
                    while(currId != -1):
                        if (currId == dind_i) or (currId == dind_j):
                            if dind_i == dind_j: # use second derivative
                                d2currX = self.robot.get_d2Xmat_hom_Func_by_id(currId)(q[currId])
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(d2currX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                            else: # use first derivative values
                                dcurrX = self.robot.get_dXmat_hom_Func_by_id(currId)(q[currId])
                                d2Xmat_hom_arr[:,:,dind_i,dind_j] = np.matmul(dcurrX,d2Xmat_hom_arr[:,:,dind_i,dind_j])
                        else: # use normal transform
                            currX = self.robot.get_Xmat_hom_Func_by_id(currId)(q[currId])
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
                    if (dind_i not in jidChain) or (dind_j not in jidChain):
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
        return d2eePos_arr
    
    def apply_external_forces(self, q, f_in, f_ext):
        """ Implementation based on spatial v2: https://github.com/ROAM-Lab-ND/spatial_v2_extended/blob/main/dynamics/apply_external_forces.m
        
        Subtracts external forces from input f_in. 
        F_ext must take the structure of either a 6/3xNB matrix, or a shortened
        planar vector with length == NB, with f[i] corresponding to the force applied to body i.

        Parameters:
        - f_in (numpy.ndarray): Initial forces applied to links. 
        - f_ext (numpy.ndarray): The external force.

        Returns:
        - f_out (numpy.ndarray): The updated force.
        TODO Check the correct way to index the forces!
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
            if self.robot.floating_base and curr_id == 0: vJ = np.matmul(S, np.transpose(np.matrix(_qd)))
            else: vJ = S * _qd
            v[:, curr_id] += np.squeeze(np.array(vJ))  # reduces shape to (6,) matching v[:,curr_id]
            a[:, curr_id] += self.mxS(vJ, v[:, curr_id])
            
            if qdd is not None:
                _qdd = qdd[inds_v]
                if self.robot.floating_base and curr_id == 0: aJ = np.matmul(S, np.transpose(np.matrix(_qdd)))
                else: aJ = S * _qdd
                a[:, curr_id] += np.squeeze(np.array(aJ))  # reduces shape to (6,) matching a[:,curr_id]
            # compute f
            Imat = self.robot.get_Imat_by_id(curr_id)
            f[:, curr_id] = np.matmul(Imat, a[:, curr_id]) + self.vxIv(v[:, curr_id], Imat)

        return (v, a, f)

    def rnea_bpass(self, q, f, USE_VELOCITY_DAMPING = False):
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

    def rnea(self, q, qd, qdd=None, GRAVITY=-9.81, USE_VELOCITY_DAMPING = False, f_ext=None):
            """
            Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

            RNEA divided into 3 parts: 
                1) calculate the velocity and acceleration of each body in the tree
                2) Calculate the forces necessary to produce these accelertions
                3) Calculate the forces transmitted across the joints from the forces acting on the bodies
                
            INPUT:
            q, qd, qdd: position, velocity, acceleration. Nx1 arrays where N is the number of bodies
            GRAVITY - gravitational field of the body; default is earth surface gravity, 9.81
            USE_VELOCITY_DAMPING: flag for whether velocity is damped, representing ___
            
            OUTPUTS: 
            c: Coriolis terms and other forces potentially be applied to the system. 
            v: velocity of each joint in world base coordinates rather than motion subspace
            a: acceleration of each joint in world base coordinates rather than motion subspace
            f: forces that joints must apply to produce trajectory
            """
            # forward pass
            (v, a, f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
            # backward pass
            (c, f) = self.rnea_bpass(q, f, USE_VELOCITY_DAMPING)

            return (c, v, a, f)

    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        """
        Performs the forward pass of the gradient of the Recursive Newton-Euler Algorithm (RNEA) with respect to joint positions.

        Args:
            q (numpy.ndarray): Joint positions.
            qd (numpy.ndarray): Joint velocities.
            v (numpy.ndarray): Spatial velocities.
            a (numpy.ndarray): Spatial accelerations.
            GRAVITY (float, optional): Gravity value. Defaults to -9.81.

        Returns:
            tuple: A tuple of np.ndarrays containing the derivative matrices dv_dq, da_dq, and df_dq.
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
            # Xmat access sequence
            inds_v = self.robot.get_joint_index_v(ind) # handles floating base joint indexing
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,inds_v,ind] += self._mxS(S,np.matmul(Xmat,v[:,parent_ind])) # replace with new mxS
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            for c in range(n):
                # TODO compress the if condition to handle 6x6 in one step
                if parent_ind == -1 and self.robot.floating_base:
                    # dv_dq should be all 0s => this results in all 0s
                    for ii in range(len(inds_v)):
                        da_dq[:,c,ii] += self._mxS(S[ii],dv_dq[:,c,ii],qd[ii]) # dv/du x S*q
                else:
                    da_dq[:,c,ind] += self._mxS(S,dv_dq[:,c,ind],qd[inds_v]) # replace with new mxS
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,inds_v,ind] += self._mxS(S,np.matmul(Xmat,a[:,parent_ind])) # replace with new mxS
            else:
                da_dq[:,inds_v,ind] += self._mxS(S,np.matmul(Xmat,gravity_vec)) # replace with new mxS 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))
        
        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
        """
        Performs the forward pass of the Recursive Newton-Euler Algorithm (RNEA) for gradient computation with respect to qd.

        Args:
            q (np.ndarray): The joint positions.
            qd (np.ndarray): The joint velocities.
            v (6,NB) (np.ndarray): The body spatial velocities.

        Returns:
            Tuple: A tuple of np.ndarrays containing the gradient of spatial acceleration (dv_dqd), 
            gradient of spatial force (da_dqd), and gradient of spatial force derivative (df_dqd).
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
            for c in range(n): # TODO check here and redo as vectorized
                if parent_ind == -1 and self.robot.floating_base:
                    for ii in range(len(inds_v)):
                        da_dqd[:,c,ind] += self._mxS(S[ii],dv_dqd[:,c,ind],qd[ii]) # TODO maybe add special case of _mxS when S is 6x6 identity
                else:
                    da_dqd[:,c,ind] += self._mxS(S,dv_dqd[:,c,ind],qd[inds_v]) 

            da_dqd[:,inds_v,ind] += self._mxS(S,v[:,ind]) 
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))

        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
        """
        Calculates the gradient of the bias-passing recursive Newton-Euler algorithm with respect to the joint positions.

        Args:
            q (numpy.ndarray): Array of joint positions.
            f (numpy.ndarray): Array of joint forces.
            df_dq (numpy.ndarray): Array of partial derivatives of joint forces with respect to joint positions.

        Returns:
            dc_dq (numpy.ndarray): Array representing the gradient of RNEA with respect to the joint positions.
        """
        
        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel() # assuming len(q) = len(qd)
        dc_dq = np.zeros((n,n))
        
        for ind in range(NB-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)
            # dc_du is S^T*df_du
            inds_v = self.robot.get_joint_index_v(ind)
            S = self.robot.get_S_by_id(ind)
            if parent_ind == -1 and self.robot.floating_base:
                dc_dq[inds_v] = df_dq[:,:,ind]
            else:
                dc_dq[inds_v,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            if parent_ind != -1:
                # Xmat access sequence
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                df_dq[:6,inds_v,parent_ind] += delta_dq

        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING=False):
        """
        Calculates the gradient of the RNEA (Recursive Newton-Euler Algorithm) backward pass with respect to the joint velocities.

        Args:
            q (numpy.ndarray): Array of joint positions.
            df_dqd (numpy.ndarray): Array of partial derivatives of the forward dynamics residual with respect to the joint velocities.
            USE_VELOCITY_DAMPING (bool, optional): Flag indicating whether to include velocity damping. Defaults to False.

        Returns:
            numpy.ndarray: Array representing the gradient of the RNEA backward pass with respect to the joint velocities.
        """

        # allocate memory
        NB = self.robot.get_num_bodies()
        n = self.robot.get_num_vel()  # len(qd) always
        dc_dqd = np.zeros((n, n))

        for ind in range(NB - 1, -1, -1):
            parent_ind = self.robot.get_parent_id(ind)
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            inds_v = self.robot.get_joint_index_v(ind)
            dc_dqd[inds_v, :] = np.matmul(np.transpose(S), df_dqd[:, :, ind])
            # df_du_parent += X^T*df_du
            if parent_ind != -1:
                inds_q = self.robot.get_joint_index_q(ind)
                _q = q[inds_q]
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                df_dqd[:, :, parent_ind] += np.matmul(np.transpose(Xmat), df_dqd[:, :, ind])

        # add in the damping and simplify this expression later
        # suggestion: have a getter function that automatically indexes and allocates for floating base functions
        if USE_VELOCITY_DAMPING:
            for ind in range(NB):
                if self.robot.floating_base and self.robot.get_parent_id(ind) == -1:
                    dc_dqd[ind : ind + 5, ind : ind + 5] += self.robot.get_damping_by_id(ind)
                else:
                    dc_dqd[ind, ind] += self.robot.get_damping_by_id(ind)

        return dc_dqd

    def rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
        """
        The gradients of inverse dynamics can be very extremely useful inputs into trajectory optimization algorithmss.
        Input: trajectory, including position, velocity, and acceleration
        Output: Computes the gradient of joint forces with respect to the positions and velocities. 
        """ 
        
        (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)

        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        dc_du = np.hstack((dc_dq,dc_dqd))
        return (dc_dq, dc_dqd)

    def idsva(self, q, qd, qdd, GRAVITY = -9.81):
        """alternative to rnea_grad(), described in "Efficient Analytical Derivatives of Rigid-Body Dynamics using
    Spatial Vector Algebra" (Singh, Russel, and Wensing)

        :param q: initial joint positions
        :type q: array representing a 1 by n vector, where n is the number of joints
        :param qd: initial joint velocities
        :type qd: array representing a 1 by n vector
        :param qdd: desired joint accelerations 
        :type qdd: array representing a 1 by n vector
        :param GRAVITY: defaults to -9.81
        :type GRAVITY: float, optional
        :return: dtau_dq, dtau_dqd-- the gradient of the resulting torque with respect to initial joint position and velocity
        :rtype: list

        NOTE: This function has not been tested for floating base.
        """
        # allocate memory
        n = len(qd)
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame 
        Xdown0 = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        Sdd = np.zeros((6,n))
        Sj = np.zeros((6,n)) 
        IC = [None] * n 
        BC  = [None] * n 
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        
        # forward pass
        for i in range(n):
            parent_i = self.robot.get_parent_id(i)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])

            # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i])

            S[:,i] = self.robot.get_S_by_id(i)

            S[:,i] = Xdown0[i] @ S[:,i]
            Sd[:,i] = self.cross_operator(v[:,i]) @ S[:,i]
            Sdd[:,i]= self.cross_operator(a[:,i])@ S[:,i]
            Sdd[:,i] = Sdd[:,i] + self.cross_operator(v[:,i])@ Sd[:,i]
            Sj[:,i] = 2*Sd[:,i] + self.cross_operator(S[:,i]*qd[i])@ S[:,i]

            v6x6 = self.cross_operator(v[:,i])
            v[:,i] = v[:,i] + S[:,i]*qd[i]
            a[:,i] = a[:,i] + np.array(v6x6 @ S[:,i]*qd[i])

            if qdd is not None:
                a[:,i] += S[:,i]*qdd[i]
            
            # compute f, IC, BC
            Imat = self.robot.get_Imat_by_id(i)

            IC[i] = np.array(Xup0[i]).T  @ (Imat @ Xup0[i])
            f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @ v[:,i]
            f[:,i] = np.asarray(f[:,i]).flatten()
            BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
        

        t1 = np.zeros((6,n))
        t2 = np.zeros((6,n))
        t3 = np.zeros((6,n))
        t4 = np.zeros((6,n))
        dtau_dq = np.zeros((n,n))
        dtau_dqd = np.zeros((n,n))


        #backward pass
        for i in range(n-1,-1,-1):

            t1[:,i] = IC[i] @ S[:,i]
            t2[:,i] = BC[i] @ S[:,i].T + IC[i] @ Sj[:,i]
            t3[:,i] = BC[i] @ Sd[:,i] + IC[i] @ Sdd[:,i] + self.icrf(f[:,i]) @ S[:,i]
            t4[:,i] = BC[i].T @ S[:,i]

            subtree_ids = self.robot.get_subtree_by_id(i) #list of all subtree ids (inclusive)



            dtau_dq[i, subtree_ids[1:]] = S[:,i] @ t3[:,subtree_ids[1:]]
            
            dtau_dq[subtree_ids[0:], i] = Sdd[:,i] @ t1[:,subtree_ids[0:]] + \
                                        Sd[:,i] @ t4[:,subtree_ids[0:]] 
            
            dtau_dqd[i, subtree_ids[1:]] = S[:,i] @ t2[:,subtree_ids[1:]]

            dtau_dqd[subtree_ids[0:], i] = Sj[:,i] @ t1[:,subtree_ids[0:]] + \
                                        S[:,i] @ t4[:,subtree_ids[0:]] 

            p = self.robot.get_parent_id(i)
            if p >= 0:
                IC[p] = IC[p] + IC[i]
                BC[p] = BC[p] + BC[i]
                f[:,p] = f[:,p] + f[:,i]


        return dtau_dq, dtau_dqd

    # crf_idsva is being called by second_order IDSVA
    def crf_idsva(self, v):
        vcross = -self.crm(v).conj().T #negative complex conjugate transpose
        return vcross
    # dot_matrix is being called by second_order IDSVA
    def dot_matrix(self, I, v):
        return self.crf_idsva(v) @ I - I @ self.crm(v)

    def second_order_idsva_series(self, q, qd, qdd, GRAVITY = -9.81):
        """
        :param q: initial joint positions
        :type q: array representing a 1 by n vector, where n is the number of joints
        :param qd: initial joint velocities
        :type qd: array representing a 1 by n vector
        :param qdd: desired joint accelerations 
        :type qdd: array representing a 1 by n vector
        :param GRAVITY: defaults to -9.81
        :type GRAVITY: float, optional
        :return: d2tau_dq, d2tau_dv, d2tau_dqv, dM_dq
        :rtype: list

        NOTE: This function has not been tested for floating base.
        """    
        # allocate memory
        n = len(qd) # n = 7
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame
        Xdown0 = [None] * n
        IC = [None] * n
        BC = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        vJ = np.zeros((6,n))
        aJ = np.zeros((6,n))
        psid = np.zeros((6,n))
        psidd = np.zeros((6,n))
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        # forward pass 
        modelNB = n
        modelNV = self.robot.get_num_joints()
        for i in range(modelNB):
            parent_i = self.robot.get_parent_id(i)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])
            # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec # 
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i]) 
            S[:,i] = self.robot.get_S_by_id(i)
            S[:,i] = Xdown0[i] @ S[:,i]
            vJ[:,i] = S[:,i] * qd[i]
            aJ[:,i] = self.crm(v[:,i])@vJ[:,i] + S[:,i] * qdd[i]
            psid[:,i] = self.crm(v[:,i])@S[:,i]
            psidd[:,i] = self.crm(a[:,i])@S[:,i] + self.crm(v[:,i])@psid[:,i]
            v[:,i] = v[:,i] + vJ[:,i]
            a[:,i] = a[:,i] + aJ[:,i]
            I = self.robot.get_Imat_by_id(i)
            IC[i] = np.array(Xup0[i]).T @ (I @ Xup0[i])
            Sd[:, i] = self.crm(v[:,i]) @ S[:,i]
            assert Sd[:, i].shape == (6,), f"Unexpected shape for Sd[:, {i}]: {Sd[:, i].shape}"
            BC[i] = (self.crf_idsva(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.crm(v[:,i]))
            f[:,i] = IC[i] @ a[:,i] + self.crf_idsva(v[:,i]) @ IC[i] @v[:,i] 
        
        # Matrix Intialization
        dM_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dqd = np.zeros((modelNV,modelNV,modelNV))
        d2tau_cross = np.zeros((modelNV,modelNV,modelNV))

        #backward pass
        for i in range(modelNB-1,-1,-1):
            modelnv = 1 # DOF of ith joint, Since revolute type for iiwa so this is 1
            for p in range(modelnv):
                S_p = S[:, i]  
                Sd_p = Sd[:, i]
                psid_p = psid[:, i]
                psidd_p = psidd[:, i]
                
                Bic_phii = self.icrf(IC[i] @ S_p)
                Bic_psii_dot = 2 * 0.5 * (self.crf_idsva(psid_p) @ IC[i] + self.icrf(IC[i] @ psid_p) - IC[i] @ self.crm(psid_p))

                A0 = self.icrf(IC[i] @ S_p )
                A1 = self.dot_matrix(IC[i], S_p) 
                A2 = 2 * A0 - Bic_phii
                A3 = Bic_psii_dot + self.dot_matrix(BC[i], S_p)
                A4 = self.icrf(BC[i].T @ S_p)
                A5 = self.icrf(BC[i] @ psid_p  + IC[i]@psidd_p + self.crf_idsva(S_p) @ f[:, i])
                A6 = self.crf_idsva(S_p) @ IC[i] + A0
                A7 = self.icrf(BC[i] @ S_p + IC[i] @ (psid_p + Sd_p) )
                ii = i
                j = i 

                while j >= 0 :
                    jj = j 
                    modelnvj = 1 # DOF of jth joint
                    for t in range(modelnvj):
                        S_t = S[:, j]
                        Sd_t = Sd[:, j]
                        psid_t = psid[:, j]
                        psidd_t = psidd[:, j]    
                        u1 = A3.T @ S_t
                        u2 = A1.T @ S_t
                        u3 = A3 @ psid_t + A1 @ psidd_t + A5 @ S_t
                        u4 = A6 @ S_t
                        u5 = A2 @ psid_t + A4 @ S_t
                        u6 = Bic_phii @ psid_t + A7 @ S_t
                        u7 = A3 @ S_t + A1 @ (psid_t + Sd_t)
                        u8 = A4 @ S_t - Bic_phii.T @ psid_t
                        u9 = A0 @ S_t
                        u10 = Bic_phii @ S_t
                        u11 = Bic_phii.T @ S_t
                        u12 = A1 @ S_t
                        
                        k = j
                        while k >= 0:
                            kk = k 
                            modelnvk = 1 # DOF of kth joint
                            for r in range(modelnvk):
                                S_r = S[:, k]
                                Sd_r = Sd[:, k]
                                psid_r = psid[:, k]
                                psidd_r = psidd[:, k]
                                p1 = psid_r @ u11.T
                                p2 =  psid_r @ u8.T + psidd_r @ u9.T
                                d2tau_dq[ii, jj, kk] = p2
                                d2tau_cross[ii, kk, jj] = -p1

                                if j != i:
                                    d2tau_dq[jj, kk, ii] = psid_r @ u1.T + psidd_r @ u2.T
                                    d2tau_dq[jj, ii, kk] = d2tau_dq[jj, kk, ii]
                                    d2tau_cross[jj, kk, ii] = p1
                                    d2tau_cross[jj, ii, kk] = S_r @ u1.T + (psid_r + Sd_r) @ u2.T
                                    d2tau_dqd[jj, kk, ii] = S_r @ u11.T
                                    d2tau_dqd[jj, ii, kk] = d2tau_dqd[jj, kk, ii]
                                    dM_dq[kk, jj, ii] = u12 @ S_r.T
                                    dM_dq[jj, kk, ii] = u12 @ S_r.T

                                if k != j:
                                    d2tau_dq[ii, kk, jj] = p2
                                    d2tau_dq[kk, ii, jj] = u3 @ S_r.T
                                    d2tau_dqd[ii, jj, kk] = -S_r @ u11.T
                                    d2tau_dqd[ii, kk, jj] = -S_r @ u11.T
                                    d2tau_cross[ii, jj, kk] = u5 @ S_r.T + (psid_r + Sd_r) @ u9.T
                                    d2tau_cross[kk, jj, ii] = u6 @ S_r.T
                                    dM_dq[kk, ii, jj] = u9 @ S_r.T
                                    dM_dq[ii, kk, jj] = u9 @ S_r.T
                                
                                    if j != i:
                                        d2tau_dq[kk, jj, ii] = d2tau_dq[kk, ii, jj]
                                        d2tau_dqd[kk, ii, jj] = u10 @ S_r.T
                                        d2tau_dqd[kk, jj, ii] = d2tau_dqd[kk, ii, jj]
                                        d2tau_cross[kk, ii, jj] = u7 @ S_r.T
                                    else:
                                        d2tau_dqd[kk,jj,ii] = u4 @ S_r.T
                                else:
                                    d2tau_dqd[ii,jj,kk] = - S_r @ u2.T
                            g = self.robot.get_parent_id(k)
                            k = g
                    z = self.robot.get_parent_id(j)
                    j = z
            pi = self.robot.get_parent_id(i)
            if pi >= 0:
                IC[pi] = IC[pi] + IC[i]
                BC[pi] = BC[pi] + BC[i]
                f[:, pi] = f[:, pi] + f[:, pi + 1]
        return d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq

    def minv_bpass(self, q):
        """
        Performs the backward pass of the Minv algorithm.

        NOTE:
        If floating base, treat floating base joint as 6 joints (Px,Py,Pz,Rx,Ry,Rz) where P=prismatic R=Revolute.
        Thus, allocate memroy and assign a "matrix_ind" shifting indices to match 6 joint representation.
        This can be accessed using self.robot.get_joint_index_v(ind).
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
                adj_subtreeInds = list(
                    np.array(subtreeInds) + 5
                )  # adjusted for matrix calculation
            else:
                adj_subtreeInds = subtreeInds
            inds_v = self.robot.get_joint_index_v(ind) # formerly matrix_ind
            parent_ind = self.robot.get_parent_id(ind)
            if (
                parent_ind == -1 and self.robot.floating_base
            ):  # floating base joint check
                # Compute U, D
                S = self.robot.get_S_by_id(ind)  # np.eye(6) for floating base
                U[inds_v, :] = np.matmul(IA[ind], S)
                fb_Dinv = np.linalg.inv(
                    np.matmul(S.transpose(), U[inds_v, :])
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
                U[inds_v, :] = np.matmul(IA[ind], S).reshape(6,)
                Dinv[inds_v] = np.matmul(S.transpose(), U[inds_v, :])
                # Update Minv and subtrees
                Minv[inds_v, inds_v] = 1 / Dinv[inds_v]
                # Deals with issue where result is np.matrix instead of np.array (can't shape np.matrix as 1 dimension)
                Minv[inds_v, adj_subtreeInds] -= np.squeeze(
                    np.array(
                        1
                        / (Dinv[inds_v])
                        * np.matmul(S.transpose(), F[inds_v, :, adj_subtreeInds].T)
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
                        F[inds_v, :, subInd] += (
                            U[inds_v, :] * Minv[inds_v, subInd]
                        )
                        F[matrix_parent_ind, :, subInd] += np.matmul(
                            np.transpose(Xmat), F[inds_v, :, subInd]
                        )
                    # update IA
                    Ia = IA[ind] - np.outer(
                        U[inds_v, :],
                        ((1 / Dinv[inds_v]) * np.transpose(U[inds_v, :])),
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
        This can be accessed using self.robot.get_joint_index_v(ind)
        See Spatial_v2_extended algorithm for alterations to fpass algorithm.
        Additionally, made convenient shift to F[i] accessing based on matrix structure in math.

        Args:
            q (numpy.ndarray): The joint positions.
            Minv (numpy.ndarray): The inverse mass matrix.
            F (numpy.ndarray): The spatial forces.
            U (numpy.ndarray): The joint velocity transformation matrix.
            Dinv (numpy.ndarray): The inverse diagonal inertia matrix.

        Returns:
            Minv (numpy.ndarray): The updated inverse mass matrix Minv.
        """
        NB = self.robot.get_num_bodies()
        # # Forward pass
        for ind in range(NB):
            inds_v = self.robot.get_joint_index_v(ind) # formerly matrix_ind
            inds_q = self.robot.get_joint_index_q(ind)
            _q = q[inds_q]
            parent_ind = self.robot.get_parent_id(ind)
            S = self.robot.get_S_by_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
            if parent_ind != -1:
                Minv[inds_v, :] -= (1 / Dinv[inds_v]) * np.matmul(
                    np.matmul(U[inds_v].transpose(), Xmat), F[parent_ind]
                )
                F[ind] = np.matmul(Xmat, F[parent_ind]) + np.outer(
                    S, Minv[inds_v, :]
                )
            else:
                if self.robot.floating_base:
                    F[ind] = np.matmul(S, Minv[inds_v, ind:])
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

    # 
    def crm(self,v):
        if len(v) == 6:
            vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
        else:
            vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
        return vcross


    def aba(self, q, qd, tau, f_ext=[], GRAVITY = -9.81):
        """
        Compute the Articulated Body Algorithm (ABA) to calculate the joint accelerations.
        """
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
            gravity_vec[5] = GRAVITY  # a_base is gravity vec

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
                U[:, inds_v] = np.matmul(Xmat.T, U[:, inds_v]) # spatial edit

                if parent_ind != -1:

                    rightSide = np.reshape(U[:, inds_v], (6,1)) @ np.reshape(U[:, inds_v], (6,1)).T / d[ind]
                    Ia = np.matmul(Xmat.T, np.matmul(IA[ind], Xmat)) - rightSide # spatial edit

                    pa = np.matmul(Xmat.T, pA[:, ind] + np.matmul(IA[ind], c[:, ind]))
                    pa = pa + (np.reshape(U[:, inds_v], (6,1)) @ ((1/d[ind]) * u[inds_v])).T
                    
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
                    a[:, ind] = -gravity_vec
                else:
                    a[:, ind] = a[:, parent_ind]
                
                S = self.robot.get_S_by_id(ind)
                temp = u[inds_v] - np.matmul(np.transpose(U[:, inds_v]), a[:, ind])

                if parent_ind == -1:
                    # qdd[inds_v] = np.matmul(np.linalg.inv(d[ind]), temp)
                    if self.robot.floating_base:
                        qdd[inds_v] = np.linalg.solve(d[ind], temp)
                        a[:, ind] = np.matmul(Xmat, a[:, ind]) + np.matmul(S.T,qdd[inds_v]) + c[:, ind]
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

                pA[:,ind]=np.matmul(temp,v[:,ind])[0]
            
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
        
        return qdd




    def crba(self, q):
        """
        Computes the Composite Rigid Body Algorithm (CRBA) to calculate the joint-space inertia matrix.
        # Based on Featherstone implementation of CRBA p.182 in rigid body dynamics algorithms book.

        NOTE:
        If Floating base, treat floating base joint as 6 joints (Px,Py,Pz,Rx,Ry,Rz) where P=prismatic R=Revolute.
        Thus, allocate memroy and assign a "matrix_ind" shifting indices to match 6 joint representation.
        Propagate changes to any indexing of j, F, U, Dinv, Minv, etc. to match 6 joint representation.

        Parameters:
        - q (numpy.ndarray): Joint positions.

        Returns:
        - H (numpy.ndarray): Joint-space inertia matrix.
        """
        if self.robot.floating_base:
            NB = self.robot.get_num_bodies()
            H = np.zeros((NB, NB))

            IC = copy.deepcopy(
                self.robot.get_Imats_dict_by_id()
            )  # composite inertia calculation
            for ind in range(NB - 1, -1, -1):
                parent_ind = self.robot.get_parent_id(ind)
                inds_v = self.robot.get_joint_index_v(ind)
                if parent_ind != -1:
                    _q = q[self.robot.get_joint_index_q(ind)]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    IC[parent_ind] = IC[parent_ind] + np.matmul(
                        np.matmul(Xmat.T, IC[ind]), Xmat
                    )
                    fh = np.matmul(IC[ind], S)
                    H[inds_v, inds_v] = np.matmul(S.T, fh)
                    j = ind
                    while self.robot.get_parent_id(j) > 0:
                        Xmat = self.robot.get_Xmat_Func_by_id(j)(
                            q[self.robot.get_joint_index_q(j)]
                        )
                        fh = np.matmul(Xmat.T, fh)
                        j = self.robot.get_parent_id(j)
                        H[inds_v, j + 5] = np.matmul(fh.T, S)
                        H[j + 5, inds_v] = H[inds_v, j + 5]
                    # # treat floating base 6 dof joint
                    inds_q = self.robot.get_joint_index_q(j)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(j)(_q)
                    S = np.eye(6)
                    fh = np.matmul(Xmat.T, fh)
                    H[inds_v, :6] = np.matmul(fh.T, S)
                    H[:6, inds_v] = H[inds_v, :6].T
                else:
                    inds_q = self.robot.get_joint_index_q(ind)
                    _q = q[inds_q]
                    Xmat = self.robot.get_Xmat_Func_by_id(ind)(_q)
                    S = self.robot.get_S_by_id(ind)
                    parent_ind = self.robot.get_parent_id(ind)
                    fh = np.matmul(IC[ind], S)
                    H[ind:6, ind:6] = np.matmul(S.T, fh)
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
                    fh = np.matmul(Xmat.T, fh)
                    j = self.robot.get_parent_id(j)
                    S = self.robot.get_S_by_id(j)
                    H[ind, j] = np.matmul(S.T, fh)
                    H[j, ind] = H[ind, j]

        return H 
    
    def forward_dynamics(self, q, qd, u):
        """
        Compute the forward dynamics with respect to the generalized coordinates, velocities, and torques.

        Parameters:
        q (numpy.ndarray): Generalized coordinates (joint positions).
        qd (numpy.ndarray): Generalized velocities (joint velocities).
        tau (numpy.ndarray): Generalized forces (joint torques).

        Returns:
        tuple: A tuple containing:
            - dqdd_dq (numpy.ndarray): Gradient of the joint accelerations with respect to the joint positions.
            - dqdd_dqd (numpy.ndarray): Gradient of the joint accelerations with respect to the joint velocities.
            - dqdd_dc (numpy.ndarray): Gradient of the joint accelerations with respect to the joint torques.
        """
        (c,v,a,f) = self.rnea(q, qd)
        minv = self.minv(q)
        return np.matmul(minv, -c)

    def forward_dynamics_grad(self, q, qd, tau):
        """
        Compute the gradient of the forward dynamics with respect to the generalized coordinates, velocities, and torques.

        Parameters:
        q (numpy.ndarray): Generalized coordinates (joint positions).
        qd (numpy.ndarray): Generalized velocities (joint velocities).
        tau (numpy.ndarray): Generalized forces (joint torques).

        Returns:
        tuple: A tuple containing:
            - dqdd_dq (numpy.ndarray): Gradient of the joint accelerations with respect to the joint positions.
            - dqdd_dqd (numpy.ndarray): Gradient of the joint accelerations with respect to the joint velocities.
            - dqdd_dc (numpy.ndarray): Gradient of the joint accelerations with respect to the joint torques.
        """

        qdd = self.forward_dynamics(q, qd, tau)

        dc_dq, dc_dqd = self.rnea_grad(q, qd, qdd = qdd)

        Minv = self.minv(q)

        dqdd_dq = np.matmul(-Minv, dc_dq)
        dqdd_dqd = np.matmul(-Minv, dc_dqd)

        return dqdd_dq, dqdd_dqd


