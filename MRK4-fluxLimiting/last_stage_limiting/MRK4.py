import numpy as np
from matplotlib import pyplot as plt

order=2
L=1.0

# These functions are designed for a 1D periodic domain from [0,L] #

######################################################
# COMPUTE MATRICES: TRANSPORT AND DISCRETE UPWINDING #
######################################################
def compute_matrices(U,N):
    dtMax = 1.0e10
    h=L/N
    if order==2:
        D1=np.diag(np.ones(N-1),1)-np.diag(np.ones(N-1),-1)
        D1[0,-1]=-1.
        D1[-1,0]=1
        D1*=-U/2.0/h
    else:
        D1=-np.diag(np.ones(N-2),2) + 8*np.diag(np.ones(N-1),1) - 8*np.diag(np.ones(N-1),-1) + np.diag(np.ones(N-2),-2)
        D1[0,-2]=1; D1[0,-1]=-8
        D1[1,-1]=1
        D1[-1,0]=8; D1[-1,1]=-1
        D1[-2,0]=-1
        D1*=-U/12.0/h
    #
    # MPP artificial viscosity operator; i.e, discrete upwinding matrix #
    DL=D1*0
    for i in range(N):
        for j in np.arange(i+1,N):
            DL[i,j]=np.max([-D1[i,j],0,-D1[j,i]])
            DL[j,i]=DL[i,j]
        DL[i,i]=-np.sum(DL[i,:])

        dtMax = min(dtMax,1.0/np.abs(D1[i,i]+DL[i,i]))
    return D1,DL,dtMax
#

##################################################################
# jVector: this determines the sparsity pattern of the operators #
##################################################################
def get_jVector(i,D1):
    if order==2:
        jVector = [len(D1)-1,i,i+1] if i==0 else ([i-1,i,0] if i==len(D1)-1 else [i-1,i,i+1])
    else:
        if i==0:
            jVector = [len(D1)-2,len(D1)-1,i,i+1,i+2]
        elif i==1:
            jVector = [len(D1)-1,i-1,i,i+1,i+2]
        elif i == len(D1)-1:
            jVector = [i-2,i-1,i,0,1]
        elif i == len(D1)-2:
            jVector = [i-2,i-1,i,i+1,0]
        else:
            jVector = [i-2,i-1,i,i+1,i+2]
        #
    #
    return jVector
#

##########################
# SPATIAL DISCRETIZATION #
##########################
def sdisc(u,D1,DL,compute_flux_matrices=True):
    fG = np.dot(D1+DL,u)
    fB = np.dot(-DL,u)

    fijG = D1*0
    fijB = D1*0

    if compute_flux_matrices:
        for i in range(len(D1)):
            jVector = get_jVector(i,D1)
            for j in jVector:
                fijG[i,j] = D1[i,j]*u[j]-D1[j,i]*u[i] + DL[i,j]*(u[j]-u[i])
                fijB[i,j] = -DL[i,j]*(u[j]-u[i])
            #
        #
    #
    return fG+fB,fG,fB,fijG,fijB
#

#################################
# LIMITED NON-CONSERVATIVE FLUX #
#################################
def limited_non_conservative_flux(dt,un,fG,fB,global_bounds=False,single_gamma=False,gmin=0.0,gmax=1.0):
    gamma=0*fB
    for i in range(len(fB)):
        # compute bounds
        if global_bounds:
            umax=gmax
            umin=gmin
        else:
            jVector = get_jVector(i,un)
            for j in jVector:
                umax=max(umax,un[j])
                umin=min(umin,un[j])
            #
        # compute non-conservative limiters
        gamma_neg = (umin-un[i]-dt*fG[i])/dt/fB[i] if (un[i]+dt*(fG[i]+fB[i])<umin and fB[i]!=0) else 1.0
        gamma_pos = (umax-un[i]-dt*fG[i])/dt/fB[i] if (un[i]+dt*(fG[i]+fB[i])>umax and fB[i]!=0) else 1.0
        gamma[i]=min(gamma_neg,gamma_pos)
    #
    if single_gamma:
        sfB=np.min(gamma)*fB
    else:
        sfB=np.multiply(gamma,fB)
    return sfB
#

###############################################
# LIMITED CONSERVATIVE FLUX VIA ITERATIVE FCT #
###############################################
def limited_conservative_flux(un,uL,fij,global_bounds=False,gmin=0.0,gmax=1.0,num_iter=1):
    tol=1.0e-12
    min_Lij=1.0e10
    Rpos=un*0; Rneg=un*0; limited_flux_correction=un*0
    umax=un*0-1.0e10; umin=un*0+1.0e10

    fluxMatrix = np.copy(fij)
    limitedFlux= fij*0
    uLim = np.copy(uL)

    ######################
    # compute the bounds #
    ######################
    for i in range(len(un)):
        # compute bounds
        if global_bounds:
            umin[i]=gmin
            umax[i]=gmax
        else:
            jVector = get_jVector(i,un)
            for j in jVector:
                umax[i]=max(umax[i],un[j])
                umin[i]=min(umin[i],un[j])
            #
            # check uL
            if (uL[i]-umin[i]<-tol or uL[i]-umax[i]>tol):
                print (umin[i], uL[i],umax[i])
                assert(umin[i]<=uL[i] and uL[i]<=umax[i]),"Error: low-order solutin is not on bounds"
            #
        #
    #
    #################
    # Iterative FCT #
    #################
    for iter in range(num_iter):
        for i in range(len(un)):
            Pposi=0; Pnegi=0
            jVector = get_jVector(i,un)
            # compute p vectors
            for j in jVector:
                fluxij = fluxMatrix[i,j] - limitedFlux[i,j]
                Pposi += max(fluxij,0)
                Pnegi += min(fluxij,0)
            #
            # Compute Q vectors
            uLimi = uLim[i]
            Qposi = umax[i]-uLimi
            Qnegi = umin[i]-uLimi

            # Compute R vectors
            Rpos[i] = min(1.,Qposi/Pposi) if Pposi!=0 else 1.0
            Rneg[i] = min(1.,Qnegi/Pnegi) if Pnegi!=0 else 1.0
        #
        # compute limited flux
        for i in range(len(un)):
            ith_limiter_times_fluxCorrectionMatrix =0.
            jVector = get_jVector(i,un)
            for j in jVector:
                fluxij = fluxMatrix[i,j] - limitedFlux[i,j]
                Lij = min(Rpos[i],Rneg[j]) if fluxij>0 else min(Rneg[i],Rpos[j])

                # Compute the limited flux
                ith_limiter_times_fluxCorrectionMatrix += Lij*fluxij

                # Update vectors for next FCT Iteration #
                limitedFlux[i,j] = Lij*fluxij
                fluxMatrix[i,j] = fluxij

                min_Lij = min(min_Lij,Lij)
            #
            uLim[i] += ith_limiter_times_fluxCorrectionMatrix
            limited_flux_correction[i] += ith_limiter_times_fluxCorrectionMatrix
        #
    #
    return limited_flux_correction,min_Lij
#
