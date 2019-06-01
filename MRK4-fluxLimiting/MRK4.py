import numpy as np
from matplotlib import pyplot as plt

# These functions are designed for a 1D periodic domain from [0,1] #

def compute_matrices(U,N):
    h=1/(N-1)
    D1=np.diag(np.ones(N-1),-1)-np.diag(np.ones(N-1),1)
    D1[0,-1]=1.    
    D1[-1,0]=-1.
    D1*=U/2/h

    # MPP artificial viscosity operator #
    DL=D1*0
    for i in range(N):
        for j in np.arange(i+1,N):
            DL[i,j]=np.max([-D1[i,j],0,-D1[j,i]])
            DL[j,i]=DL[i,j]
        DL[i,i]=-np.sum(DL[i,:])
        
    return D1,DL
#

def sdisc(u,D1,DL,compute_flux_matrices=True):
    fG = np.dot(D1+DL,u)
    fB = np.dot(-DL,u)
    
    fijG = D1*0
    fijB = D1*0
    
    if compute_flux_matrices:
        for i in range(len(D1)):
            jVector = [len(D1)-1,i+1] if i==0 else ([0,i-1] if i==len(D1)-1 else [i-1,i+1])
            for j in jVector:
                fijG[i,j] = D1[i,j]*u[j]-D1[j,i]*u[i] + DL[i,j]*(u[j]-u[i])
                fijB[i,j] = -DL[i,j]*(u[j]-u[i])                
            #
        #
    #
    return fG+fB,fG,fB,fijG,fijB
#

def limited_non_conservative_flux(dt,un,fG,fB,global_bounds=False,single_gamma=False):
    gamma=0*fB
    for i in range(len(fB)):
        # compute bounds 
        if global_bounds:
            umax=1
            umin=0
        else:
            jVector = [len(un)-1,i,i+1] if i==0 else ([i-1,i,0] if i==len(un)-1 else [i-1,i,i+1])
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

def limited_conservative_flux(un,uL,fij,global_bounds=False):
    min_Lij=1.0e10
    umax=-1.0e10; umin=1.0e10
    Rpos=un*0; Rneg=un*0; limited_flux_correction=un*0
    for i in range(len(un)):
        # compute bounds
        if global_bounds:
            umax=1.; umin=0.
        else:
            jVector = [len(un)-1,i,i+1] if i==0 else ([i-1,i,0] if i==len(un)-1 else [i-1,i,i+1])
            for j in jVector:
                umax=max(umax,un[j])
                umin=min(umin,un[j])
            #
            # check uL
            assert(umin<=uL[i] and uL[i]<=umax),"Error: low-order solutin is not on bounds"
        #
        jVector = [len(un)-1,i+1] if i==0 else ([0,i-1] if i==len(un)-1 else [i-1,i+1])
        Ppos=0; Pneg=0
        for j in jVector:
            # compute p vectors 
            Ppos += max(fij[i,j],0)
            Pneg += min(fij[i,j],0)
        # Compute R vectors 
        Rpos[i] = min(1.,(umax-uL[i])/Ppos) if Ppos!=0 else 1.0
        Rneg[i] = min(1.,(umin-uL[i])/Pneg) if Pneg!=0 else 1.0
    #
    # compute limited flux 
    for i in range(len(un)):
        jVector = [len(un)-1,i+1] if i==0 else ([i-1,0] if i==len(un)-1 else [i-1,i+1])
        for j in jVector:
            Lij = min(Rpos[i],Rneg[j]) if fij[i,j]>0 else min(Rneg[i],Rpos[j])
            min_Lij = min(min_Lij,Lij)
            limited_flux_correction[i] += Lij*fij[i,j]
        #
    #
    return limited_flux_correction,min_Lij
#



