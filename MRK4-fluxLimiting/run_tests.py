import numpy as np
from matplotlib import pyplot as plt
from MRK4 import *

global_bounds=True

## ************************ ##
## ***** STANDARD FCT ***** ##
## ************************ ##
def run_FCT(N,name,smooth=False):
    USE_FCT=True
    
    # physical parameters #
    U=1
    T=1.0
    DT=0.1
    x=np.linspace(0,1,N)

    # Numerical parameters #
    h=1.0/(N-1);
    C=0.75
    dt=C*h*U
    
    # Initial conditions #
    u0 = (0.5*(np.sin(2*np.pi*x)+1)) if smooth==True else (1.0*(x>0.45)*(x<0.55))
    un=1.0*u0;

    (D1,DL)=compute_matrices(U,N)
    initial_mass=h*np.sum(u0)
    
    # Time loop #
    NT=T/dt
    times = np.linspace(0,T,int(NT)+1)
    uu = np.zeros([len(u0),int(NT)+1])
    i=0
    for t in times:
        # First stage #
        f,fG,fB,fijG,fijB=sdisc(un,D1,DL) # spatial discretization
        sf,_ = limited_conservative_flux(un,un+dt*fG,dt*fijB,global_bounds=global_bounds) # limit anti-diffusive fluxes 
        sf/=dt
        u1 = un + dt*(fG + (sf if USE_FCT else fB))
        
        # Second stage 
        f,fG,fB,fijG,fijB=sdisc(u1,D1,DL)
        sf,_ = limited_conservative_flux(u1,u1+dt*fG,dt*fijB,global_bounds=global_bounds)
        sf/=dt
        u2 = 3./4*un + 1./4*(u1 + dt*(fG + (sf if USE_FCT else fB)))
        
        # Third stage
        f,fG,fB,fijG,fijB=sdisc(u2,D1,DL)
        sf,_ = limited_conservative_flux(u2,u2+dt*fG,dt*fijB,global_bounds=global_bounds)
        sf/=dt
        u3 = 1./3*un + 2./3*(u2 + dt*(fG + (sf if USE_FCT else fB)))

        # update solution at new time step
        unp1 = u3
    
        # update old solution 
        un = unp1
        
        # save solution for animation 
        uu[:,i] = unp1
        i+=1
    #
    loss = (initial_mass - h*np.sum(unp1))/initial_mass

    print('min, max value of (FCT solution)=',np.min(uu),np.max(uu))
    print('loss in mass with FCT solution=',loss)

    # plot final solutions #
    plt.figure(figsize=([15,10]))
    plt.plot(x,unp1,'-c',linewidth=1)
    plt.plot(x,u0,'--k',linewidth=3)
    plt.legend(['FCT solution','Exact'])
    plt.savefig(name)
#

def run_MRK4(N,name,smooth=False):
    # physical parameters #
    U=1
    T=1.0
    DT=0.1
    x=np.linspace(0,1,N)

    # Numerical parameters #
    h=1.0/(N-1);
    C=0.8
    dt=C*h*U
    dt/=1.0

    # Initial conditions #
    u0 = (0.5*(np.sin(2*np.pi*x)+1)) if smooth==True else (1.0*(x>0.45)*(x<0.55))
    un_RK4=1.0*u0;
    un_MRK4=1.0*u0;
    
    (D1,DL)=compute_matrices(U,N)
    initial_mass=h*np.sum(u0)
    
    # Time loop #
    NT=T/dt
    times = np.linspace(0,T,int(NT)+1)
    uu_MRK4 = np.zeros([len(u0),int(NT)+1])
    uu_RK4 = np.zeros([len(u0),int(NT)+1])
    index=0
    mmin_Lij=1.0e10 + np.zeros(int(NT)+1)
    for t in times:
        # RK4 without stabilization in space
        f1,_,_,_,_=sdisc(un_RK4,D1,DL)
        f2,_,_,_,_=sdisc(un_RK4+0.5*dt*f1,D1,DL)
        f3,_,_,_,_=sdisc(un_RK4+0.5*dt*f2,D1,DL)
        f4,_,_,_,_=sdisc(un_RK4+dt*f3,D1,DL)
        unp1_RK4 = un_RK4 + dt/6.0*(f1+2*f2+2*f3+f4)

        # MRK4: stabilization in space with limiters
        f1,f1G,f1B,f1ijG,f1ijB=sdisc(un_MRK4,D1,DL)
        f2,f2G,f2B,f2ijG,f2ijB=sdisc(un_MRK4+0.5*dt*f1,D1,DL)
        f3,f3G,f3B,f3ijG,f3ijB=sdisc(un_MRK4+0.5*dt*f2,D1,DL)
        f4,f4G,f4B,f4ijG,f4ijB=sdisc(un_MRK4+dt*f3,D1,DL)
        fG=f1G # good flux is given by forward Euler 
    
        # conservative RK-FCT #
        fij=(-5*f1ijG+f1ijB + 2*(f2ijG+f2ijB) +2*(f3ijG+f3ijB) + f4ijG+f4ijB)/6.0
        sfB,min_Lij = limited_conservative_flux(un_MRK4,un_MRK4+dt*fG,dt*fij,global_bounds=global_bounds)
        mmin_Lij[index] = min(mmin_Lij[index],min_Lij)
        unp1_MRK4 = (un_MRK4 + dt*fG) + sfB
        
        # update old solution 
        un_RK4 = unp1_RK4
        un_MRK4 = unp1_MRK4

        # save solution  
        uu_RK4[:,index] = unp1_RK4
        uu_MRK4[:,index] = unp1_MRK4

        index+=1
    #
    loss_RK4 = (initial_mass - h*np.sum(unp1_RK4))/initial_mass
    loss_MRK4 = (initial_mass - h*np.sum(unp1_MRK4))/initial_mass

    #print('(min, max) value of RK4=',np.min(uu_RK4),np.max(uu_RK4))
    #print('(min, max) value of MRK4=',np.min(uu_MRK4),np.max(uu_MRK4))
    print('(min, max) value of RK4 at final time t=T=',np.min(unp1_RK4),np.max(unp1_RK4))
    print('(min, max) value of MRK4 at final time t=T=',np.min(unp1_MRK4),np.max(unp1_MRK4))

    print('loss in mass of (RK4, MRK4)=',loss_RK4,loss_MRK4)
    print ('min Lij=', mmin_Lij.min())

    print ("L1 error with RK4: \t", h*np.sum(np.abs(unp1_RK4-u0))/np.sum(np.abs(u0)))
    print ("L1 error with MRK4: \t", h*np.sum(np.abs(unp1_MRK4-u0))/np.sum(np.abs(u0)))

    # plot final solutions #
    plt.figure(figsize=([15,10])) 
    plt.plot(x,u0,'--k',linewidth=3)
    plt.plot(x,unp1_RK4,'-b',linewidth=2)
    plt.plot(x,unp1_MRK4,'-r',linewidth=2)
    plt.legend(['Exact','RK4','MRK4'])
    plt.savefig(name)
#

N=41
#print ("***** FCT with non smooth initial data *****")
#run_FCT(N=N,name='FCT_non_smooth.png',smooth=False)

#print ("***** Modified RK4 with non smooth initial data *****")
#run_MRK4(N=N,name='MRK4_non_smooth.png',smooth=False)

print ("***** Modified RK4 with smooth initial data *****")
run_MRK4(N=N,name='MRK4_smooth.png',smooth=True)
