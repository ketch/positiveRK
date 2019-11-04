import numpy as np
from matplotlib import pyplot as plt
from MRK4 import *

def run_MRK4(N,
             name=None,
             smooth=0,
             modify_RK=True,
             save_solution=None,
             reference_solution=None,
             dt_factor=1.0,
             verbose=0,
             global_bounds=False,
             gmin=0.0,
             gmax=1.0,
             num_iter=1):
    
    #######################
    # physical parameters #
    #######################
    L=1.0
    U=1.0
    T=1.0
    DT=0.1 # frequency for outputting
    # domain #
    xFull=np.linspace(0,L,N+1)
    x=xFull[0:-1]

    ########################
    # Numerical parameters #
    ########################
    h=L/(N) #if periodic: =L/N. Otherwise =L/(N-1)
    # compute matrices
    (D1,DL,dtMax)=compute_matrices(U,N)
    # compute time step
    C=0.8
    dt = C*dtMax
    dt/=dt_factor #1, 2, 4, 8, 16, 32... reference: 1024

    ######################
    # Initial conditions #
    ######################
    disc = (1.0*(x>0.45)*(x<0.55))
    sinusoidal = (0.5*(np.sin(2*np.pi*x)+1))
    tanh = (0.5*(np.tanh((x-0.3)/0.05)+1) - 0.5*(np.tanh((x-0.7)/0.05)+1))
    u0 = disc if smooth == 0 else (sinusoidal if smooth==1 else tanh)
    un_MRK4=1.0*u0

    # compute initial mass #
    initial_mass=h*np.sum(u0)

    #############
    # Time loop #
    #############
    mmin_Lij=1E10
    time=0
    aux_counter=0
    last_time_step=False
    while(True):
        # compute time
        time += dt

        # disc in space for RK4
        f1,f1G,f1B,f1ijG,f1ijB=sdisc(un_MRK4,D1,DL)
        f2,f2G,f2B,f2ijG,f2ijB=sdisc(un_MRK4+0.5*dt*f1,D1,DL)
        f3,f3G,f3B,f3ijG,f3ijB=sdisc(un_MRK4+0.5*dt*f2,D1,DL)
        f4,f4G,f4B,f4ijG,f4ijB=sdisc(un_MRK4+dt*f3,D1,DL)

        if modify_RK:
            fG=f1G # good flux is given by forward Euler
            # conservative RK-FCT #
            fij=(-5*f1ijG+f1ijB + 2*(f2ijG+f2ijB) +2*(f3ijG+f3ijB) + f4ijG+f4ijB)/6.0
            sfB,min_Lij = limited_conservative_flux(un_MRK4,un_MRK4+dt*fG,dt*fij,global_bounds=global_bounds,gmin=gmin,gmax=gmax,num_iter=num_iter)
            mmin_Lij = min(mmin_Lij,min_Lij)
            unp1_MRK4 = (un_MRK4 + dt*fG) + sfB
        else:
            unp1_MRK4 = un_MRK4 + dt/6.0*(f1+2*f2+2*f3+f4)
        #

        # update old solution
        un_MRK4 = unp1_MRK4[:]

        # To increase verbosity
        if (time>=DT*aux_counter) and verbose>=2:
            print ("\t Time: " + str(time))
            aux_counter+=1
        #

        # breaking the time loop #
        if last_time_step:
            break
        #
        if (time+dt)>=T:
            last_time_step=True
            dt=T-time
            assert (dt>0)
        #
    #

    # compute loss of mass
    loss_MRK4 = (initial_mass - h*np.sum(unp1_MRK4))/initial_mass

    # print errors
    if verbose>=1:
        print("\t (min, max) value of MRK4 at final time t=T: " + str(np.min(unp1_MRK4)) + " , " + str(np.max(unp1_MRK4)))
        #print('\t loss in mass of MRK4='+str(loss_MRK4))
        #print ('min Lij=', mmin_Lij)
        print ("\t L1 error with MRK4:  \t\t" + str(np.sum(np.abs(unp1_MRK4-u0))/np.sum(np.abs(u0))))
        print ("\t L-inf error with MRK4: \t" + str(np.max(np.abs(unp1_MRK4-u0))/np.max(np.abs(u0))))
    #

    # plot the solution
    if name is not None:
        # plot final solutions #
        plt.figure(figsize=([15,10]))
        uexact = u0
        plt.plot(xFull,[i for i in uexact]+[u0[0]],'--k',linewidth=3)
        plt.plot(xFull,[i for i in unp1_MRK4]+[unp1_MRK4[0]],'-r',linewidth=2)
        plt.plot(xFull,[i for i in unp1_MRK4]+[unp1_MRK4[0]],'*',linewidth=2)
        plt.legend(['Exact','Num Sol.'])
        plt.savefig(name)
    #

    # save solution (as reference) or use a reference solution to compute the errors #
    if save_solution is not None:
        np.savetxt(save_solution, unp1_MRK4, delimiter=",")
    elif reference_solution is not None:
        reference = np.genfromtxt(reference_solution, delimiter=",")
        print ('min,max='+str(unp1_MRK4.min())+" , "+str(unp1_MRK4.max()))
        error = h*np.sum(np.abs(unp1_MRK4-reference))
        print ("\t"+str(error))
    #
#

##############################
# ********** MAIN ********** #
##############################
num_iter=1 # iterations for flux limiting

# Exact global minimum and maximum of tanh initial condition
gmin=6.1441739107603865e-06
gmax=0.99932929973906703

#############################
# CONVERGENCE IN SPACE-TIME #
#############################
if True:
    print ("* Standard RK4 with smooth initial data *")
    for N in [21, 41, 81, 161, 321, 641, 1281]:
        run_MRK4(N=N,name=None, #'solN='+str(N)+'.png',
                 smooth=2,modify_RK=False,verbose=1)
    #

    print ("* Modified RK4 with global bounds and smooth initial data *")
    for N in [21, 41, 81, 161, 321, 641, 1281]:
        run_MRK4(N=N,name=None, #'solN='+str(N)+'.png',
                 smooth=2,modify_RK=True,global_bounds=True,verbose=1,num_iter=num_iter,gmin=gmin,gmax=gmax)
    #

    print ("* Modified RK4 with local bounds and smooth initial data *")
    for N in [21, 41, 81, 161, 321, 641, 1281]:
        run_MRK4(N=N,name=None, #'solN='+str(N)+'.png',
                 smooth=2,modify_RK=True,global_bounds=False,verbose=1,num_iter=num_iter)
    #
#

############################
# SELF CONVERGENCE IN TIME #
############################
N=21 # sin: 81, tanh: 321
if False:
    print ("************************************************")
    print ("********** Running reference solution **********")
    print ("************************************************")
    print ("* Standard RK4 with smooth initial data *")
    run_MRK4(N=N,name=None,smooth=2,modify_RK=False,save_solution='reference_RK4.csv',dt_factor=1024.0)

    print ("* Modified RK4 with global bounds and smooth initial data *")
    run_MRK4(N=N,name=None,smooth=2,modify_RK=True,save_solution='reference_MRK4_globalBounds.csv',dt_factor=1024.0,
             global_bounds=True,gmin=gmin,gmax=gmax,num_iter=num_iter)

    print ("* Modified RK4 with local bounds and smooth initial data *")
    run_MRK4(N=N,name=None,smooth=2,modify_RK=True,save_solution='reference_MRK4_localBounds.csv',dt_factor=1024.0,
             global_bounds=False,num_iter=num_iter)

    print ("*******************************************************")
    print ("********** Running self convergence solution **********")
    print ("*******************************************************")
    print ("*... Standard RK4 with smooth initial data ...*")
    for dt_factor in [1.0,2.0,4.0,8.0,16.0,32.0]:
        run_MRK4(N=N,name=None, #'sol_dt_factor'+str(int(dt_factor))+'.png',
                 smooth=2,modify_RK=False,reference_solution='reference_RK4.csv',dt_factor=dt_factor)
    #

    print ("*... Modified RK4 with global bounds and smooth initial data ...*")
    for dt_factor in [1.0,2.0,4.0,8.0,16.0,32.0]:
        run_MRK4(N=N,name=None, #'sol_dt_factor'+str(int(dt_factor))+'.png',
                 smooth=2,modify_RK=True,reference_solution='reference_MRK4_globalBounds.csv',dt_factor=dt_factor,
                 global_bounds=True,gmin=gmin,gmax=gmax,num_iter=num_iter)
    #

    print ("*... Modified RK4 with local bounds and smooth initial data ...*")
    for dt_factor in [1.0,2.0,4.0,8.0,16.0,32.0]:
        run_MRK4(N=N,name=None, #'sol_dt_factor'+str(int(dt_factor))+'.png',
                 smooth=2,modify_RK=True,reference_solution='reference_MRK4_localBounds.csv',dt_factor=dt_factor,
                 global_bounds=False,num_iter=num_iter)
#
