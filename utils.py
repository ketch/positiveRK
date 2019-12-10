"""
This file contains utility functions for investigating properties of the rk methods

plot_cnvergence: function to plot the convergence of methods

find_dt: method for investigating certain dts using a bisect algorithm

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(time_integrator,rkm,f,u0,dt,refernce,step=1,error='abs',dx='1',Norm = 2,Params = {}):
    """"
    Parameters:
    time_integrator:    function used to integrate the ODE
    rkm,f,u0:           Arguments for time integrator
    dt:                 dt array with dts
    reference:          Array with reference solutions to compare the computet solution against 
    error:              Definition of error computation, one of 'abs','rel','grid'
    dx:                 Discretisation for grid norm
    Norm:               Norm to use for Error calculation ||u-u'||_Norm
    Params:             Parameters for time integrator
    
    Return:
    sol:                Array with the solutions used for calculationg the errors
    err:                Array with errors

    """
    
    err = np.zeros_like(dt)
    
    sol = []
    
    
    for i in range(dt.size):
        print('dt='+str(dt[i]))
        t,u,b = time_integrator(rkm,dt[i],f,u0,**Params)
        dif = refernce[:,i]-u[:,step]
        if error == 'abs':
            err[i] = np.linalg.norm(dif,ord=Norm)
        elif error == 'rel':
            err[i] = np.linalg.norm(dif,ord=Norm)/np.linalg.norm(refernce[:,i],ord=Norm)
        elif error == 'grid': #Grid function Norm (LeVeque Appendix A.5)
            error[i] = dx**(1/Norm)*np.linalg.norm(dif,ord=Norm)
        else:
            print('Error not defined')
            print(error)
            raise ValueError
        sol.append(u[:,step])
        
    plt.plot(dt,err,'o-')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.ylabel('Error')
    plt.xlabel('dt')
    
    return sol,err
    
        

def find_dt(time_integrator,rkm,f,u0,dt_start,cond = '',tol = 0.001,Params = {}):
    
    """"
    The function searchs for the upper bound of dt that sattisfies a condition.
    A bisect search approach is used

    Parameters:
    time_integrator:    function used to integrate the ODE
    rkm,f,u0:           Arguments for time integrator
    dt_start:           dt to start with

    cond: which condition to search for one of:
            'dt_pos'
            'dt_stable'
            'dt_feasible'

    tol:                the toleranc for dt
    Params:             Parameters for time integrator

    Returns:
    dt_sol:             The wanted dt
    dt:                 Array with the tried timesteps
    val:                Array with information if the condition was fullfiled for the timesteps
    
    """
    #Check the settings for the integator
    if 'b_fixed' in Params.keys():
        print('overiding setting "b_fixed"')

    if cond in ['dt_pos','dt_stable']:
        Params['b_fixed'] = True
    elif cond in ['dt_feasible']:
        Params['b_fixed'] = False
    else:
        print('no knwn conition')
        raise ValueError


    dt = np.array([0])
    val = np.array([True])

    run = 0 # number of iteration

    while True:
        #calculate new timestep 
        if run == 0:
            dt_new = dt_start
        elif len(dt[~val]) == 0: #No failed run so far 
            dt_new = 2*dt[-1]
        else:
            if val[-1] == False: #Last try failed
                inter = (max(dt[val]), dt[-1]) #between the highest succesfull run and the last run
            else: #last try succeded
                inter = (dt[-1], min(dt[~val])) #between last run and lowest failed run

            dt_new = 0.5 * np.sum(inter)
            
        dt = np.append(dt,dt_new)

        #run integration
        print('Testing:',dt[-1])
        t,u,b,status = time_integrator(rkm,dt[-1],f,u0,dumpK=False,return_status = True,**Params)

        #Test if condition is met
        if cond == 'dt_pos':
            if np.all(u > -1e-8):
                val_new = True
            else:
                val_new = False

        elif cond == 'dt_feasible':
            if status['success']:
                val_new = True
            else:
                val_new = False
        
        elif cond == 'dt_stable':
            n_start = np.linalg.norm(u[:,0])
            n_end = np.linalg.norm(u[:,-1])

            if n_end < 100*n_start:
                val_new = True
            else:
                val_new = False
        else:
            print('no knwn conition')
            raise ValueError
        
        val = np.append(val,val_new)
        print(val_new)

        #Test if we alredy know the solution
        if len(dt[~val]) > 0:
            if abs(min(dt[~val]) - max(dt[val])) < tol:
                dt_sol = max(dt[val])
                break

        run += 1
        if run == 500:
            print('500 iterations reached')
            raise ValueError

        if dt_new >= dt_start*1e5:
            print('Time is getting to big. Apparently the time is not valid. Setting to 0')
            dt_sol = 0
            break

    return (dt_sol,dt,val) 


def findall_dt(time_integrator,rkm,f,u0,dt_start,tol = 0.001,Params = {}):
    """"
    This function seachs for all important dt for a RKM.
    These are 'dt_pos','dt_stable' and 'dt_feasible'
    The find_dt() method is used


    Parameters:
    time_integrator:    function used to integrate the ODE
    rkm:                The RKM to use, if tuple then the times for all RKM are computet 
    f,u0:               Arguments for time integrator
    dt_start:           dt to start with
    tol:                the toleranc for dt
    
    Params:             Parameters for time integrator
    
    Returns:
    dt:                 dict with the dt's
    """

    conds = ('dt_pos','dt_stable','dt_feasible')
    dt = {}
    if not type(rkm) is tuple:
        for cond in conds:
            print('search for',cond)
            dt_sol,dt_,val_ = find_dt(time_integrator,rkm,f,u0,dt_start,cond = cond,tol = tol,Params = Params)
            dt[cond] = dt_sol
        return dt

    else: #check for more methods
        times = ()
        for rkm_ in rkm:
            print(rkm_)
            dt = findall_dt(time_integrator,rkm_,f,u0,dt_start,tol = tol,Params = Params)
            print(dt)
            times = times + (dt,)
        return times



def plot_times(methods,dt,effective = False,title = ''):
    """"
    Function to plot the dt for multiple methods.

    Paramters:
    methods:    tuple with the methods
    dt:         tuple with dicts of the methods
    effective:  If true plot the effective timesteps
    title:      String as title for the plot


    """


    labels = []
    stages = []
    dt_pos = []
    dt_stable = []
    dt_feasible = []

    for i in range(len(methods)):
        rkm = methods[i]
        labels.append(rkm.name)
        stages.append(len(rkm))
        dt_pos.append(dt[i]['dt_pos'])
        dt_stable.append(dt[i]['dt_stable'])
        dt_feasible.append(dt[i]['dt_feasible'])
    

    stages = np.array(stages)
    dt_pos = np.array(dt_pos)
    dt_stable = np.array(dt_stable)
    dt_feasible = np.array(dt_feasible)
  
    
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    if effective:
        rects1 = ax.bar(x - width, dt_pos/stages, width, label='dt_pos_eff')
        rects2 = ax.bar(x , dt_stable/stages, width, label='dt_stable_eff')
        rects3 = ax.bar(x + width, dt_feasible/stages, width, label='dt_feasible_eff')
    else:
        rects1 = ax.bar(x - width, dt_pos, width, label='dt_pos')
        rects2 = ax.bar(x , dt_stable, width, label='dt_stable')
        rects3 = ax.bar(x + width, dt_feasible, width, label='dt_feasible')    

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid()
    print(labels)
    