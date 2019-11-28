#This file defines Order conditions for Runge Kutta methods in the form O@b = r


import numpy as np
from scipy.optimize import fsolve
from OrderCondition import *
import cvxpy as cp


def RK_variable_b(rkm, dt, f, w0=[1.,0], t_final=1.,b_fixed = False,solver = cp.ECOS,
                 fallback = True,num_fallback = 1,dumpK=False,verbose=False):
    """    
    Options:
    
        rkm: Base Runge-Kutta method, in Nodepy format
        dt: time step size
        f: RHS of ODE system
        w0: Initial data
        t_final: final solution time        
    """
    
    #setup Variables for Soulution storage
    p = len(w0) #number of dimentions
    
    uu = np.zeros([p,int(t_final/dt)+100])
    uu[:,0] = w0.copy()
    tt = np.zeros([int(t_final/dt)+100])
    
    if dumpK:
        KK = ['null']
    
    
    #Setup Runge Kutta 
    c = rkm.c
    A = rkm.A #has to be lower left triangle
    s = len(c) #number of Stages
    K = np.zeros([p,s])
    
    u = np.array(w0)
    t = 0.
    n = 0
    
    
    if b_fixed == False:
        if fallback == True:
            O, rhs = OrderCond(rkm.A,rkm.c,order = rkm.p-num_fallback) #Fallback
        else:
            O, rhs = OrderCond(rkm.A,rkm.c,order = rkm.p) 
        ap_op =cp.Variable(s)
        an_op =cp.Variable(s)
        e = np.ones(s) #vector for gola Fnction, just generates the 1-Norm of b

    
    
          #Maybee set up Problem here and treat H as an Paramter
        
    #for debugging b's    
    bb = np.zeros([s,int(t_final/dt)+2])
        
    #print('set up starting to solve')
    
    #Solve ODE
    while t<t_final:
        for i in range(s):
            #compute Stages
            
                
            #K[:,i] = f(t+c[i]*dt,u+dt*K@A[i,:]) 
            #the 0s in A should make shure that no data from an older Step is used
            
            #Maybe better Approach, because A[i,j] = 0 in many places
            u_prime = u.copy()
            for m in range(i):
                u_prime += dt*A[i,m]*K[:,m]
            
            K[:,i] = f(t+c[i]*dt,u_prime)
            
            if np.any(u_prime<-1.e-5)and verbose: print(n+1,i,u_prime) #print input to f(t,u) if it is negative
            
            #print('intermediatestep computed')
            
        if dumpK:
            KK.append(K.copy())
        
        if b_fixed == False:
            #test if positifity is correct
            if (u+dt*K@rkm.b >= 0).all():
                b =rkm.b
            
            else:
            #Run Optimisation Problem
        
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                              [O@(ap_op-an_op+rkm.b)==rhs,u+dt*K@(ap_op-an_op+rkm.b)>=0,ap_op>=0,an_op>=0])  
                prob.solve(solver=solver)
                if prob.status == cp.OPTIMAL:
                    b = ap_op.value - an_op.value + rkm.b
                elif prob.status == cp.OPTIMAL_INACCURATE:
                    print(prob.status)
                    b = ap_op.value - an_op.value + rkm.b
                else:
                    print(prob.status)
                    print('Replaced b with original b')
                    b = rkm.b

                print('b changed')
        else:
            b =rkm.b
        #update
        u += dt*K@b
        n += 1
        t += dt
        
        uu[:,n] = u.copy()
        bb[:,n] = b.copy()
        tt[n] = t
        #print('updated')

    if dumpK:
        return (tt[0:n+1],uu[:,0:n+1],bb[:,0:n+1],KK)
    else:
        return (tt[0:n+1],uu[:,0:n+1],bb[:,0:n+1])
        

        
        
#For Implicit Methods

#Define a solver for the equation system 
def solver_Matrix(t,u,dt,a,A,preconditioner = None):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    where f(t,u)=Au
    """ 
    x = np.linalg.solve(dt*a*A-np.eye(u.size),-A@u)
    #print(max(abs((dt*a*A-np.eye(u.size))@x+A@u)))
    return x
    
    




def solver_nonlinear(t,u,dt,a,f):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    
    f is a function of t and u
    """ 
    stageeq = lambda x: f(t,u+dt*a*x)-x  # it seems like solving for the argument is better
    x, info, ier, mesg = fsolve(stageeq,u,full_output=1)
    if ier != 1: 
        print(mesg)
    return x

def solver_nonlinear_arg(t,u,dt,a,f,preconditioner=None):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    
    f is a function of t and u
    
    preconditioner: method for getting a starting point for fsolve (in terms of an y=u_start)
    """ 
    if preconditioner != None:
        y_0 = preconditioner(t,u,dt,a,f)
    else:
        y_0 = u
    
    #print('res orig:',np.linalg.norm(-u+u+dt*a*f(t,u)))
    #print('res new:',np.linalg.norm(-y_0+u+dt*a*f(t,y_0)))
    
    stageeq = lambda y: -y+u+dt*a*f(t,y)   
    y, info, ier, mesg = fsolve(stageeq,y_0,full_output=1)
    
    #check if solution is exact
    if np.any(np.abs(-y+u+dt*a*f(t,y))>0.0001):
        print('stageeq. solved non accurate')
        print(np.linalg.norm(-y+u+dt*a*f(t,y)))
        
    if np.any(u+dt*a*f(t,y)<0):
        print('res:')
        print(max(np.abs(-y+u+dt*a*f(t,y))))
        print('stageq solved with negative argument')
        #print(u+dt*a*f(t,y))
        print(min(u+dt*a*f(t,y)))
    
    if ier != 1: 
        print(mesg)
    return(f(t,y))
    
    


def RK_variable_b_implicit(rkm, dt, f, w0=[1.,0], t_final=1.,solver_eqs = solver_Matrix,preconditioner = None,
                           b_fixed = False,solver = cp.ECOS,fallback = True,num_fallback = 1,dumpK=False,verbose=False):
    """   
    for Diagonally Implicit methods
    Options:
    
        rkm: Base Runge-Kutta method, in Nodepy format
        dt: time step size, can be a float or a vector of timesteps
        f: Right hand side of ODE in appropiate form for the used solver
        w0: Initial data
        t_final: final solution time     
        solver_eqs: the solver for the apearing equation system of the form x = f(t,u+dt*a*x)
        f: Right hand side of ODE in appropiate form for the used solver
        num_fallback: How many orders the system should be reduced
    """
    
    dt_ = dt
    if isinstance(dt_, (list, tuple, np.ndarray)):
        t_final = sum(dt_)
    
    
    #setup Variables for Soulution storage
    p = len(w0) #number of dimentions
    
    if isinstance(dt_, (list, tuple, np.ndarray)):
        uu = np.zeros([p,dt.size+1])
        uu[:,0] = w0.copy()
        tt = np.zeros([dt.size+1])
    else:
        uu = np.zeros([p,int(t_final/dt)+100])
        uu[:,0] = w0.copy()
        tt = np.zeros([int(t_final/dt)+100])
        
    if dumpK:
        KK = ['null']
    
    #Setup Runge Kutta 
    c = rkm.c
    A = rkm.A #has to be lower left triangle
    s = len(c) #number of Stages
    K = np.zeros([p,s])
    
    u = np.array(w0)
    t = 0.
    n = 0
    
    
    #Setup Optimisation Problem
    if b_fixed == False:
        if fallback == True:
            O, rhs = OrderCond(rkm.A,rkm.c,order = rkm.p-num_fallback) #Fallback
        else:
            O, rhs = OrderCond(rkm.A,rkm.c,order = rkm.p) 
        ap_op =cp.Variable(s)
        an_op =cp.Variable(s)
        e = np.ones(s) #vector for gola Fnction, just generates the 1-Norm of b
    
    
          #Maybee set up Problem here and treat H as an Paramter
        
    #for debugging b's  
    if isinstance(dt_, (list, tuple, np.ndarray)):
        bb = np.zeros([s,dt.size+1])
    else:
        bb = np.zeros([s,int(t_final/dt)+100])
        
    #print('set up starting to solve')
    
    #Solve ODE
    while t<t_final:
        if isinstance(dt_, (list, tuple, np.ndarray)): #for adapted stepsizes
            dt = dt_[n]
        if t+dt>t_final:
            dt = t_final-t #MAtch final time exactly
            
        for i in range(s):
            #compute Stages
            
                
            #K[:,i] = f(t+c[i]*dt,u+dt*K@A[i,:]) 
            #the 0s in A should make shure that no data from an older Step is used
            
            #Maybe better Approach, because A[i,j] = 0 in many places
            u_prime = u.copy()
            for m in range(i):
                u_prime += dt*A[i,m]*K[:,m]
            
            K[:,i] = solver_eqs(t+c[i]*dt,u_prime,dt,A[i,i],f,preconditioner=preconditioner)
            
            if np.any(u_prime<-1.e-20)and verbose: print(n+1,i,u_prime) #print input to f(t,u) if it is negative
            #print('intermediatestep computed')
            
        if dumpK:
            KK.append(K.copy())
        
        if b_fixed == False:
            #test if positifity is correct
            if (u+dt*K@rkm.b >= 0).all():
                b =rkm.b
            
            else:
            #Run Optimisation Problem
        
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                              [O@(ap_op-an_op+rkm.b)==rhs,u+dt*K@(ap_op-an_op+rkm.b)>=0,ap_op>=0,an_op>=0])  
                prob.solve(solver=solver)
                if prob.status == cp.OPTIMAL:
                    b = ap_op.value - an_op.value + rkm.b
                elif prob.status == cp.OPTIMAL_INACCURATE:
                    print(prob.status)
                    b = ap_op.value - an_op.value + rkm.b
                else:
                    print(prob.status)
                    print('Replaced b with original b')
                    b = rkm.b

                
                print('b changed')
        else:
            b =rkm.b
        #update
        u += dt*K@b
        t += dt
        n += 1
        
        uu[:,n] = u.copy()
        bb[:,n] = b.copy()
        tt[n] = t
        #print('updated')
       
    if dumpK:
        return (tt[0:n+1],uu[:,0:n+1],bb[:,0:n+1],KK)
    else:
        return (tt[0:n+1],uu[:,0:n+1],bb[:,0:n+1])
    
