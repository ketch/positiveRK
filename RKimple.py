#This file conains implementation for an genreric RKM method with adaptive b control.
#It contains a seperate implementation for explicit RK methods and DIRK methods 


import numpy as np
import scipy.optimize as opt
from OrderCondition import *
import cvxpy as cp
from scipy.optimize import linprog



#Todo rewrite this function and encorporate a better handling of maxvals and make it possible to use vectors
def solve_LP(solver,O,rhs,rkm,u,K,dt,reduce = False,verbose_LP = False,maxval=None, **options): 
    """
    This method solves the LP Problem
    
    Parameters:
    
    solver:     a string with the solver that should be used
    O:          Order condition matrix
    rhs:        Order condition right hand side
    
    reduce:     If set, the LP problem is first solved with a reduced set of constriants
    verbose_LP: prints additional messages
    maxval:     If scalar enforce the value as maximum on solution. If not used set to inf
    options:    additional Options, are passed through to the used solver
    
    Returns:
    status: integer representing the status of the algorithm.
      For scipy
        0 : Optimization proceeding nominally.
        1 : Iteration limit reached.
        2 : Problem appears to be infeasible.
        3 : Problem appears to be unbounded.
        4 : Numerical difficulties encountered.
      For cvypy
        0 : Optimization proceeding nominally.
        2 : Problem appears to be infeasible.
        4 : Numerical difficulties encountered.
        5 : solver crashed
        6 : Trivial Problem
    l = Array with number of constraints
    b: found b, if solver failed rkm.b
    
    """
    s = len(rkm.b)

    if np.all(maxval==np.inf):
        maxval = None
                
    if solver == 'scipy_ip' or solver == 'scipy_sim':
        bounds = (0, None)
        A_eq = np.concatenate((O,-O),axis = 1)
        b_eq = rhs - O@rkm.b
            
        e = np.ones(2*s)
        
        if solver == 'scipy_ip':
            method = 'interior-point'
        elif solver == 'scipy_sim':
            method = 'revised simplex'
        
        
        if reduce:
            u_ = u +dt*K@rkm.b
            if maxval:
                i = (u_<0)|(u_>maxval)
            else:
                i = (u_<0)
            l = [] #List to store number of positifity constraints considered in LP-Problem
            if not np.any(i):
                #apparently we got the trivial problem that is already ok
                if verbose_LP: print('trival Problem for LP-Solver')
                return (6,rkm.b,[0])
            while not maxval and np.any(u_<0) or maxval and np.any((u_<0) | (u_>maxval)):
                if maxval:
                    i = (u_<0)|(u_>maxval)|i #update indecies for conditions
                else:
                    i = (u_<0)|i #update indecies for conditions
                if verbose_LP >= 2: print(np.sum(i),'constraints')
                l.append(np.sum(i))
                
                u_slice = u[i] #slice the u and K
                K_slice = K[i,:]
                
                if maxval:
                    if verbose_LP >= 3: print('using maxval')
                    A_ub_ = np.concatenate((-K_slice,K_slice),axis = 1)
                    A_ub = np.concatenate((A_ub_,-A_ub_),axis = 0)
                    b_ub_1 = 1/dt*u_slice+K_slice@rkm.b  
                    b_ub_2 = 1/dt*(maxval-u_slice)-K_slice@rkm.b
                    b_ub = np.concatenate((b_ub_1,b_ub_2),axis = 0)
                else:
                    if verbose_LP >= 3: print('not using maxval')
                    A_ub = np.concatenate((-K_slice,K_slice),axis = 1)
                    b_ub = 1/dt*u_slice+K_slice@rkm.b  
                
                res = linprog(e, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method,
                              options = options)
                
                b = rkm.b+res.x[:s]-res.x[s:]
                u_ = u+dt*K@b 
                
                if (not maxval and np.any(u_<0) and np.all((u_<0)|i == i) 
                    or maxval and np.any((u_<0)|(u_>maxval)) and np.all((u_<0)|(u_>maxval)|i == i)): # there are no new conditions
                    if verbose_LP: print('number of conditions is not increasing')
                    break
      
        else:
            l = [len(u)]
            if maxval:
                    if verbose_LP >= 3: print('using maxval')
                    A_ub_ = np.concatenate((-K,K),axis = 1)
                    A_ub = np.concatenate((A_ub_,-A_ub_),axis = 0)
                    b_ub_1 = 1/dt*u+K@rkm.b  
                    b_ub_2 = 1/dt*(maxval-u)-K@rkm.b
                    b_ub = np.concatenate((b_ub_1,b_ub_2),axis = 0)
            else:
                    if verbose_LP >= 3: print('not using maxval')
                    A_ub = np.concatenate((-K,K),axis = 1)
                    b_ub = 1/dt*u+K@rkm.b    
            res = linprog(e, A_ub=A_ub, b_ub=b_ub,A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method,options = options)
        
        
        if res.success:
            b = rkm.b+res.x[:s]-res.x[s:]
            status = 0
        else:
            print('solver did not find solution')
            print('solver_state',res.status)
            status = res.status
            b = None
            
                    
                    
    else:
        ap_op =cp.Variable(s)
        an_op =cp.Variable(s)
        e = np.ones(s) #vector for objective Function, just generates the 1-Norm of b
        
        
        if reduce:
            u_ = u +dt*K@rkm.b
            if maxval:
                i = (u_<0)|(u_>maxval)
            else:
                i = (u_<0)
            l = [] #List to store number of positifity constraints considered in LP-Problem
            if not np.any(i):
                #apparently we got the trivial problem that is already ok
                if verbose_LP: print('trival Problem for LP-Solver')
                return (6,rkm.b,[0])
            while not maxval and np.any(u_<0) or maxval and np.any((u_<0)| (u_>maxval)):
                if maxval:
                    i = (u_<0)|(u_>maxval)|i #update indecies for conditions
                else:
                    i = (u_<0)|i #update indecies for conditions
                if verbose_LP >= 2: print(np.sum(i),'constraints')
                l.append(np.sum(i))
                
                u_slice = u[i] #slice the u and K
                K_slice = K[i,:]
                
                #solve problem for slices u and K
                if maxval:
                    if verbose_LP >= 3: print('using maxval')
                    prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                        [O@(ap_op-an_op+rkm.b)==rhs,u_slice+dt*K_slice@(ap_op-an_op+rkm.b)>=0, 
                        u_slice+dt*K_slice@(ap_op-an_op+rkm.b)<=maxval,ap_op>=0,an_op>=0]) 
                else:
                    if verbose_LP >= 3: print('not using maxval')
                    prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                        [O@(ap_op-an_op+rkm.b)==rhs,u_slice+dt*K_slice@(ap_op-an_op+rkm.b)>=0,ap_op>=0,an_op>=0]) 
                try:
                    prob.solve(solver=solver,**options)
                    b = ap_op.value - an_op.value + rkm.b
                except:
                    if verbose_LP: print('Solver crashed')
                    b = None
                    status = 5
                    break

                u_ = u+dt*K@b
                
                if (not maxval and np.any(u_<0) and np.all((u_<0)|i == i) 
                    or maxval and np.any((u_<0)|(u_>maxval)) and np.all((u_<0)|(u_>maxval)|i == i)): # there are no new conditions
                    if verbose_LP: print('number of conditions is not increasing')
                    break
                      
        else:
            l = [len(u)]
            if maxval:
                if verbose_LP >= 3: print('using maxval')
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [O@(ap_op-an_op+rkm.b)==rhs,u+dt*K@(ap_op-an_op+rkm.b)>=0,u+dt*K@(ap_op-an_op+rkm.b)<=0,ap_op>=0,an_op>=0]) 
            else:
                if verbose_LP >= 3: print('not using maxval')
                prob = cp.Problem(cp.Minimize(e@ap_op+e@an_op),
                    [O@(ap_op-an_op+rkm.b)==rhs,u+dt*K@(ap_op-an_op+rkm.b)>=0,ap_op>=0,an_op>=0]) 
            try:
                prob.solve(solver=solver,**options)
            except:
                if verbose_LP: print('Solver crashed')
                b = rkm.b
                status = 5
                
        if prob.status == cp.OPTIMAL:
            b = ap_op.value - an_op.value + rkm.b
            status = 0
        elif prob.status == cp.OPTIMAL_INACCURATE:
            if verbose_LP: print('LP-solve possibly inaccurate, Status:',prob.status)
            b = ap_op.value - an_op.value + rkm.b
            status = 4
        else:
            if verbose_LP: print('LP-solve encountered problem. Status:',prob.status)
            b = None
            status = 2


    return (status,l,b)
    


def calculate_stages_imp(t,dt,u,rkm,f,solver_eqs,verbose=False,solveropts={}):
    """
    Function to calculate the stagevalues for a diagonaly implicit RKM
    
    Paramters:
    t:          time at beginning of step
    dt:         Timestep
    u:          solution
    rkm:        RKM to use
    f:          right hand side
    solver_eqs: solver to solve the equation system of the stageequations
    solveropts: optins for solver as dict
    verbose:    print additional messages to terminal

    Returns:
    K:          Matrix with the function evaluations of f. K = [f(t',u'),...,f(t',u')]
    message:    A status message as string
    neg_stage:  A vector with the length s where neg_stage[i] = 1 if negative values occured for u' at stage i
    """
    s= len(rkm)
    c= rkm.c
    A= rkm.A   
    K = np.zeros([len(u),s])

    message = ''
    neg_stage = np.zeros(s)

    for i in range(s): #compute Stages      
        u_prime = u.copy() 
        for m in range(i):
            u_prime += dt*A[i,m]*K[:,m]
            
        K[:,i] = solver_eqs(t+c[i]*dt,u_prime,dt,A[i,i],f,**solveropts)
            
        if np.any(u_prime<-1.e-6):
            message = message + 'negative u\' at stage' + str(s) + '\n'
            neg_stage[i] = 1 #np.nonzero(u_prime > 1.e-6)
            if verbose: print('negative stagevalue found')
            if verbose >=2: print(i,u_prime) #print input to f(t,u) if it is negative
    return K,message,neg_stage



def calculate_stages_exp(t,dt,u,rkm,f,verbose=False):
    """
    Function to calculate the stagevalues for explicit RKM
    
    Paramters:
    t:          time at beginning of step
    dt:         Timestep
    u:          solution
    rkm:        RKM to use
    f:          right hand side
    verbose:    print additional messages to terminal

    Returns:
    K:          Matrix with the function evaluations of f. K = [f(t',u'),...,f(t',u')]
    message:    A status message as string
    neg_stage:  A vector with the length s where neg_stage[i] = 1 if negative values occured for u' at stage i
    """
    s= len(rkm)
    c= rkm.c
    A= rkm.A
    K = np.zeros([len(u),s])

    message = ''
    neg_stage = np.zeros(s)

    for i in range(s): #compute Stages   
        u_prime = u.copy()
        for m in range(i):
            u_prime += dt*A[i,m]*K[:,m]
            
        K[:,i] = f(t+c[i]*dt,u_prime)
            
        if np.any(u_prime<-1.e-6):
            message = message+ 'negative u\' at stage' + str(s) + '\n'
            neg_stage[i] = 1 #np.nonzero(u_prime > 1.e-6)
            if verbose: print('negative stagevalue found')
            if verbose >=2: print(i,u_prime) #print input to f(t,u) if it is negative
    return K,message,neg_stage

    

def adapt_b(rkm,K,dt,u,minval,maxval,tol_neg,tol_change,p,theta,solver,solveropts,verbose = False):
    """
    function to adapt the b to meke sure it complies with the boundaries
    Parameters:
    rkm:        RKM used
    K:          Matrix with stagevalues
    dt:         dt used to calculate the stagevalues
    u:          solution at timestep
    minval:     Minimum value, corrently only 0 supported
    maxval:     Maximum value,if not needed set so None
    tol_neg:    Which negativevalues are accepted for u
    tol_change: MAximum value for |K@(b-rkm.b)|_2 accepted
    p:          range of orders to try enforcing as iterabel
    theta:      factors of timesteps to try as iterable. Element of [0 to 1]^k
    solver:     solver to use
    solveropts: optins for the LP-Problem
    verbose:    Print additional messages

    return:
    success:     True if a new b could be found
    u_n:          u^{n+1}
    b:          The b used
    dt:         the dt used 
    message:    A status message as text
    status:     Status as dict

    """
    message = ''
    status = {}

    for i,the in enumerate(theta): #loop through all the sub-timesteps
        for p_new in p:     #loop through orders

            if verbose: print('Try: Order=',p_new,'Theta=',the)
            #Construct Order conditions
            O,rhs = OrderCond(rkm.A,rkm.c,order=p_new,theta=the)

            (status_LP,l,b) = solve_LP(solver,O,rhs,rkm,u,K,dt,maxval = maxval,**solveropts)
            
            if status_LP in [2,3,5]:
                #Error Handling for didn not work
                if verbose:    print('LP-Solve failed, probably infeasibel')
            else: #Did work, testing further
                u_n = u + dt*K@b
                if not (np.all(u_n >= minval-tol_neg) and np.all(u_n <= maxval+tol_neg)) : 
                    #got a solution form the LP solver that is stil not positive...
                    #do some error handling here
                    if verbose:    print('LP-Solve returned a b that leads to a negative solution')
                    if verbose >= 2:    print(min(u_n-minval)); print(u_n)
                else:
                    change = np.linalg.norm(K@(b-rkm.b))
                    if change > tol_change: # to big adaption...
                        #do some error handling here
                        if verbose:    print('a to big adaptation to the solution by changing the b')
                        if verbose >= 2: print('|K(b_new-b)|=',change)
                    else: #we got a acceptable solution
                        if verbose: print('found new b')
                        return True, u_n,b, dt*the, message, change,p_new,the,status

    return False, None,np.zeros_like(rkm.b)*np.nan, 0, message, None, 0, 0,status

"""
            else:
                 #Run Optimisation Problem
                
                status['b'].append(n+1)
                
                if status_LP in [1,4]:
                    status['message'] = status['message'] + 'LP-solver reported Problem:'+ str(status_LP)+ 'at step' + str(n+1) + '\n'
                elif status_LP in [2,3,5]: 
                     status['success'] = False
                     status['message'] = status['message'] + 'LP-solver failed at step '+ str(n+1) + '\n'
                     break
                status['change'].append(np.linalg.norm(K@(b-rkm.b)))
                status['nup_pos_constriants'][n]=l
"""

class Solver:
    def __init__(self, rkm = None,dt = None,t_final = None,b_fixed = None,tol_neg=None,
                tol_change=None,p=None,theta=None,solver=None,LP_opts=None,solver_eqs = None,fail_on_requect = True):
        self.rkm = rkm #        Base Runge-Kutta method, in Nodepy format
        self.dt = dt#         time step size
        self.t_final = t_final #    final solution time  
        self.b_fixed = b_fixed #    if True rkm.b are used
        self.tol_neg = tol_neg #    Which negativevalues are accepted for u
        self.tol_change = tol_change # Maximum value for |K@(b-rkm.b)|_2 accepted
        self.p = p#        range of orders to try enforcing as iterabel
        self.theta = theta#     factors of timesteps to try as iterable. Element of [0 to 1]^k
        self.solver = solver#    the solver used for solving the LP Problem
        self.LP_opts = LP_opts#:    Dict containing options for LP-solver
        self.solver_eqs = solver_eqs
        self.fail_on_requect = fail_on_requect#if True breaks if there is no fesible b

    def __str__(self):
        string =    ("RKM:           " +self.rkm.name + "\n" +
                    "dt:            " +str(self.dt) + "\n" +
                    "t_final:       " +str(self.t_final) + "\n" +
                    "b_fixed:       " +str(self.b_fixed) + "\n" +
                    "tol_neg:       " +str(self.tol_neg) + "\n" +
                    "tol_change:    " +str(self.tol_change) + "\n" +
                    "p:             " +str(self.p) + "\n" +
                    "theta:         " +str(self.theta) + "\n" +
                    "solver:        " +str(self.solver) + "\n" +
                    "LP_opts:       " +str(self.LP_opts) + "\n" +
                    "solver_eqs:    " +str(self.solver_eqs) + "\n" +
                    "fail on re:    " +str(self.fail_on_requect) + "\n" )
        return string
                



class Problem:
    def __init__(self, f=None, u0 = None,minval = None,maxval = None,description = ''):
        self.f = f #        RHS of ODE system
        self.u0 = u0#         Initial data
        self.minval = minval#
        self.maxval = maxval#        Limits for Problem
        self.description = description

    def __str__(self):
        return self.description

def RK_integrate(solver = [], problem = [],dumpK=False,verbose=False):

    """    
    Options:
        solver: Solver Object with the fields:
            rkm:        Base Runge-Kutta method, in Nodepy format
            dt:         time step size
            t_final:    final solution time  
            b_fixed:    if True rkm.b are used
            tol_neg:    Which negativevalues are accepted for u
            tol_change: Maximum value for |K@(b-rkm.b)|_2 accepted
            p:          range of orders to try enforcing as iterabel
            theta:      factors of timesteps to try as iterable. Element of [0 to 1]^k
            solver:     the solver used for solving the LP Problem
            LP-opts:    Dict containing options for LP-solver
            solver_eqs: Solver for stageeqation for implicit method

        
        problem: Problem object with the fields:
            f:          RHS of ODE system
            u0:         Initial data
            minval:
            maxval:        Limits for Problem
                
        
        dumpK:      if True the stage values are also returned
        verbose:    if True function prints additional messages


    Returns:
        u:      Matrix with the solution
        t:      vector with the times
        b:      Matrix with the used b's

        if dumpK = True:
         K:     Array of the K Matrix containing the stagevalues
        if return_status
         status: dict containing
                'dt': the used dt
                'success': True if solver succeded, False if a inveasible or illdefined LP-Problem occured or the solver crashed
                'message': String containing more details
                'b':       Array with the indecies where b was changed
    """
    
    if not 'verbose_LP' in solver.LP_opts.keys():
        solver.LP_opts['verbose_LP'] = verbose


    #setup Variables for Soulution storage
    uu = [problem.u0]
    tt = [0]
    

    #setup Problem Solve
    explicit = solver.rkm.is_explicit()
    t = 0
    u = problem.u0
    dt= solver.dt

    #setup stepsize control
    dt_old = 1 #variable with the last tried dts


    success = True #For stepsize control at first step
    
    
    if dumpK:
        KK = ['null']


    #for debbugging bs
    bb = [solver.rkm.b]
        
    status = {
        'Solver':  solver,
        'Problem': problem,
        'success': True,
        'message': '',
        'neg_stage': [None],
        'LP_stat': [None],
        'b':[None],
        'change':[None],
        'order':[None],
        'theta':[None]
    }
    if verbose: print('set up, starting to solve')

    while t<solver.t_final:
        #Control new stepsize
        if t+dt > solver.t_final:
            dt = solver.t_final-t
        if not success:
            dt = 0.5 *dt_old
        if success:
            dt = solver.dt
        dt_old = dt    
        

        #Compute the new K
        if verbose: print('calculation new set of stagevalues for t =',t,'dt=',dt)
        if explicit:
            K,message,neg_stage = calculate_stages_exp(t,dt,u,solver.rkm,problem.f,verbose=verbose)
        else:
            K,message,neg_stage = calculate_stages_imp(t,dt,u,solver.rkm,problem.f,solver.solver_eqs,verbose=verbose)
        status['message'] += message
        status['neg_stage'].append(neg_stage)



        if dumpK:
            KK.append(K)
        
        #compute initial guess
        u_n = u +dt*K@solver.rkm.b

        if solver.b_fixed:
            u_n = u_n
            b = solver.rkm.b
            status['b'].append('o')
        else:
            if np.all(u_n >= problem.minval) and np.all(u_n <= problem.maxval):
                #everything is fine
                b = solver.rkm.b
                success = True
                #TODO add some additional code here as later nedded
                status['b'].append('o')
                status['LP_stat'].append(None)
                status['change'].append(None)
                status['order'].append(None)
                status['theta'].append(None)
            else:
                success,u_n,b,dt,message, change, order, the,LP_stat = adapt_b(solver.rkm,K,dt,u,problem.minval,problem.maxval,
                        solver.tol_neg,solver.tol_change,solver.p,solver.theta,solver.solver,solver.LP_opts,verbose = verbose)
                status['b'].append('c')
                status['message'] += message
                status['LP_stat'].append(LP_stat)
                status['change'].append(change)
                status['order'].append(order)
                status['theta'].append(the)

            
        if success:
            if verbose: print('advancing t')
            t += dt
            u = u_n
        else:
            if verbose: print('step reqect')
            if solver.fail_on_requect:
                status['b'][-1] = 'r'
                status['success'] = False
                break
            else:
                status['b'][-1] = 'r'


        bb.append(b)
        uu.append(u)
        tt.append(t)

        
    ret = (status,tt,uu,bb)
    if dumpK:
        ret= ret + (KK,)

    
    return ret









        
        
#For Implicit Methods

#Define a solver for the equation system 
def solver_Matrix(t,u,dt,a,A,preconditioner = None,verbose_solver = False):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    where f(t,u)=Au
    """ 
    x = np.linalg.solve(dt*a*A-np.eye(u.size),-A@u)
    #print(max(abs((dt*a*A-np.eye(u.size))@x+A@u)))
    return x
    
    




def solver_nonlinear(t,u,dt,a,f,verbose_solver = False):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    
    f is a function of t and u
    """ 
    stageeq = lambda x: f(t,u+dt*a*x)-x  # it seems like solving for the argument is better
    x, info, ier, mesg = opt.fsolve(stageeq,u,full_output=1)
    if ier != 1: 
        print(mesg)
    return x

def solver_nonlinear_arg(t,u,dt,a,f,verbose_solver = False,preconditioner=None):
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
    y, info, ier, mesg = opt.fsolve(stageeq,y_0,full_output=1)
    
    #check if solution is exact
    if np.any(np.abs(-y+u+dt*a*f(t,y))>0.0001):
        print('stageeq. solved non accurate')
        print(np.linalg.norm(-y+u+dt*a*f(t,y)))
        
    if np.any(u+dt*a*f(t,y)<0) and verbose_solver or np.any(u+dt*a*f(t,y)<-1e-8) :
        print('stageq solved with negative argument')
        print('res:')
        print(max(np.abs(-y+u+dt*a*f(t,y))))
        #print(u+dt*a*f(t,y))
        print('min:')
        print(min(u+dt*a*f(t,y)))
    
    if ier != 1: 
        print(mesg)
    return(f(t,y))
    
def solver_nonlinear_nk(t,u,dt,a,f,verbose_solver = False,preconditioner=None):
    """ 
    The function solves a equation system of the Form 
    x = f(t,u+dt*a*x)
    and returns x
    
    f is a function of t and u
    
    The method uses the Newton-Krylov solver from scipy
    """ 
    if preconditioner != None:
        y_0 = preconditioner(t,u,dt,a,f)
    else:
        y_0 = u
    
    #print('res orig:',np.linalg.norm(-u+u+dt*a*f(t,u)))
    #print('res new:',np.linalg.norm(-y_0+u+dt*a*f(t,y_0)))
    
    stageeq = lambda y: -y+u+dt*a*f(t,y)  
    
    y = opt.newton_krylov(stageeq,y_0)
    
    #check if solution is exact
    if np.any(np.abs(-y+u+dt*a*f(t,y))>1e-10):
        print('stageeq. solved non accurate')
        print(np.linalg.norm(-y+u+dt*a*f(t,y)))
        
    if np.any(u+dt*a*f(t,y)<0) and verbose_solver or np.any(u+dt*a*f(t,y)<-1e-8) :
        print('stageq solved with negative argument')
        print('res:')
        print(max(np.abs(-y+u+dt*a*f(t,y))))
        #print(u+dt*a*f(t,y))
        print('min:')
        print(min(u+dt*a*f(t,y)))
        
    return(f(t,y))

