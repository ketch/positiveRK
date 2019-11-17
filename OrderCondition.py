#This file defines Order conditions for Runge Kutta methods in the form O@b = r


import numpy as np

def tau(k,A,c): 
    #generates tau vector 
    return 1/ np.math.factorial(k)*c**k - 1/np.math.factorial(k-1) *A @ c**(k-1)
    
def tau_hat(k,A,c):
    return c**k-k*A@(k-1)


def OrderCond(A,c,order = 1):
    #Generates Order Condition Matrix O and right side vector r for Linear Equation System O@b=r
    
    s = len(c) #number of stages
    
    r = []
    O_rows = []
    
    
    if A.shape != (s,s):
        raise InputError
        
    else:
        if order >= 1:
            O_rows.append(np.ones(s));      r.append(1)
            
        if order >= 2:
            O_rows.append(c);               r.append(1/2)
            
        if order >= 3:
            O_rows.append(c**2);            r.append(1/3)
            O_rows.append(tau(2,A,c));      r.append(0.)
            
        if order >= 4:
            O_rows.append(c**3);            r.append(1/4)
            O_rows.append(tau(2,A,c)*c);    r.append(0.)
            O_rows.append(tau(2,A,c)@A.T);  r.append(0.)
            O_rows.append(tau(3,A,c));      r.append(0.)
        
        if order >= 5:
            O_rows.append(c**4);                     r.append(1/5)
            O_rows.append(A@np.diag(c)@tau(2,A,c));  r.append(0.)
            O_rows.append(A@A@tau(2,A,c));           r.append(0.)
            O_rows.append(A@tau(3,A,c));             r.append(0.)
            O_rows.append(tau(4,A,c));               r.append(0.)
            O_rows.append(np.diag(c)@A@tau(2,A,c));  r.append(0.)
            O_rows.append(np.diag(c)@tau(3,A,c));    r.append(0.)
            O_rows.append(np.diag(c**2)@tau(2,A,c)); r.append(0.)
            O_rows.append(tau(2,A,c)**2);            r.append(0.)
        if order >= 6:
            print('too high order')
            raise NotImplementedError
        
        O = np.vstack(O_rows)
        return (O,np.array(r))
            
                

