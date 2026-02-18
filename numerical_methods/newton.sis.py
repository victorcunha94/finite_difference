import numpy as np
import sympy 
import scipy

tol = 1e-3

def nonlinear_sis(x):
    f1 = x[0] + x[1] - 3
    f2 = x[0]**2 + x[1]**2 - 9
    return np.array([f1, f2])


def Jacobiana(nonlinear_sis, x0, h=1e-6):
    n = len(x0)
    F_x = nonlinear_sis(x0)
    m = len(F_x)
    J = np.zeros((m, n))
    for j in range(n):
        xh = x0.copy()
        xh[j] = x0[j] + h
        F_h = nonlinear_sis(xh)
        J[:,j] = (F_h - F_x)/h
    return J
            

def Jac(f_nonlinear, x, n):
    J = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            J[i,j] = np.gradient(f_nonlinear[i], x[j])
    return J

x0 = [0.1, 0.4]
eps = 2.0
x_k = x0

while eps < tol:
    A = Jacobiana(nonlinear_sis, x_k)  
    F_x = nonlinear_sis(x_k)
    s_k = np.linalg.solve(A, F_x)
    x_k1 = x_k + s_k
    eps = x_k1 - x_k
    

print(s_k)
