import numpy as np
import sympy 
import scipy

"""
Este programa encontra uma solução aproximada para um sistema não linear, utilizando o método de Newton. A Jacobiana aqui é calculada utilizando diferenças finitas progressivas.

** Criar a animação das iterações para visualizar o método convergindo.

** Implementar problemas de Equações Diferenciais 

** Por didática, trocar o loop da montagem da jacobiana para dois loops, percorrendo dos os elementos da matriz

** Implementar outras aproximações de diferenças finitas, por exemplo, diferenças centradas.

** Encontrar uma forma de verificar a solução do sistema.

** Analisar o decaimento do erro para diferentes formas de diferenciação automática

"""
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


x0 = [1,5]
x_k = x0
iteration = 0
norm = 20
while norm > tol:
    A = Jacobiana(nonlinear_sis, x_k)  
    F_x = nonlinear_sis(x_k)
    s_k = np.linalg.solve(A, -F_x)
    x_k = x_k + s_k
    norm = np.linalg.norm(F_x, np.inf)
    iteration += 1
 
 
print(f'Solução do sistema é dada por: {x_k}')
print(A)
print(norm)
print(f'{iteration} Iterações')
