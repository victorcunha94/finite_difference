from email.utils import unquote

import numpy as np
from numpy import linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt

"""
Esse programa, resolve a equação de poisson em uma domínio unidimensional 

    u''(x,y) = f(x,y)  para  -1 < x < 1, -1 < y < 1
    
    onde f(x, y) = −2π 2sen(πx)sen(πy),

    Com as seguintes condições de contorno

    u(xi) = alpha      u(xf) = beta 
Por conveniência, aplicamos como condição de contorno a função analítica..
"""


#method = "Jacobi"
method = "Gauss-Seidel"
# Construção da malha
xi, xf, yi, yf = -1, 1, -1, 1
N = 32  # Número de espaçamentos

#Criação da malha bidimensional
x, dx = np.linspace(xi, xf, N + 1, retstep=True, endpoint=True)
y, dy = np.linspace(yi, yf, N + 1, retstep=True, endpoint=True)
mesh  = np.meshgrid(x, y,indexing='ij')


maxiter = 5000
tol     = 1e-6


### Declaração da solução analítica ###
sol_exact = lambda x, y: np.sin(np.pi*x) * np.sin(np.pi*y)


def sol_exact_d(mesh):
    return np.sin(np.pi*mesh[0]) * np.sin(np.pi*mesh[1])

# Condição de contorno
def g(x,y):
    return sol_exact(x, y)

# Termo fonte
def f(x,y):
    return  -2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)


u = np.zeros((N + 1, N + 1))
u[0, :] = g(x[0], y)
u[N, :] = g(x[N], y)
u[:, 0] = g(x, y[0])
u[:, N] = g(x, y[N])
un1 = np.copy(u)

if method == "Gauss-Seidel":
    for iter in range(maxiter):
        u_old = u.copy()
        for i in range(1, N):
            for j in range(1, N):
                u[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - (((dx)**2)/4)*f(x[i],y[j])

        #erro_iter = np.linalg.norm(u - u_old)
        erro_iter = np.max(np.abs(u - u_old))


        if erro_iter < tol:
            print(f"{method} convergiu com {iter} iterações\n")
            print(f" com erro de {erro_iter:.15f}")
            break


if method == 'Jacobi':
    for iter in range(maxiter):
        for i in range(1, N):
            for j in range(1, N):
                un1[i, j] = 1/4 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - (((dx)**2)/4)*f(x[i],y[j])


        erro_iter = np.linalg.norm(u - un1)
        u = un1.copy()

        if erro_iter < tol:
            print(f"{method} convergiu com {iter} iterações\n")
            print(f" com erro de {erro_iter:.15f}")

            break



sol_exact_d = sol_exact_d(mesh)
# Plotagem da malha
plt.plot(mesh[0], mesh[1], marker='o', color='k', linestyle='none')
plt.show()

# Plotagem do erro
Uerror = u - sol_exact_d
fig, ax = plt.subplots()
ax.set_title(f"Erro numérico para N = {N} - (U - u_exato)")
pcm = ax.pcolormesh(x, y, Uerror, shading="auto")
fig.colorbar(pcm, ax=ax)
plt.show()

# Gráfico da solução numérica
fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y, sol_exact_d, shading="auto")
fig.colorbar(pcm, ax=ax)
ax.set_title(f"Solução numérica - {method}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.show()