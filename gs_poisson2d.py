import numpy as np
from numpy import linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt

"""
Esse programa, resolve a equação de poisson em uma domínio unidimensional 

    u''(x,y) = f(x,y)  para  -2 < x < 2, -2 < y < 2
    
    onde f(x, y) = -6x

    Com as seguintes condições de contorno

    u(xi) = alpha      u(xf) = beta 
Por conveniência, aplicamos como condição de contorno a função analítica..
"""

# Construção da malha
xi, xf, yi, yf = -2, 2, -2, 2
N = 10  # Número de espaçamentos
# N_inter = N - 2
x, dx = np.linspace(xi, xf, N + 1, retstep=True, endpoint=True)
y, dy = np.linspace(yi, yf, N + 1, retstep=True, endpoint=True)
mesh  = np.meshgrid(x, y,indexing='ij')


maxiter = 1000
tol     = 1e-6
print(float(tol))

### Declaração da solução analítica ###
sol_exact = lambda x, y: x ** 3 + y**3


def sol_exact_d(mesh):
    return mesh[0] ** 3 + mesh[1] ** 3


# Condição de contorno
def g(x,y):
    return sol_exact(x, y)

# Termo fonte
def f(x,y):
    return 6 * x +  6*y


u = np.zeros((N + 1, N + 1))
u[0, :] = g(x[0], y)
u[N, :] = g(x[N], y)
u[:, 0] = g(x, y[0])
u[:, N] = g(x, y[N])
un1 = np.zeros((N + 1, N + 1))

for iter in range(maxiter):
    for i in range(1, N):
        for j in range(1, N):
            un1[i, j] = 1/4 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - ((dx)**2)/4*f(x[i],y[j])

    erro_iter = np.linalg.norm(un1 - u, np.inf)
    u = un1.copy()

    if erro_iter < tol:
        print(f"Gauss seidel convergiu com {iter} iterações\n")
        print(f" com erro de {erro_iter:.10f}")
        break





sol_exact_d = sol_exact_d(mesh)

print(u)

print(sol_exact_d)

plt.plot(mesh[0], mesh[1], marker='o', color='k', linestyle='none')
plt.show()
Uerror = u - sol_exact_d



fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y, Uerror, shading="auto")
fig.colorbar(pcm, ax=ax)
plt.show()

# Gráfico da solução numérica
fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y, sol_exact_d, shading="auto")
fig.colorbar(pcm, ax=ax)
ax.set_title("Solução numérica - Gauss-Seidel")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
plt.show()