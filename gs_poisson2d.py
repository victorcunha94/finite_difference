from email.utils import unquote
from time import perf_counter

import numpy as np
from numpy import linalg
from numpy.linalg import solve
import matplotlib.pyplot as plt
from numpy.ma.core import append
import time


"""
Esse programa, resolve a equação de poisson em uma domínio unidimensional 

    u''(x,y) = f(x,y)  para  -1 < x < 1, -1 < y < 1
    
    onde f(x, y) = −2π 2sen(πx)sen(πy),

    Com as seguintes condições de contorno

    u(xi) = alpha      u(xf) = beta 
Por conveniência, aplicamos como condição de contorno a função analítica..
"""

method = "Aitken"
#method = "Gauss-Seidel_sor"
#method = "Jacobi"
#method = "Gauss-Seidel"
# Construção da malha
xi, xf, yi, yf = -1, 1, -1, 1
N = 64 # Número de espaçamentos

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
ugs = u.copy()



###############GEMINI######################
# Função auxiliar para dar APENAS UM passo de Gauss-Seidel puro
def passo_gauss_seidel(u_atual, dx, x, y, N, f):
    u_out = u_atual.copy()
    for i in range(1, N):
        for j in range(1, N):
            u_out[i, j] = 0.25 * (u_out[i - 1, j] + u_out[i + 1, j] + u_out[i, j - 1] + u_out[i, j + 1]) - (
                    (dx ** 2) / 4) * f(x[i], y[j])
    return u_out


if method == "Aitken":
    star = time.perf_counter()
    # 1. Configuração Inicial (Iteração 0)
    omega  = 0.2 # omega_0 inicial arbitrário
    omegas = []
    erros  = []

    u_tilde = passo_gauss_seidel(u, dx, x, y, N, f)
    r_old = u_tilde - u  # Primeiro resíduo r^(1)
    u = u + omega * r_old  # Primeira atualização u^(1)

    erro_primeiro = np.linalg.norm(r_old)  # aproximação do erro inicial
    erros.append(erro_primeiro)

    # 2. Loop Principal (A partir da iteração 1)
    for iter in range(1, maxiter):
        u_old_ciclo = u.copy()

        # Passo de Gauss-Seidel puro para obter a previsão não relaxada
        u_tilde = passo_gauss_seidel(u, dx, x, y, N, f)
        r_new = u_tilde - u  # Resíduo atual r^(v+2)

        # Diferença entre o resíduo atual e o anterior (denomador da fórmula)
        delta_r = r_new - r_old

        # Aplicação da Equação (33) usando produto escalar global np.sum(A * B)
        num   = np.sum(r_old * delta_r)
        denom = np.sum(delta_r * delta_r)

        if denom > 1e-12:
            omega = -omega * (num / denom)

        omegas.append(omega)


        # NOTA SOBRE O CLIP: O seu texto limita omega em (0, 1] devido ao problema de
        # Interação Fluido-Estrutura (FSI). Para a equação de Poisson clássica,
        # o omega pode passar de 1 (Super-relaxação). Se quiser restringir como o texto:
        # omega = np.clip(omega, 0.01, 1.0)

        # Atualiza a matriz u com o novo omega dinâmico
        u = u + omega * r_new

        # Guarda o resíduo atual para a próxima iteração
        r_old = r_new.copy()

        # Critério de parada baseado na mudança da solução
        erro_iter = np.linalg.norm(u - u_old_ciclo)
        erros.append(erro_iter)
        if erro_iter < tol:
            print(f"Aitken convergiu na iteração {iter}")
            print(f"Último Omega calculado: {omega:.6f}")
            print(f"Erro final: {erro_iter:.15f}\n")
            break
    end = time.perf_counter()

    print(f"Tempo final do Aitken = {end - star}")



omega_otimo = 2/(1 + np.sin(np.pi*dx)) #Cálculo do omega ótimo para o método SOR
print(omega_otimo)

if method == "Gauss-Seidel_sor":
    erros = []
    for iter in range(maxiter):
        u_old = u.copy()
        for i in range(1, N):
            for j in range(1, N):
                ugs[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - (((dx) ** 2) / 4) * f(x[i],y[j])
                u[i, j] = u[i, j] + omega_otimo * (ugs[i, j] - u[i, j])

        erro_iter = np.linalg.norm(u - u_old)
        # erro_iter = np.max(np.abs(u - u_old))
        erros.append(erro_iter)

        if erro_iter < tol:
            print(f"{method} convergiu com {iter} iterações\n")
            print(f" com erro de {erro_iter:.15f}")
            break


if method == "Gauss-Seidel":
    erros = []
    for iter in range(maxiter):
        u_old = u.copy()
        for i in range(1, N):
            for j in range(1, N):
                u[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - (((dx)**2)/4)*f(x[i],y[j])

        erro_iter = np.linalg.norm(u - u_old)
        #erro_iter = np.max(np.abs(u - u_old))
        erros.append(erro_iter)
        if erro_iter < tol:
            print(f"{method} convergiu com {iter} iterações\n")
            print(f" com erro de {erro_iter:.15f}")
            break


if method == 'Jacobi':
    erros = []
    for iter in range(maxiter):
        for i in range(1, N):
            for j in range(1, N):
                un1[i, j] = 1/4 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1]) - (((dx)**2)/4)*f(x[i],y[j])


        erro_iter = np.linalg.norm(u - un1)
        erros.append(erro_iter)
        u = un1.copy()

        if erro_iter < tol:
            print(f"{method} convergiu com {iter} iterações\n")
            print(f" com erro de {erro_iter:.15f}")

            break



sol_exact_d = sol_exact_d(mesh)
# Plotagem da malha
plt.plot(mesh[0], mesh[1], marker='o', color='k', linestyle='none')
#plt.show()

# Plotagem do erro
Uerror = u - sol_exact_d
fig, ax = plt.subplots()
ax.set_title(f"Erro numérico para N = {N} - (U - u_exato)")
pcm = ax.pcolormesh(x, y, Uerror, shading="auto")
fig.colorbar(pcm, ax=ax)
#plt.show()

# Gráfico da solução numérica
fig, ax = plt.subplots()
pcm = ax.pcolormesh(x, y, sol_exact_d, shading="auto")
fig.colorbar(pcm, ax=ax)
ax.set_title(f"Solução numérica - {method}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
#plt.show()


# plot dos Omegas de Aitken
fig, ax = plt.subplots(figsize=(10, 8))
ax.semilogy(range(len(erros)), erros, linewidth=2, color='#1f77b4')
ax.set_yscale('log')
ax.grid(True, linestyle="--")
ax.set_xlabel("Número de iterações")
ax.set_ylabel("Norma do Erro (Escala Log)")
ax.set_title(f"Histórico de convergência - {method}")
plt.show()
print(erros)




# 1. Calcula o Omega Ótimo Teórico do SOR para comparação
#x_omegas = list(range(0, len(omegas)))
omega_sor_teorico = 2 / (1 + np.sin(np.pi * dx))
fig2,ax2 = plt.subplots(figsize=(8, 6))
# --- GRÁFICO 2: Evolução do Omega Dinâmico ---
ax2.scatter(range(len(omegas)), omegas, color='blue', s=10, alpha=0.7, label='Omega Dinâmico (Aitken)')
# Desenha a linha horizontal do valor analítico ideal
ax2.axhline(y=omega_sor_teorico, color='black', linestyle='--', linewidth=1,
            label=rf'Omega Ótimo Teórico SOR ({omega_sor_teorico:.4f})')

ax2.grid(True, linestyle="--")
ax2.set_xlabel("Iteração")
ax2.set_ylabel("Valor de Omega")
ax2.set_title("Evolução e Adaptação do Fator de Relaxação")
ax2.legend()
plt.show()