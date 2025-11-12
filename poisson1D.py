import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
"""
Esse programa, resolve a equação de poisson em uma domínio unidimensional 

    u'' = f(x)  para  -2 < x < 2
    
    Com as seguintes condições de contorno
    
    u(xi) = alpha      u(xf) = beta 
Por conveniência, aplicamos como condição de contorno a função analítica.

"""

# Construção da malha
xi, xf = -2, 2
N = 514 # Número de pontos totais
# N_inter = N - 2
x, dx = np.linspace(xi, xf, N+1, retstep = True, endpoint = True)


### Declaração da solução analítica ###
sol_exact = lambda x: x**3
x_e = np.linspace(xi, xf, 200, endpoint= True)
s = sol_exact(x_e)

# Condição de contorno
def g(x):
    return sol_exact(x)

#Termo fonte
def f(x):
    return -6*x

# Montagem da Matriz do método centrado
vec_s = np.ones(N-2)
vec_p = -2*np.ones(N -1 )
vec_i = np.ones(N - 2)

diagonal_superior  = np.diag(vec_s, -1)
diagonal_principal = np.diag(vec_p, 0)
diagonal_inferior  = np.diag(vec_s,1)


#Escolhemos aqui dividir dx**2 por A, mas, poderíamos multiplicar F por dx**2
A = 1/(dx ** 2) * (diagonal_superior + diagonal_principal + diagonal_inferior)
F = np.zeros(N - 1) # Tamanho 4


for i in range(N-1): # i Varia de 0 à 3
    F[i] = - f(x[i + 1]) # F recebe os valores de f avaliados em x1, x2, x3, x4 -> f(x1), ...,f(x4)


# Implementação das condições de contorno no vetor F
F[0] -= g(xi) / dx**2  # Divisão das condições de contorno por dx**2
F[-1] -= g(xf) / dx**2


U_interno = solve(A, F)


# Não sei se tem uma forma elegante e até mais inteligente para fazer essa construção de U.
U = np.zeros(N + 1)
U[0] = g(xi)
U[1:-1] = U_interno[:]
U[-1] = g(xf)


# Cálculo do erro utilizando a norma do máximo
norm = np.max(np.abs(U - sol_exact(x)))
print(norm)

#Estudo de convergência
Vetor_N = [18, 34, 66, 130, 258, 514]


plt.figure(figsize=(8, 6))
plt.plot(x, U, 'b-', linewidth=2, label='Solução numérica')
plt.plot(x_e, s, 'r--', linewidth=2, label='Solução analítica')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparação: Solução Numérica vs Analítica')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()