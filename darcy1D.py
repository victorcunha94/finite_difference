import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt


# Construção da malha
xi, xf = 1, 3
N = 7 # Número de pontos totais
x, dx = np.linspace(xi, xf, N, retstep = True, endpoint = True)
x_centro = np.linspace(xi + dx/2, xf - dx/2, N-1)
print(x_centro)
print(len(x_centro))

### Declaração da solução analítica ###
sol_exact = lambda x: np.sinh(x)
x_e = np.linspace(xi, xf, 200, endpoint=True)
s = sol_exact(x_e)


# Condição de contorno
def g(x):
    return sol_exact(x)

def k(x):
    return np.exp(x)

#Termo fonte
def f(x):
    return np.exp(2*x)
    #return  -4*np.pi**2*np.cos(2*np.pi*x)*x #-2*np.pi*(2*x*np.sin(2*np.pi*x)  + x**2 * 2*np.pi*np.cos(2*np.pi*x))


## Cálculo de k(x) nos vértices
k_vertices = np.zeros(N) #Tamanho 5
for v, t in enumerate(x):
    k_vertices[v] = k(t)

#####  WARNING  #####
# calcule o K(x) analítico nos centros, isso ajudará a entender melhor qual é o erro do código

# Cálculo de k(x) entre os vértices
k_centros = np.zeros(N-1) #Tamanho 4 o-----|-----|-----|-----o
print(k_centros)
for w in range(N-1): #Tamanho 4 - Ou seja, são 4 valores para w de 0 a 3
    #k_centros[w] = 2*(k_vertices[w] * k_vertices[w + 1])/(k_vertices[w] + k_vertices[w + 1])
    k_centros[w] = (k_vertices[w] + k_vertices[w + 1])/2
    #k_centros[w] = k(x_centro[w])

## Montagem do vetor diagonal de k(x)
vec_Ks = np.zeros(N - 3)
vec_Kp = np.zeros(N - 2)
vec_Ki = np.zeros(N - 3)

for q in range(N-2):
    vec_Kp[q] = -(k_centros[q] + k_centros[q + 1])

for z in range(N-3):
    vec_Ks[z] = k_centros[z+1]
    vec_Ki[z] = k_centros[z+1]

###### Montagem da Matriz do método centrado #####
diagonal_superior  = np.diag(vec_Ks, -1)
diagonal_principal = np.diag(vec_Kp, 0)
diagonal_inferior  = np.diag(vec_Ki,1)

#Escolhemos aqui dividir dx**2 por A, mas, poderíamos multiplicar F por dx**2
A = 1/(dx ** 2) * (diagonal_superior + diagonal_principal + diagonal_inferior)
print(A)


F = np.zeros(N-2) # Tamanho 3

for i in range(N - 2): # i Varia de 0 à 2
    F[i] = f(x[i+1])# F recebe os valores de f avaliados em x1, x2, x3, x4 -> f(x1), ...,f(x4)


# Implementação das condições de contorno no vetor F
F[0]  -= k_centros[0] * g(xi) / dx**2 # Divisão das condições de contorno por dx**2
F[-1] -= k_centros[-1] * g(xf) / dx**2

print(f"Forma da Matriz A: {np.shape(A)}")
print(f"Forma do Vetor F: {np.shape(F)}")

U_interno = solve(A, F)

print(U_interno)
# Não sei se tem uma forma elegante e até mais inteligente para fazer essa construção de U.
U = np.zeros(N)
U[0] = g(xi)
U[1:-1] = U_interno[:]
U[-1] = g(xf)


# Cálculo do erro utilizando a norma do máximo
norm = np.max(np.abs(U - sol_exact(x)))
print(norm)


plt.figure(figsize=(8, 6))
plt.plot(x, U, 'b-', linewidth=2, label='Solução numérica')
plt.plot(x_e, s, 'r--', linewidth=2, label='Solução analítica')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparação: Solução Numérica vs Analítica')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()