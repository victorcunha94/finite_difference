import numpy as np
#from main import kron
from scipy.sparse import csr_matrix

N = 3
M = 3

Matriz = np.zeros((M, N))
Diagonal = np.ones(M)
#Diagonal = np.ones(N)
Matriz = np.diag(Diagonal, k=0)
Matriz = csr_matrix(Matriz)
#print(Matriz)

##### Criação do bloco da diagonal ####
a_local = 4 * np.ones(N)
b_local = -1 * np.ones(N - 1)
c_local = -1 * np.ones(N - 1)
a = np.diag(a_local, 0)
b = np.diag(b_local, 1)
c = np.diag(c_local, -1)
T = a + b + c


### Criação das diagonais secundárias
e1_local = 1 * np.ones(N-1)

d = np.diag(e1_local, -1)
e = np.diag(e1_local, 1)

S = d + e

i_local = -1*np.ones(N)
I = np.diag(i_local, 0)

##### Diagonais da matriz A #####
A = (np.kron(I, T) + np.kron(S, I))
#A = csr_matrix(A)

print(A)
print(A[1:-1, 5])

H = A[0:4, 1] + A[0:4, 1] + A[0:4, 1]
print(H)
# Verificar a criação de matriz sparsa com Scipy
