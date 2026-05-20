import numpy as np
import time

Ax = 5500
Ay = 5500

Bx = 5500
By = 5500

A = np.random.randint(1, 1000, size=(Ax, Ay))
B = np.random.randint(1, 1000, size=(Bx, By))

C = np.zeros_like(A)
N = len(A)

#Vamos tratar o caso em que A_n,p x B_pxm = C
def prod_matrix(A, B):
    Ax = np.shape(A[0])
    Ay = np.shape(A[1])

    Bx = np.shape(B[0])
    By = np.shape(B[1])

    if Ay != Bx :
        print("O número de linhas de A é distinto do número de colunas de B!\n")
        print("Adicione matrizes válidas!")
        exit()

    else:
        t_start = time.time()
        for j in range(N):
            for i in range(N):
                C[i, j] = A[i, j] * B[i, j]
        t_end = time.time()

        t = t_end - t_start

    return C, t


C = prod_matrix(A, B)
print(f"{C[0]} no tempo de {C[1]} segundos")