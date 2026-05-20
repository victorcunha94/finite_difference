import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

################# Geração da malha ####################
xl, xr, yt, yb = 0, 1, 1,0
N = 51
dx = (xl - xr)/(N + 1)
dy = (yt - yb)/(N + 1)
#######################################################

##### Criação da Matriz U
U = np.zeros(( N, N ))

#Condição inicial
def init(x, y):
  x = 2*x
  y = 2*y
  return x, y

#Condição de contorno
def cc(U):
  U[0, :]  = 0 # Top
  U[:,-1]  = 0 # Right
  U[-1,:]  = 0 # Bottom
  U[:, 0]  = 0 # Left
  return U

U0 = cc(U)

U0_vetor = np.reshape(U0, N*N)


def f(x, y):
  f = - np.sin(2*np.pi*x) + np.sin(2*np.pi*y)
  return f


def kron(A, B):
  "Produto de Kronecker"
  p = len(A)
  q = len(A[0])
  r = len(B)
  s = len(B[0])

  C = np.zeros((p*r, q*s))

  for i in range(p):
    for j in range(q):
      row = i * r
      for k in range(r):
        col = j * s
        for l in range(s):
          C[row, col] = A[i][j]*B[k][l]
          col += 1

        row += 1
  return C


def diff_2ord(xl, xr, yt, yb, U0, dx, dy, N, f):
  X = np.linspace(xl, xr, N, endpoint=True)
  Y = np.linspace(yt, yb, N, endpoint=True)
  x, y = np.meshgrid(X, Y, indexing='ij')



  a_local = 4 * np.ones(N)
  b_local = -1 * np.ones(N - 1)
  c_local = -1 * np.ones(N - 1)
  a = np.diag(a_local, 0)
  b = np.diag(b_local, 1)
  c = np.diag(c_local, -1)
  T = a + b + c

  ### Criação das diagonais secundárias
  e_local = -1 * np.ones(N-1)
  d = np.diag(e_local, -1)
  e = np.diag(e_local, 1)

  S = e + d
  i_local = -1*np.ones(N)
  I = np.diag(i_local, 0)

  ##### A e b #####
  A = (kron(I, T) + kron(S, I))/(dx*dy)**2
  i = f(x, y)
  i = np.reshape(i, N*N)

  #Solução do sistema linear A*U = b

  u = solve(A, i)

  return u, x, y



# print(U0[0])
Un, x, y = diff_2ord(xl, xr, yt, yb, U0, dx, dy, N, f)
u2d = Un.reshape((N, N))

plt.figure(figsize=(5,4))
plt.pcolormesh(x, y, u2d, cmap='viridis')
plt.show()
