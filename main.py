import numpy as np
from numpy.linalg import solve

################# Geração da malha ####################
xl, xr, yt, yb = 0, 1, 1,0
N = 4
X = np.linspace(xl, xr, N, endpoint=True)
Y = np.linspace(yt, yb, N, endpoint=True)
x, y = np.meshgrid(X, Y, indexing='ij')
dx = (xl - xr)/(N + 1)
dy = (yt - yb)/(N + 1)
#######################################################

##### Criação da Matriz U
U = np.zeros((N, N ))

#Condição inicial
def init(x, y):
  x = 2*x
  y = 2*y
  return x, y

#Condição de contorno
def cc(U):
  U[0, :]  = 2 # Top
  U[:,-1]  = 1 # Right
  U[-1,:]  = 1 # Bottom
  U[:, 0]  = 2 # Left
  return U

U0 = cc(U)

print(U0)
U0_vetor = np.reshape(U0, N*N)
print(U0_vetor)


######### Discretização da equação de Poisson 2D ##########
def f(x, y):
  f = np.sin(2*np.pi*x) + np.sin(2*np.pi*y)
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


def diff_2ord(U0, dx, dy, N):
  U0 = U0

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
  b = U0
  b = np.reshape(b, N*N)

  #Solução do sistema linear A*U = b

  u = solve(A, b)

  return u


def jacobi(U, Unew, dx, dy, N):
  U = U[0]
  Unew = Unew[0]
  for k in range(0, 4000):
    for i in range(N - 1):
      for j in range(N - 1):
        Unew[i, j] = 0.25*(U[i - 1, j] + U[i + 1, j] + U[i, j + 1] + dx*dy*f(i, j) )
        U[i, j] = Unew[i, j]
  return Unew




# print(U0[0])
Un = diff_2ord(U0, dx, dy, N)
# #Un = jacobi(U0, U_new, dx, dy, N)
print(Un)