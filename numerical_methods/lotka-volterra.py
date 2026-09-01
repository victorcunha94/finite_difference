#autor: github.com/victorcunha94


import numpy as np
import matplotlib.pyplot as plt
from newtonsys import*

N = 256
t, h = np.linspace(0, 25.0,N, retstep=True)

u1 = np.zeros(N)
u2 = np.zeros(N)

#Condições iniciais
u10 = 1.5
u20 = 1.0
u1[0] = u10
u2[0] = u20

un1 = np.zeros((2,N))
un1[0:] = u1
un1[1:] = u2


def f(u):
    f1 = -2 * u[0] + u[0] * u[1]
    f2 =      u[1] - u[0] * u[1]
    return np.array([f1, f2])


def jacobiana(f, un1):
    dG1 = [1 + 2*h - h*un1[1],  -h*un1[0]]
    dG2 = [h*un1[1],      1 - h + h*un1[0]]
    return np.array([dG1, dG2])


def taylor_q1(un1, f, h):
    for n in range(0, N-1): #N - 1 pq?
        un1[:,n+1] = un1[:,n] + h * f(un1[:,n])
    return un1


def taylor_q1_implicit(un1, f, h):
    for n in range(0, N-1): #N - 1 pq?
        def G(u):
            return u - un1[:, n] - h * f(u)

        un1[:,n+1]  = newtonsys(G, jacobiana, un1[:,n])
    return un1


def taylor_q2_trapezio(un1, f, h):
    for n in range(0, N-1): #N - 1 pq?
        def G(u):
            return u - (h/2) * f(u) - (un1[:, n]) - (h/2) * f(un1[:, n])
        un1[:,n+1], iter  = newtonsys(G, jacobiana, un1[:,n])
    return un1





un1 = taylor_q2_trapezio(un1, f, h)

################## PLOTAGEM ###############################
fig, ax = plt.subplots()
plt.axhline(0, color='black',linewidth=1) # Eixo X horizontal
plt.axvline(0, color='black',linewidth=1) # Eixo Y vertical
plt.plot(t, un1[0], color='red', linewidth=1, label=r'Predador')
plt.plot(t, un1[1], color='blue',linewidth=1, label=r'Presa')
plt.legend()
plt.show()
############################################################
