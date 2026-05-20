import math
import numpy as np
from math import factorial
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5

taylor1 = lambda x: 6*x + 45

x = np.linspace(-3, 3, 20)
y = f(x)

t1 = taylor1(x)

def taylor_exp(x, n):
    e_to_2 = 0
    for i in range(n + 1):
        e_to_2 += x**i / factorial(i)
    return e_to_2


def taylor_cos(x, n):
    cos = 0
    for i in range(n + 1):
        coef = (-1)**i
        numerador = x**(2*i)
        denominador = factorial(2*i)
        cos += coef * (numerador/denominador)
    return cos


angles = np.arange(-2*np.pi, 2*np.pi, 0.1)
cos = np.cos(angles)

fig2, ax = plt.subplots()
ax.plot(angles, cos)

for k in range(1,6):
    t_cos = [taylor_cos(angle, k) for angle in angles]
    plt.plot(angles, t_cos, "-b")

print(t_cos)


ax.set_ylim([-5.0, 5.0])
ax.set_xlim([-7.0, 7.0])
plt.grid()
plt.show()

# ## Plotagem ##
fig1, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(0, 20)
plt.grid()
plt.plot(x, y, "o-b")
plt.plot(x, t1)
plt.show()
#
# fig2, ax = plt.subplots()
# ax.plot(angles, cos)
# plt.plot(angles, t_cos, "-b")
# ax.set_ylim([-5.0, 5.0])
# ax.set_xlim([-7.0, 7.0])
# plt.grid()
# plt.show()
