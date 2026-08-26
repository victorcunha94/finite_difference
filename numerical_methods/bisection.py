import numpy as np
import matplotlib.pyplot as plt
"""
O objetivo do método é reduzir a amplitude do intervalo que contém a raiz até se atingir a precisão
requerida: (b - a) < eps, usando para isso a sucessiva divisão de [a,b] ao meio. 
"""

tol = 0.001
a, b = 0,1
xi = []
xmin, xmax, ymin, ymax = -5.0, 5, -10, 15
domain = np.linspace(xmin, xmax, 50)

def f(x):
    #2*np.sin(x) - x
    return x**3 - 9*x + 3

def h(x):
    return 1/x

x = (a + b) / 2
xi.append(x)

iteration = 0
interval = [a, b]
eps = abs(b - a)
while abs(eps) > tol:
    M = f(a)
    x = (a + b) / 2
    xi.append(x)
    x_bar = a
    if M * f(x) > 0:
        a = x
    else:
        b = x
    eps = abs(b - a)
    iteration += 1

x_raiz = (a + b) / 2.0
xi.append(x)
print(iteration)
xi = np.array(xi)
yi = f(xi)
y = f(domain)
################## PLOTAGEM ###############################
fig, ax = plt.subplots()
plt.axis([xmin, xmax, ymin, ymax])
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.plot(domain, f(domain), label=r'$f(x) = x \ln(x) - 1$')
plt.scatter(xi, f(xi), color='red', zorder=5, label=f'Raiz ≈ {x_raiz:.4f}')
plt.grid(True)
plt.legend()
plt.show()