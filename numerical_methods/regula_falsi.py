import numpy as np
import matplotlib.pyplot as plt
"""
O objetivo do método é reduzir a amplitude do intervalo que contém a raiz até se atingir a precisão
requerida: (b - a) < eps, usando para isso a sucessiva divisão de [a,b] ao meio. 
"""

tol = 0.001
tol2 = 0.005
a, b = 0,1
xi = []
xmin, xmax, ymin, ymax = -5.0, 5, -10, 15
domain = np.linspace(xmin, xmax, 50)

def f(x):
    #2*np.sin(x) - x
    return x**3 - 9*x + 3

def h(x):
    return 1/x

iteration = 0
interval = [a, b]
x = (a * abs(f(b)) + b * abs(f(a))) / (abs(f(b)) + abs(f(a)))
xi.append(x)

eps = abs(b - a)
while min(abs(f(x)), (b -a)) > tol:
    if f(a)*f(b) > tol:
        a = x
    else:
        b = x
    x = (a * abs(f(b)) + b*abs(f(a))) / (abs(f(b)) + abs(f(a)))
    xi.append(x)
    eps = abs(b - a)
    iteration += 1

x_raiz = x
print(iteration)
y = f(domain)
xi = np.array(xi)
yi = f(xi)

################## PLOTAGEM ###############################
fig, ax = plt.subplots()
plt.axis([xmin, xmax, ymin, ymax])
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.plot(domain, f(domain), label=r'$f(x) = x^{3} - 9x + 3$')
#plt.scatter(x_raiz, f(x_raiz), color='red', zorder=5, label=f'Raiz ≈ {x_raiz:.4f}')
plt.scatter(xi, yi, color='red', zorder=5, label=f'Raiz ≈ {x_raiz:.4f}')
plt.grid(True)
plt.legend()
plt.show()