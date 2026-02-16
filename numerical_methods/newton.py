import numpy as np
import matplotlib.pyplot as plt

tol = 0.001
xi = []
xmin, xmax, ymin, ymax = -2.5, 2.5, -2, 2
domain = np.linspace(xmin, xmax, 100)


def f(x):
    return 2*np.sin(x) - x


def dfdx(x):
    return 2*np.cos(x) - 1


tangente = lambda x: dfdx(x) * x + f(0) 
    

x0 = np.float64(1.8)
xi.append(x0)
eps = x0
iteration = 0

while tol <= eps: 
    x = x0 - f(x0) / dfdx(x0)
    eps = abs(x0 - x)
    print(eps)
    xi.append(x)
    x0 = x
    iteration += 1
    
y = f(domain)
yi = f(xi)
print(iteration)
print(xi)


################## PLOTAGEM ###############################
fig, ax = plt.subplots()
plt.axis([xmin, xmax, ymin, ymax])
plt.axhline(0, color='black',linewidth=1) # Eixo X horizontal
plt.axvline(0, color='black',linewidth=1) # Eixo Y vertical
plt.plot(domain, y)
plt.plot(xi, tangente(xi))
plt.scatter(xi, yi)
plt.show()
############################################################

