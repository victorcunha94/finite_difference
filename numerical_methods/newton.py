#autor: github.com/victorcunha94

import numpy as np
import matplotlib.pyplot as plt
"""
Este programa utiliza o método de Newton/Raphson para encontrar 
zero das funções de uma variável, quando a função e sua derivada
são conhecidas.
Ainda precisa ser feito: 
* Função tangente precisa plotar a reta cuja taxa de variação é 
  a derivada no ponto de teste.
* Criar a animação das retas tangentes se aproximando da raíz da função

"""
tol = 0.001
xi = []
xmin, xmax, ymin, ymax = -5.5, 12.5, -50, 500
domain = np.linspace(xmin, xmax, 100)


def f(x):
    #2*np.sin(x) - x
    return x**3 - 3*x - 5


def dfdx(x):
    return 3*(x**2) - 3


tangente = lambda x: dfdx(x) * x + f(0) 
    

x0 = 6.8
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
xi = np.array(xi)
yi = f(xi)
print(f' Convergio com {tol} de tolerância em {iteration} iterações.')
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

