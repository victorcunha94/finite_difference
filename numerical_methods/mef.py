import numpy as np
from interpolate import*
import matplotlib.pyplot as plt 


omega = [0, 1]
n_elements = 3


def elementos(omega, n_elements):
    partition = linspace(omega[0], omega[1], n_elements)
    #for i in range(n_elements):
    return partition

     

def phi(interpol, elementos):
    return 
    

X = [0, 0.5]
x = np.linspace(0,1,100)
Y = [1, 0]


plt.plot(x, lagrange(x, X, Y))
plt.scatter(X,Y)
plt.show()


