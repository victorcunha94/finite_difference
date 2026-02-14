import numpy as np
import matplotlib.pyplot as plt


xi = []

def f(x):
    return 2*np.sin(x) - x


def dfdx(x):
    return 2*np.cos(x) - 1

x0 = 2.2

for i in range(100):
    x = x0 - f(x0) / dfdx(x0)
    x0 = x
    xi.append(x)

y = f(xi)
plt.scatter(xi, y)
plt.show()
print(x)
