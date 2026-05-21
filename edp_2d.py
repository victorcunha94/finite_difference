import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

def func(X, Y):
    u = X**2 + Y**2 - X*Y
    return u

u = func(X, Y)
fig, ax = plt.subplots()
im = ax.imshow(u)
ax.set_title('2d')

fig.colorbar(im, ax=ax, label='Interactive colorbar')

plt.show()