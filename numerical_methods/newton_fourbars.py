import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from numerical_methods.newton import newton_raphson

# Comprimentos dos elos
a = 1.0   # elo 2 (entrada)
b = 2.5   # elo 3 (acoplador)
c = 2.0   # elo 4 (saída)
d = 3.0   # elo 1 (terra)

def F(theta3_theta4, theta2):
    theta3, theta4 = theta3_theta4
    return np.array([
        a*np.cos(theta2) + b*np.cos(theta3) - c*np.cos(theta4) - d,
        a*np.sin(theta2) + b*np.sin(theta3) - c*np.sin(theta4)
    ])

def jacobian(theta3_theta4):
    theta3, theta4 = theta3_theta4
    return np.array([
        [-b*np.sin(theta3),  c*np.sin(theta4)],
        [ b*np.cos(theta3), -c*np.cos(theta4)]
    ])

def newton(theta2, x0, tol=1e-12, maxit=30):
    x = np.array(x0, dtype=float)
    for _ in range(maxit):
        fx = F(x, theta2)
        if np.linalg.norm(fx, np.inf) < tol:
            return x
        J = jacobian(x)
        dx = np.linalg.solve(J, -fx)
        x = x + dx
    raise RuntimeError("Newton não convergiu.")

# Ângulos de entrada para uma volta completa
theta2_vals = np.linspace(0, 2*np.pi, 181)

# Chute inicial para a configuração aberta.
# Para theta2 = 0, este chute converge para uma das configurações do mecanismo.
x_guess = np.deg2rad([35.0, 110.0])

theta3_vals = []
theta4_vals = []

for theta2 in theta2_vals:
    sol = newton(theta2, x_guess)
    theta3_vals.append(sol[0])
    theta4_vals.append(sol[1])
    # continuação: a solução anterior é o chute da próxima posição
    x_guess = sol

theta3_vals = np.array(theta3_vals)
theta4_vals = np.array(theta4_vals)

# Coordenadas das articulações
O2 = np.array([0.0, 0.0])
O4 = np.array([d, 0.0])

A_pts = np.column_stack([
    a*np.cos(theta2_vals),
    a*np.sin(theta2_vals)
])

B_pts = np.column_stack([
    a*np.cos(theta2_vals) + b*np.cos(theta3_vals),
    a*np.sin(theta2_vals) + b*np.sin(theta3_vals)
])

# Criar animação
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_aspect("equal")
ax.set_xlim(-1.5, 4.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Mecanismo de quatro barras — solução por Newton-Raphson")
ax.grid(True)

(line,) = ax.plot([], [], "o-", lw=2)
(path,) = ax.plot([], [], "--", lw=1)
angle_text = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top")

def init():
    line.set_data([], [])
    path.set_data([], [])
    angle_text.set_text("")
    return line, path, angle_text

def update(i):
    A = A_pts[i]
    B = B_pts[i]

    # O2 -> A -> B -> O4
    xs = [O2[0], A[0], B[0], O4[0]]
    ys = [O2[1], A[1], B[1], O4[1]]

    line.set_data(xs, ys)
    path.set_data(B_pts[:i+1, 0], B_pts[:i+1, 1])

    angle_text.set_text(
        rf"$\theta_2={np.rad2deg(theta2_vals[i]):.1f}^\circ$"
    )
    return line, path, angle_text

anim = FuncAnimation(
    fig,
    update,
    frames=len(theta2_vals),
    init_func=init,
    interval=40,
    blit=True
)

gif_path = "/home/victorcunha/Documentos/repository/finite_difference/numerical_methods/fourbars.gif"
anim.save(gif_path, writer=PillowWriter(fps=25))
plt.close(fig)


