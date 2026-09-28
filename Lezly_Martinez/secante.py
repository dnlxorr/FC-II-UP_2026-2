"""
Metodo de la secante aplicado al mismo problema fisico del paracaidista:
encontrar el coeficiente de arrastre c_d tal que la velocidad en t=4s
sea 36 m/s.

v(t) = sqrt(g*m/c_d) * tanh( sqrt(g*c_d/m) * t )
f(c_d) = v(c_d) - v_objetivo

Formula de la secante (deducida de Newton, reemplazando f' por una
diferencia finita con los dos ultimos puntos):

    c_(n+1) = c_n - f(c_n) * (c_n - c_(n-1)) / (f(c_n) - f(c_(n-1)))
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.8
m = 68.1
t = 4.0
v_obj = 36.0


def v(cd):
    return np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t)


def f(cd):
    return v(cd) - v_obj


def fprime(cd):
    return (1 / (2 * cd)) * (g * t / np.cosh(np.sqrt(g * cd / m) * t)**2 - v(cd))


def secante(f, x0, x1, tol=1e-10, max_iter=50):
    errores = []
    for it in range(max_iter):
        f0, f1 = f(x0), f(x1)
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        errores.append(error)
        x0, x1 = x1, x2
        if error < tol:
            break
    return x1, errores, it + 1


def newton_raphson(f, fprime, x0, tol=1e-10, max_iter=50):
    x = x0
    errores = []
    for it in range(max_iter):
        x_nuevo = x - f(x) / fprime(x)
        error = abs(x_nuevo - x)
        errores.append(error)
        x = x_nuevo
        if error < tol:
            break
    return x, errores, it + 1


def biseccion(f, a, b, tol=1e-10, max_iter=100):
    anchos = []
    for it in range(max_iter):
        c = (a + b) / 2
        anchos.append(b - a)
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        if (b - a) < tol:
            break
    return (a + b) / 2, anchos, it + 1


# ---------------------------------------------------------
# Resolver con los tres metodos
# ---------------------------------------------------------
cd_sec, err_sec, n_sec = secante(f, 0.1, 0.2)
cd_new, err_new, n_new = newton_raphson(f, fprime, 0.1)
cd_bis, err_bis, n_bis = biseccion(f, 0.1, 0.2)

print(f"Secante:        c_d = {cd_sec:.8f}  en {n_sec} iteraciones")
print(f"Newton-Raphson: c_d = {cd_new:.8f}  en {n_new} iteraciones")
print(f"Bisección:      c_d = {cd_bis:.8f}  en {n_bis} iteraciones")

# ---------------------------------------------------------
# Grafica comparativa
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(err_sec, '^-', label=f"Secante ({n_sec} iter)")
plt.plot(err_new, 'o-', label=f"Newton-Raphson ({n_new} iter)")
plt.plot(err_bis, 's-', label=f"Bisección ({n_bis} iter)")
plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Error (escala log)")
plt.title("Secante vs. Newton-Raphson vs. bisección — mismo problema físico")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("secante_comparacion.png", dpi=150)
plt.show()