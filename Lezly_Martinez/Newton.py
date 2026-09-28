"""
Metodo de Newton-Raphson aplicado a un problema fisico sencillo: el mismo
problema del paracaidista resuelto antes con biseccion. Se busca el
coeficiente de arrastre c_d tal que la velocidad en t=4s sea 36 m/s.

v(t) = sqrt(g*m/c_d) * tanh( sqrt(g*c_d/m) * t )
f(c_d) = v(c_d) - v_objetivo

Formula de Newton:
    c_d_(n+1) = c_d_(n) - f(c_d_(n)) / f'(c_d_(n))

con la derivada (obtenida por regla de la cadena y simplificada):

    f'(c_d) = (1/(2*c_d)) * [ g*t / cosh^2(sqrt(g*c_d/m)*t) - v(c_d) ]
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


def newton_raphson(f, fprime, x0, tol=1e-10, max_iter=50):
    x = x0
    errores = []
    for it in range(max_iter):
        fx = f(x)
        x_nuevo = x - fx / fprime(x)
        error = abs(x_nuevo - x)
        errores.append(error)
        x = x_nuevo
        if error < tol:
            break
    return x, errores, it + 1


# ---------------------------------------------------------
# Resolver y comparar contra biseccion
# ---------------------------------------------------------
cd_newton, errores_newton, n_iter_newton = newton_raphson(f, fprime, x0=0.1)
print(f"Newton-Raphson: c_d = {cd_newton:.8f}  en {n_iter_newton} iteraciones")
print(f"Verificacion: v(4s) = {v(cd_newton):.6f} m/s (objetivo 36 m/s)")


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


cd_bis, anchos_bis, n_iter_bis = biseccion(f, 0.1, 0.2)
print(f"Bisección:      c_d = {cd_bis:.8f}  en {n_iter_bis} iteraciones")

# ---------------------------------------------------------
# Grafica comparativa de convergencia
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(errores_newton, 'o-', label=f"Newton-Raphson ({n_iter_newton} iter)")
plt.plot(anchos_bis, 's-', label=f"Bisección ({n_iter_bis} iter)")
plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Error / ancho de intervalo (escala log)")
plt.title("Newton-Raphson vs. bisección — mismo problema físico")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("newton_vs_biseccion.png", dpi=150)
plt.show()