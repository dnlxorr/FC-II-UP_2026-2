"""
Metodo de biseccion aplicado a un problema fisico sencillo: encontrar el
coeficiente de arrastre c_d de un paracaidista en caida libre con
resistencia del aire, tal que su velocidad en t=4s sea 36 m/s.

Velocidad analitica en caida libre con arrastre cuadratico:
    v(t) = sqrt(g*m/c_d) * tanh( sqrt(g*c_d/m) * t )

Se busca la raiz de:
    f(c_d) = v(t; c_d) - v_objetivo = 0
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.8        # gravedad [m/s^2]
m = 68.1       # masa del paracaidista [kg]
t = 4.0        # tiempo [s]
v_obj = 36.0   # velocidad objetivo [m/s]


def v(cd):
    """Velocidad analitica en caida libre con arrastre cuadratico."""
    return np.sqrt(g * m / cd) * np.tanh(np.sqrt(g * cd / m) * t)


def f(cd):
    return v(cd) - v_obj


def biseccion(f, a, b, tol=1e-6, max_iter=100):
    """
    Metodo de biseccion. f debe cambiar de signo entre a y b.
    Devuelve la raiz aproximada y el historial de anchos de intervalo
    (para graficar la convergencia).
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")

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

    raiz = (a + b) / 2
    return raiz, anchos, it + 1


# ---------------------------------------------------------
# Resolver
# ---------------------------------------------------------
a, b = 0.1, 0.2   # intervalo donde f cambia de signo (ver verificacion)
cd_raiz, anchos, n_iter = biseccion(f, a, b, tol=1e-8)

print(f"Coeficiente de arrastre encontrado: c_d = {cd_raiz:.6f} kg/m")
print(f"Convergencia en {n_iter} iteraciones")
print(f"Verificacion: v({t}s) = {v(cd_raiz):.4f} m/s  (objetivo: {v_obj} m/s)")

# ---------------------------------------------------------
# Grafica de convergencia: ancho del intervalo vs iteracion
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(anchos, marker='o', markersize=3)
plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Ancho del intervalo |b - a| (escala log)")
plt.title("Convergencia del método de bisección")
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("biseccion_convergencia.png", dpi=150)
plt.show()