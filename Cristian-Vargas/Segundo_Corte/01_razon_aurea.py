"""
01 - Búsqueda de la razón áurea
================================
Ejemplo físico: un bloque unido a un resorte (k = 2 N/m) es jalado además
por una fuerza constante F = 4 N. Su energía potencial es

        U(x) = (1/2) k x^2 - F x

El equilibrio está en el mínimo de U. Analíticamente: dU/dx = k x - F = 0
  =>  x* = F / k = 2 m.   Aquí lo buscamos SIN derivadas, solo evaluando U.

Idea del método: en [a, b] se colocan dos puntos interiores x1 < x2 y se
compara U(x1) con U(x2). Si U(x1) < U(x2) el mínimo está en [a, x2]; si no,
en [x1, b]. Con la razón áurea (1/phi = 0.618...) uno de los dos puntos se
REUTILIZA, así que cada iteración cuesta una sola evaluación nueva de U.

Ejecutar:  python 01_razon_aurea.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Datos del problema ----------------
k = 2.0   # constante del resorte [N/m]
F = 4.0   # fuerza constante [N]


def U(x):
    """Energía potencial [J] en función de la posición x [m]."""
    return 0.5 * k * x**2 - F * x


# ---------------- Algoritmo ----------------
def razon_aurea(f, a, b, tol=1e-3, verbose=True):
    """Devuelve (x_min, n_iteraciones, n_evaluaciones_de_f)."""
    inv_phi = (np.sqrt(5) - 1) / 2          # 1/phi = 0.618...

    x1 = b - (b - a) * inv_phi
    x2 = a + (b - a) * inv_phi
    f1, f2 = f(x1), f(x2)
    evaluaciones = 2
    n = 0

    if verbose:
        print(f"{'n':>2} {'a':>8} {'b':>8} {'x1':>8} {'x2':>8} {'f(x1)':>9} {'f(x2)':>9}")

    while (b - a) > tol:
        n += 1
        if verbose:
            print(f"{n:>2} {a:8.4f} {b:8.4f} {x1:8.4f} {x2:8.4f} {f1:9.4f} {f2:9.4f}")

        if f1 < f2:
            # El mínimo está en [a, x2]
            b = x2
            x2, f2 = x1, f1                 # se REUTILIZA el punto x1
            x1 = b - (b - a) * inv_phi      # único punto nuevo
            f1 = f(x1)
        else:
            # El mínimo está en [x1, b]
            a = x1
            x1, f1 = x2, f2                 # se REUTILIZA el punto x2
            x2 = a + (b - a) * inv_phi      # único punto nuevo
            f2 = f(x2)
        evaluaciones += 1

    return (a + b) / 2, n, evaluaciones


# ---------------- Programa principal ----------------
if __name__ == "__main__":
    a, b, tol = 0.0, 5.0, 1e-3

    x_min, n, evals = razon_aurea(U, a, b, tol)
    x_exacto = F / k

    print("\nResultados")
    print(f"  x* (razón áurea) = {x_min:.5f} m")
    print(f"  x* (analítico)   = {x_exacto:.5f} m")
    print(f"  error            = {abs(x_min - x_exacto):.2e} m")
    print(f"  iteraciones      = {n}")
    print(f"  evaluaciones de U = {evals}  (1 por iteración + 2 iniciales)")

    # Número de iteraciones esperado: n >= ln(L0/tol) / ln(phi)
    phi = (1 + np.sqrt(5)) / 2
    print(f"  n teórico        >= {np.log((b - a) / tol) / np.log(phi):.2f}")

    # ---------------- Gráfica ----------------
    x = np.linspace(a, b, 300)
    plt.plot(x, U(x), label="U(x) = x² − 4x")
    plt.axvline(x_exacto, color="gray", linestyle="--", label="mínimo exacto")
    plt.plot(x_min, U(x_min), "ro", label="razón áurea")
    plt.xlabel("x (m)")
    plt.ylabel("U (J)")
    plt.title("Equilibrio de un resorte con fuerza constante")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
