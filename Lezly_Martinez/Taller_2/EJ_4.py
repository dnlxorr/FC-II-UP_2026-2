"""
problema4_vdw_newton.py
=======================
Problema 4 – Termodinámica real y sistemas no lineales multivariables.

Parte A (1D): gas de Van der Waals (CO2)
----------------------------------------
    f(v) = (P + a/v²)(v - b) - R T = 0
Se resuelve con
  * Newton:   v_{k+1} = v_k - f/f'            (convergencia cuadrática, p = 2)
  * Secante:  v_{k+1} = v_k - f (v_k - v_{k-1})/(f_k - f_{k-1})
                                               (superlineal, p ≈ 1.618, sin f')
con  f'(v) = P + a/v² - 2a(v - b)/v³ .

Parte B (2D): sistema  F(x,y) = 0
---------------------------------
    f1 = x³ - 3xy² - 1 = 0 ,   f2 = 3x²y - y³ = 0
Equivale a z³ = 1 con z = x + iy; raíces: (1,0) y (-1/2, ±√3/2).
  * Relajación (Gauss-Seidel):  x_{k+1} = g1(x_k, y_k) = ∛(1 + 3 x y²)
                                y_{k+1} = g2(x_{k+1}, y_k) = y³ / (3 x_{k+1}²)
  * Newton 2D:  J(x_k) Δ = -F(x_k),  x_{k+1} = x_k + Δ, resolviendo el sistema
    lineal con la eliminación de Gauss del Problema 1 (algebra_lineal.py).

Uso:  python problema4_vdw_newton.py
Salida: tabla de errores por consola y la figura p4_vdw.png
"""
import math
import matplotlib.pyplot as plt
from algebra_lineal import gauss_pivoteo

# ==================================================================
# PARTE A – Van der Waals
# ==================================================================
a_vdw = 0.3643      # [Pa·m⁶/mol²]
b_vdw = 4.267e-5    # [m³/mol]
P = 2e6             # presión [Pa]
T = 300.0           # temperatura [K]
R = 8.314           # constante de los gases [J/(mol·K)]


def f(v):
    """Función cuya raíz es el volumen molar: (P + a/v²)(v - b) - R T."""
    return (P + a_vdw / v**2) * (v - b_vdw) - R * T


def df(v):
    """Derivada analítica de f respecto a v."""
    return P + a_vdw / v**2 - 2 * a_vdw * (v - b_vdw) / v**3


def newton(v0, tol=1e-14, max_iter=50):
    """Método de Newton 1D. Retorna la lista de iterados [v0, v1, ...]."""
    v, historial = v0, [v0]
    for _ in range(max_iter):
        v_nuevo = v - f(v) / df(v)
        historial.append(v_nuevo)
        if abs(v_nuevo - v) < tol:
            break
        v = v_nuevo
    return historial


def secante(v0, v1, tol=1e-14, max_iter=50):
    """Método de la secante 1D (dos semillas, sin derivada). Retorna los iterados."""
    historial = [v0, v1]
    for _ in range(max_iter):
        f0, f1 = f(v0), f(v1)
        if f1 == f0:                       # evita división por cero
            break
        v2 = v1 - f1 * (v1 - v0) / (f1 - f0)
        historial.append(v2)
        if abs(v2 - v1) < tol:
            break
        v0, v1 = v1, v2
    return historial


def parte_A():
    """Ejecuta la parte A: calcula v, imprime la tabla de errores y grafica."""
    v_ideal = R * T / P                           # semilla: gas ideal
    h_newton = newton(v_ideal)
    h_secante = secante(v_ideal, 0.9 * v_ideal)
    v_ref = h_newton[-1]                          # valor de referencia (converged)
    print(f"v (Newton) = {v_ref:.10e} m³/mol")
    print("it   |error| Newton   |error| secante")
    for i in range(max(len(h_newton), len(h_secante))):
        e_n = abs(h_newton[i] - v_ref) if i < len(h_newton) else float("nan")
        e_s = abs(h_secante[i] - v_ref) if i < len(h_secante) else float("nan")
        print(f"{i:2d}   {e_n:.3e}        {e_s:.3e}")

    # Se suma 1e-30 solo para poder graficar errores nulos en escala log
    plt.semilogy([abs(v - v_ref) + 1e-30 for v in h_newton], "o-", label="Newton")
    plt.semilogy([abs(v - v_ref) + 1e-30 for v in h_secante], "s-", label="Secante")
    plt.xlabel("iteración")
    plt.ylabel("error absoluto")
    plt.title("Van der Waals (CO₂): Newton vs secante")
    plt.legend()
    plt.grid()
    plt.savefig("p4_vdw.png", dpi=150)
    plt.close()


# ==================================================================
# PARTE B – Sistema no lineal 2D
# ==================================================================
def F(x, y):
    """Vector de funciones [f1, f2]."""
    return [x**3 - 3 * x * y**2 - 1,
            3 * x**2 * y - y**3]


def jacobiano(x, y):
    """Matriz jacobiana J = [[∂f1/∂x, ∂f1/∂y], [∂f2/∂x, ∂f2/∂y]]."""
    return [[3 * x**2 - 3 * y**2, -6 * x * y],
            [6 * x * y,            3 * x**2 - 3 * y**2]]


def raiz_cubica_real(t):
    """Raíz cúbica real (con signo) de t."""
    return math.copysign(abs(t) ** (1 / 3), t)


def relajacion_2d(x, y, tol=1e-12, max_iter=500):
    """Relajación en dos variables con actualización tipo Gauss-Seidel.

        x_{k+1} = g1(x_k, y_k)     = ∛(1 + 3 x_k y_k²)      (de f1 = 0)
        y_{k+1} = g2(x_{k+1}, y_k) = y_k³ / (3 x_{k+1}²)    (de f2 = 0)

    Cerca de (1, 0) el jacobiano de g es nulo, por eso converge muy rápido.
    Esta elección de g solo alcanza la raíz (1, 0).
    Retorna la lista de iterados [(x0,y0), (x1,y1), ...].
    """
    historial = [(x, y)]
    for _ in range(max_iter):
        x_nuevo = raiz_cubica_real(1 + 3 * x * y**2)
        y_nuevo = y**3 / (3 * x_nuevo**2)          # usa x ya actualizado
        historial.append((x_nuevo, y_nuevo))
        if max(abs(x_nuevo - x), abs(y_nuevo - y)) < tol:
            break
        x, y = x_nuevo, y_nuevo
    return historial


def newton_2d(x, y, tol=1e-14, max_iter=50):
    """Newton multivariable: en cada paso resuelve J Δ = -F con Gauss y avanza x ← x + Δ.

    Retorna la lista de iterados [(x0,y0), (x1,y1), ...].
    """
    historial = [(x, y)]
    for _ in range(max_iter):
        rhs = [-v for v in F(x, y)]                       # -F(x_k)
        dx, dy = gauss_pivoteo(jacobiano(x, y), rhs)      # solución de J Δ = -F
        x, y = x + dx, y + dy
        historial.append((x, y))
        if max(abs(dx), abs(dy)) < tol:
            break
    return historial


def parte_B():
    """Ejecuta la parte B con relajación y con Newton desde varias semillas."""
    h = relajacion_2d(1.2, 0.3)
    print(f"\nRelajación 2D desde (1.2, 0.3): {h[-1]} en {len(h) - 1} iteraciones")
    # Cada semilla cae en la cuenca de atracción de una de las tres raíces de z³ = 1
    for semilla in [(1.2, 0.3), (-0.4, 0.7), (-0.4, -0.7)]:
        hh = newton_2d(*semilla)
        raiz = tuple(round(v, 10) for v in hh[-1])
        print(f"Newton 2D desde {semilla}: raíz = {raiz} en {len(hh) - 1} iteraciones")


if __name__ == "__main__":
    parte_A()
    parte_B()