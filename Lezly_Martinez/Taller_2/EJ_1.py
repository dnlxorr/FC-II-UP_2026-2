"""
problema1_barra.py
==================
Problema 1 – Perfil de temperatura estacionario en una barra 1D.

Física
------
Barra delgada de longitud L con conductividad k y fuente interna q(x):
        -d²T/dx² = q(x)/k ,   T(0) = T_A ,  T(L) = T_B .

Discretización
--------------
N nodos internos, Δx = L/(N+1), diferencias centrales:
        -T_{i-1} + 2 T_i - T_{i+1} = (Δx²/k) q(x_i) ,   i = 1..N .
Las temperaturas de frontera pasan al lado derecho (b_1 += T_A, b_N += T_B),
lo que produce el sistema tridiagonal  A T = b  con A = tridiag(-1, 2, -1).

Qué hace el script
------------------
1) Construye A y b (N = 6, q = 5000 W/m³).
2) Resuelve con Gauss + pivoteo parcial y compara con la solución analítica.
3) Factoriza A una sola vez (Thomas) y resuelve tres perfiles de fuente.
4) Calcula A⁻¹ resolviendo A x = e_j y la compara con la fórmula cerrada
   (función de Green discreta).

Uso:  python problema1_barra.py      (requiere algebra_lineal.py en la misma carpeta)
Salida: tablas por consola y la figura p1_perfiles.png
"""
import math
import matplotlib.pyplot as plt
from algebra_lineal import gauss_pivoteo, thomas_factorizar, thomas_resolver

# ------------------------------------------------------------------
# Parámetros físicos y de discretización
# ------------------------------------------------------------------
L = 1.0        # longitud de la barra [m]
k = 45.0       # conductividad térmica [W/(m·K)]
TA = 100.0     # temperatura en x = 0 [°C]
TB = 20.0      # temperatura en x = L [°C]
N = 6          # número de nodos internos
dx = L / (N + 1)                              # espaciamiento de la malla [m]
x = [(i + 1) * dx for i in range(N)]          # posiciones de los nodos internos

# Perfiles de fuente de calor [W/m³]
def q1(s): return 5000.0                           # fuente uniforme
def q2(s): return 10000.0 * s / L                  # fuente lineal
def q3(s): return 8000.0 * math.sin(math.pi * s / L)  # fuente senoidal


def construir_A(n):
    """Matriz tridiagonal n x n con 2 en la diagonal y -1 en las codiagonales."""
    return [[2.0 if i == j else -1.0 if abs(i - j) == 1 else 0.0
             for j in range(n)] for i in range(n)]


def vector_b(q):
    """Lado derecho b para una fuente q(x).

    b_i = (Δx²/k) q(x_i); en los extremos se suman las condiciones de frontera:
    b_1 += T_A  y  b_N += T_B.
    """
    b = [dx**2 / k * q(xi) for xi in x]
    b[0] += TA
    b[-1] += TB
    return b


def temperatura_exacta_q1():
    """Solución analítica para q constante:
        T(x) = T_A + (T_B - T_A) x/L + q/(2k) x (L - x).
    Es exacta en los nodos porque la diferencia central no comete error
    de truncamiento para polinomios de grado <= 3.
    """
    return [TA + (TB - TA) * s / L + 5000.0 / (2 * k) * s * (L - s) for s in x]


def main():
    # ---------- Actividad 1: sistema explícito ----------
    A = construir_A(N)
    print("Matriz A =")
    for fila in A:
        print(["%2d" % v for v in fila])
    print("b (q1) =", [round(v, 5) for v in vector_b(q1)])

    # ---------- Actividad 2: Gauss con pivoteo parcial ----------
    T_gauss = gauss_pivoteo(A, vector_b(q1))
    print("\nT (Gauss)  =", [round(v, 4) for v in T_gauss])
    print("T (exacta) =", [round(v, 4) for v in temperatura_exacta_q1()])

    # ---------- Actividad 3: Thomas (factorizar una sola vez) ----------
    a = [-1.0] * (N - 1)    # subdiagonal
    d = [2.0] * N           # diagonal
    c = [-1.0] * (N - 1)    # superdiagonal
    l, u = thomas_factorizar(a, d, c)
    print("\nu_i =", [round(v, 4) for v in u], " (teoría: (i+1)/i)")

    xs = [0.0] + x + [L]    # se añaden los extremos para graficar
    for nombre, q in (("q1 = 5000", q1),
                      ("q2 = 10000 x/L", q2),
                      ("q3 = 8000 sin(πx/L)", q3)):
        T = thomas_resolver(l, u, c, vector_b(q))   # sin volver a factorizar
        plt.plot(xs, [TA] + T + [TB], "o-", label=nombre)
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title("Perfil de temperatura para tres fuentes de calor")
    plt.legend()
    plt.grid()
    plt.savefig("p1_perfiles.png", dpi=150)
    plt.close()

    # ---------- Actividad 4: inversa por columnas ----------
    # La columna j de A⁻¹ es la solución de A x = e_j (e_j = columna j de la identidad).
    inv = [[0.0] * N for _ in range(N)]
    for j in range(N):
        e_j = [1.0 if i == j else 0.0 for i in range(N)]
        col = thomas_resolver(l, u, c, e_j)
        for i in range(N):
            inv[i][j] = col[i]
    print("\nA^-1 =")
    for fila in inv:
        print([round(v, 4) for v in fila])

    # Verificación con la fórmula cerrada de la función de Green discreta:
    #   G_ij = min(i,j) (N+1 - max(i,j)) / (N+1)
    err = max(abs(inv[i][j] - min(i + 1, j + 1) * (N + 1 - max(i + 1, j + 1)) / (N + 1))
              for i in range(N) for j in range(N))
    print("Error máximo vs fórmula cerrada:", err)


if __name__ == "__main__":
    main()