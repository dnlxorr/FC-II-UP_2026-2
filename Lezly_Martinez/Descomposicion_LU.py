"""
FISICA COMPUTACIONAL
Descomposicion LU (con pivoteo parcial: PA = LU)
Aplicacion: Conduccion de calor 1D en estado estacionario en una barra,
            con distintos escenarios de generacion interna de calor.

Problema fisico:
-----------------
Barra de longitud L, conductividad termica k, con temperaturas fijas
T_izq y T_der en sus extremos (condiciones de Dirichlet), y una fuente
de calor interna q(x) [W/m^3] (por ejemplo, calentamiento por efecto Joule).

Ecuacion de Fourier en estado estacionario:
    k * d2T/dx2 + q(x) = 0

Discretizada por diferencias finitas centradas en n nodos internos:
    T_{i-1} - 2 T_i + T_{i+1} = -q_i * dx^2 / k

Esto da un sistema lineal tridiagonal A*T = b, donde A depende solo de
la geometria (no de la fuente de calor). Factorizamos A una sola vez
con LU y reutilizamos la factorizacion para varios escenarios de q(x).
"""

import numpy as np


# ------------------------------------------------------------------
# Construccion del modelo fisico
# ------------------------------------------------------------------
def construir_matriz_conduccion(n):
    """Matriz tridiagonal de la ecuacion de conduccion (segunda derivada)."""
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = -2.0
        if i > 0:
            A[i][i - 1] = 1.0
        if i < n - 1:
            A[i][i + 1] = 1.0
    return A


def construir_b(q, dx, k, T_izq, T_der):
    """Vector b para un perfil de generacion de calor q(x) dado."""
    n = len(q)
    b = [-(q[i] * dx ** 2) / k for i in range(n)]
    b[0] -= T_izq
    b[-1] -= T_der
    return b


# ------------------------------------------------------------------
# Descomposicion LU con pivoteo parcial:  P A = L U
# ------------------------------------------------------------------
def descomposicion_LU(A):
    n = len(A)
    U = [[float(A[i][j]) for j in range(n)] for i in range(n)]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n - 1):
        # --- buscar pivote (mayor valor absoluto en la columna k) ---
        columna = [abs(U[i][k]) for i in range(k, n)]
        fila_pivote = k + columna.index(max(columna))

        if fila_pivote != k:
            U[k], U[fila_pivote] = U[fila_pivote], U[k]
            P[k], P[fila_pivote] = P[fila_pivote], P[k]
            # intercambiar tambien la parte ya calculada de L
            for j in range(k):
                L[k][j], L[fila_pivote][j] = L[fila_pivote][j], L[k][j]

        # --- eliminacion, guardando los multiplicadores en L ---
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return P, L, U


def resolver_con_LU(P, L, U, b):
    """Resuelve A x = b reutilizando una factorizacion PA=LU ya calculada."""
    n = len(b)

    # Pb = P * b
    Pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]

    # Sustitucion hacia adelante: L y = Pb
    y = [0.0] * n
    for i in range(n):
        y[i] = Pb[i] - sum(L[i][j] * y[j] for j in range(i))

    # Sustitucion regresiva: U x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


if __name__ == "__main__":

    # ---------------- Parametros fisicos de la barra ----------------
    L_barra = 1.0      # m
    n = 6               # nodos internos
    dx = L_barra / (n + 1)
    k = 50.0            # W/(m*K), conductividad termica (tipo acero)
    T_izq = 100.0        # C
    T_der = 25.0         # C

    print("=" * 65)
    print("Conduccion de calor 1D en una barra - Descomposicion LU")
    print("=" * 65)
    print(f"L = {L_barra} m, n = {n} nodos internos, dx = {dx:.4f} m, k = {k} W/(m*K)")
    print(f"T_izq = {T_izq} C, T_der = {T_der} C\n")

    # La matriz A depende solo de la geometria -> se factoriza UNA vez
    A = construir_matriz_conduccion(n)
    P, L, U = descomposicion_LU(A)

    print("Matriz A (conduccion, tridiagonal):")
    for fila in A:
        print(["{:5.1f}".format(v) for v in fila])

    # ---------------- Verificacion de la factorizacion ----------------
    A_np, P_np, L_np, U_np = map(lambda M: np.array(M), (A, P, L, U))
    error_LU = np.max(np.abs(P_np @ A_np - L_np @ U_np))
    print(f"\nVerificacion || P A - L U || (debe ser ~0): {error_LU:.2e}\n")

    # ---------------- Escenarios de generacion de calor ----------------
    escenarios = {
        "1) Sin generacion interna (conduccion pura)": [0.0] * n,
        "2) Generacion uniforme (calentamiento resistivo)": [2.0e5] * n,
        "3) Fuente localizada en el centro (punto caliente)":
            [1.0e6 if i == n // 2 else 0.0 for i in range(n)],
    }

    for nombre, q in escenarios.items():
        b = construir_b(q, dx, k, T_izq, T_der)
        T_interna = resolver_con_LU(P, L, U, b)   # solo sustitucion, no refactorizar
        perfil = [T_izq] + T_interna + [T_der]     # perfil completo incl. fronteras

        print(f"\n{nombre}")
        print("  q(x) =", q, "W/m^3")
        print("  Perfil de temperatura (C):",
              ["{:.2f}".format(T) for T in perfil])

    # ---------------- Verificacion analitica del caso sin fuente ----------------
    # Sin generacion de calor, la solucion exacta es una linea recta:
    #   T(x) = T_izq + (T_der - T_izq) * x / L
    q0 = [0.0] * n
    b0 = construir_b(q0, dx, k, T_izq, T_der)
    T0 = resolver_con_LU(P, L, U, b0)
    x_nodos = [(i + 1) * dx for i in range(n)]
    T0_analitica = [T_izq + (T_der - T_izq) * x / L_barra for x in x_nodos]

    error_analitico = max(abs(a - b) for a, b in zip(T0, T0_analitica))
    print(f"\nVerificacion analitica (caso sin fuente vs. recta teorica):")
    print(f"  Error maximo: {error_analitico:.2e} C")