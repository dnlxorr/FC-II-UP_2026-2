"""
FISICA COMPUTACIONAL
Inversion de matriz (Gauss-Jordan con pivoteo parcial)
Aplicacion: Sistema de masas conectadas por resortes (matriz de rigidez
            y su inversa: la matriz de flexibilidad)

Problema fisico:
-----------------
4 bloques conectados en serie por 5 resortes entre dos paredes fijas:

   pared -k1- m1 -k2- m2 -k3- m3 -k4- m4 -k5- pared

En equilibrio estatico, la fuerza neta sobre cada bloque es cero
(Ley de Hooke generalizada), lo que da el sistema:

    K x = F

donde K es la matriz de rigidez (tridiagonal, simetrica por la
Ley de accion y reaccion en los resortes) y x es el vector de
desplazamientos.

Al calcular K^-1 (la "matriz de flexibilidad"), el elemento (K^-1)_ij
tiene significado fisico directo: es el desplazamiento del bloque i
cuando se aplica una fuerza UNITARIA solo en el bloque j. Por el
teorema de reciprocidad de Maxwell-Betti, K^-1 debe ser simetrica.
"""

import numpy as np


# ------------------------------------------------------------------
# Construccion de la matriz de rigidez a partir de las constantes
# de los resortes
# ------------------------------------------------------------------
def construir_matriz_rigidez(k):
    """
    k: lista de constantes de resorte [k1, k2, ..., k_{n+1}]
       (n+1 resortes para n bloques)
    """
    n = len(k) - 1  # numero de bloques
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        K[i][i] = k[i] + k[i + 1]
        if i > 0:
            K[i][i - 1] = -k[i]
        if i < n - 1:
            K[i][i + 1] = -k[i + 1]
    return K


# ------------------------------------------------------------------
# Inversion de matriz por Gauss-Jordan con pivoteo parcial
# ------------------------------------------------------------------
def inversa_gauss_jordan(A, verbose=True):
    n = len(A)
    # Matriz aumentada [A | I]
    M = [[float(A[i][j]) for j in range(n)] +
         [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n):
        # --- buscar pivote (mayor valor absoluto en la columna k) ---
        columna = [abs(M[i][k]) for i in range(k, n)]
        fila_pivote = k + columna.index(max(columna))

        if abs(M[fila_pivote][k]) < 1e-12:
            raise ValueError("La matriz es singular; no tiene inversa.")

        if fila_pivote != k:
            M[k], M[fila_pivote] = M[fila_pivote], M[k]
            if verbose:
                print(f"Intercambio de fila {k} <-> fila {fila_pivote}")

        # --- normalizar la fila del pivote ---
        pivote = M[k][k]
        M[k] = [val / pivote for val in M[k]]

        # --- eliminar la columna k en TODAS las demas filas ---
        # (arriba y abajo del pivote: esto es lo que distingue a
        #  Gauss-Jordan de la eliminacion gaussiana comun)
        for i in range(n):
            if i != k:
                factor = M[i][k]
                M[i] = [M[i][j] - factor * M[k][j] for j in range(2 * n)]

    inversa = [fila[n:] for fila in M]
    return inversa


def multiplicar_matriz_vector(A, v):
    n = len(A)
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]


if __name__ == "__main__":

    # ---------------- Constantes de los resortes (N/m) ----------------
    k = [100.0, 150.0, 120.0, 180.0, 90.0]   # k1..k5

    print("=" * 65)
    print("Sistema masa-resorte: matriz de rigidez K y su inversa K^-1")
    print("=" * 65)
    print(f"Constantes de resorte: {k} N/m\n")

    K = construir_matriz_rigidez(k)
    print("Matriz de rigidez K:")
    for fila in K:
        print(["{:7.1f}".format(v) for v in fila])

    K_inv = inversa_gauss_jordan(K)

    print("\nMatriz de flexibilidad K^-1 (m/N):")
    for fila in K_inv:
        print(["{:.6f}".format(v) for v in fila])

    # ---------------- Verificaciones ----------------
    K_np = np.array(K)
    K_inv_np = np.array(K_inv)

    error_identidad = np.max(np.abs(K_np @ K_inv_np - np.eye(len(K))))
    print(f"\nVerificacion || K K^-1 - I ||max = {error_identidad:.2e}")

    error_vs_numpy = np.max(np.abs(K_inv_np - np.linalg.inv(K_np)))
    print(f"Diferencia respecto a numpy.linalg.inv: {error_vs_numpy:.2e}")

    simetria = np.max(np.abs(K_inv_np - K_inv_np.T))
    print(f"Asimetria de K^-1 (debe ser ~0 por Maxwell-Betti): {simetria:.2e}")

    print("\nReciprocidad de Maxwell-Betti (ejemplo bloque 1 <-> bloque 3):")
    print(f"  Desplaz. de m1 por fuerza unitaria en m3: (K^-1)[0][2] = {K_inv[0][2]:.6f} m")
    print(f"  Desplaz. de m3 por fuerza unitaria en m1: (K^-1)[2][0] = {K_inv[2][0]:.6f} m")

    # ---------------- Escenarios de fuerzas aplicadas ----------------
    escenarios = {
        "A) Empujon solo en el bloque 3": [0.0, 0.0, 50.0, 0.0],
        "B) Fuerzas en los extremos (m1 y m4)": [30.0, 0.0, 0.0, 40.0],
        "C) Carga uniforme en los 4 bloques": [10.0, 10.0, 10.0, 10.0],
    }

    print("\n" + "=" * 65)
    print("Desplazamientos para distintos escenarios de fuerza (x = K^-1 F)")
    print("=" * 65)

    for nombre, F in escenarios.items():
        x = multiplicar_matriz_vector(K_inv, F)
        x_ref = np.linalg.solve(K_np, np.array(F))  # verificacion directa
        print(f"\n{nombre}")
        print(f"  F = {F} N")
        print("  x =", ["{:.5f}".format(v) for v in x], "m")
        print(f"  Error vs. resolver K x = F directamente: "
              f"{np.max(np.abs(np.array(x) - x_ref)):.2e}")