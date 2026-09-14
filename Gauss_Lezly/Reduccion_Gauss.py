"""
FISICA COMPUTACIONAL
Eliminacion Gaussiana con Pivoteo Parcial
Aplicacion: Sistema de N bloques conectados por cuerdas (2da Ley de Newton)

Problema fisico:
-----------------
N bloques de masas m1, m2, ..., mN estan sobre una superficie horizontal
SIN FRICCION, conectados en fila por cuerdas ideales (inextensibles,
sin masa). Se aplica una fuerza externa F al ultimo bloque, arrastrando
a todo el sistema con una aceleracion comun 'a'.

Incognitas: x = [a, T1, T2, ..., T_{N-1}]
  a     -> aceleracion comun de todos los bloques
  T_i   -> tension en la cuerda entre el bloque i y el bloque i+1

A partir de la 2da Ley de Newton (F_neta = m*a) aplicada a cada bloque:

  Bloque 1        :  m1*a - T1                 = 0
  Bloque i (medio) :  mi*a + T_{i-1} - T_i      = 0      (1 < i < N)
  Bloque N        :  mN*a + T_{N-1}             = F

Esto arma un sistema lineal A x = b que se resuelve con eliminacion
gaussiana con pivoteo parcial.
"""

import numpy as np


def construir_sistema_bloques(masas, F):
    """
    Construye la matriz A y el vector b del sistema de bloques conectados
    a partir de la lista de masas y la fuerza aplicada F.

    Orden de incognitas: x = [a, T1, T2, ..., T_{N-1}]
    """
    N = len(masas)
    n_incognitas = N  # 1 aceleracion + (N-1) tensiones
    A = [[0.0] * n_incognitas for _ in range(N)]
    b = [0.0] * N

    for i in range(N):  # i = indice del bloque (0-based), bloque real = i+1
        A[i][0] = masas[i]  # coeficiente de 'a' en la ecuacion del bloque i+1

        if i > 0:
            A[i][i] = 1.0       # +T_{i} (tension que lo jala desde atras)
        if i < N - 1:
            A[i][i + 1] = -1.0  # -T_{i+1} (tension que lo frena adelante)

        if i == N - 1:
            b[i] = F  # solo el ultimo bloque recibe la fuerza externa

    return A, b


def eliminacion_gaussiana_pivoteo(A, b, verbose=True):
    """
    Resuelve A x = b mediante eliminacion gaussiana con pivoteo parcial.
    """
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    intercambios = 0

    for k in range(n - 1):
        col_k = [abs(M[i][k]) for i in range(k, n)]
        fila_pivote = k + col_k.index(max(col_k))

        if abs(M[fila_pivote][k]) < 1e-12:
            raise ValueError("Sistema singular: no tiene solucion unica.")

        if fila_pivote != k:
            M[k], M[fila_pivote] = M[fila_pivote], M[k]
            intercambios += 1
            if verbose:
                print(f"Se usa como pivote la ecuacion del bloque con mayor "
                      f"masa disponible -> intercambio fila {k} <-> fila {fila_pivote}")

        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - suma) / M[i][i]

    if verbose:
        print(f"\nNumero total de intercambios de fila: {intercambios}")

    return x


if __name__ == "__main__":

    # ---------------- Datos fisicos del problema ----------------
    masas = [2.0, 0.5, 6.0, 1.0, 3.0]   # kg, bloque 1 a bloque 5
    F = 100.0                            # N, fuerza aplicada al ultimo bloque

    print("=" * 60)
    print("Sistema de bloques conectados por cuerdas (2da Ley de Newton)")
    print("=" * 60)
    print(f"Masas (kg): {masas}")
    print(f"Fuerza aplicada F = {F} N\n")

    A, b = construir_sistema_bloques(masas, F)

    print("Matriz A generada a partir del modelo fisico:")
    for fila in A:
        print(["{:6.2f}".format(v) for v in fila])
    print(f"Vector b: {b}\n")

    x = eliminacion_gaussiana_pivoteo(A, b, verbose=True)

    a = x[0]
    tensiones = x[1:]

    print(f"\nAceleracion comun del sistema: a = {a:.4f} m/s^2")
    for idx, T in enumerate(tensiones, start=1):
        print(f"  T{idx} = {T:.4f} N")

    # ---------------- Verificacion fisica ----------------
    # Sumando TODAS las ecuaciones de Newton, las tensiones se cancelan
    # (accion-reaccion) y queda: F = (m1+m2+...+mN) * a
    a_teorica = F / sum(masas)
    print(f"\nVerificacion fisica -> a = F / M_total = {a_teorica:.4f} m/s^2")
    print(f"Diferencia con el valor calculado: {abs(a - a_teorica):.2e}")

    # ---------------- Verificacion numerica con numpy ----------------
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    x_ref = np.linalg.solve(A_np, b_np)
    error = np.max(np.abs(np.array(x) - x_ref))
    print(f"Error maximo respecto a numpy.linalg.solve: {error:.2e}")