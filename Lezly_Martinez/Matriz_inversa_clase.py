"""
Solucion del sistema de la imagen mediante el METODO DE LA MATRIZ INVERSA:

     I1 + 4 I2 + 2 I3 = 13
    5 I1 + 2 I2 +   I3 = 20
    2 I1 + 8 I2 -   I3 = 23

        [ 1  4  2 ]          [13]
    A = [ 5  2  1 ]     b =  [20]
        [ 2  8 -1 ]          [23]

Procedimiento:  [A | I]  --(Gauss-Jordan)-->  [I | A^-1]   luego   x = A^-1 * b
"""

import numpy as np
from fractions import Fraction

np.set_printoptions(precision=4, suppress=True)

# ---------------------------------------------------------------
# 1. Datos del sistema (tal como en la imagen)
# ---------------------------------------------------------------
A = np.array([
    [1, 4,  2],
    [5, 2,  1],
    [2, 8, -1],
], dtype=float)

b = np.array([13, 20, 23], dtype=float)

n = A.shape[0]

# ---------------------------------------------------------------
# 2. Gauss-Jordan manual: [A | I] -> [I | A^-1]
#    (mismo procedimiento paso a paso que en la hoja)
# ---------------------------------------------------------------
def inversa_gauss_jordan(A, mostrar_pasos=True):
    n = A.shape[0]
    M = np.hstack([A.copy(), np.eye(n)])   # matriz aumentada [A | I]

    if mostrar_pasos:
        print("Matriz aumentada inicial [A | I]:")
        print(M, "\n")

    for col in range(n):
        # Pivoteo parcial si el pivote es 0
        if abs(M[col, col]) < 1e-12:
            for f in range(col + 1, n):
                if abs(M[f, col]) > 1e-12:
                    M[[col, f]] = M[[f, col]]
                    break

        # Normalizar la fila pivote (R_col / pivote)
        pivote = M[col, col]
        M[col, :] = M[col, :] / pivote
        if mostrar_pasos:
            print(f"Normalizar R{col+1} / {pivote:.4f}:")
            print(M, "\n")

        # Eliminar la columna 'col' en las demas filas
        for f in range(n):
            if f != col:
                factor = M[f, col]
                if factor != 0:
                    M[f, :] = M[f, :] - factor * M[col, :]
        if mostrar_pasos:
            print(f"Eliminar columna {col+1} en las demas filas:")
            print(M, "\n")

    A_inv = M[:, n:]
    return A_inv


print("=" * 60)
print("PASO A PASO (Gauss-Jordan)")
print("=" * 60)
A_inv = inversa_gauss_jordan(A, mostrar_pasos=True)

# ---------------------------------------------------------------
# 3. Resultado: A^-1
# ---------------------------------------------------------------
print("=" * 60)
print("Matriz inversa A^-1 (decimal):")
print("=" * 60)
print(A_inv)

print("\nMatriz inversa A^-1 (fracciones, para comparar con la hoja):")
for fila in A_inv:
    print([Fraction(v).limit_denominator(100) for v in fila])

# ---------------------------------------------------------------
# 4. Solucion del sistema: x = A^-1 * b
# ---------------------------------------------------------------
x = A_inv @ b
print("\n" + "=" * 60)
print("Solucion:  [I1, I2, I3] = A^-1 * b")
print("=" * 60)
for nombre, valor in zip(["I1", "I2", "I3"], x):
    print(f"{nombre} = {valor:.4f}")

# ---------------------------------------------------------------
# 5. Verificaciones
# ---------------------------------------------------------------
print("\nVerificacion A * A^-1 = I:")
print(np.round(A @ A_inv, 4))

print("\nVerificacion A * x = b  (debe dar [13, 20, 23]):")
print(np.round(A @ x, 4))

x_check = np.linalg.solve(A, b)
print("\nVerificacion con numpy.linalg.solve:", np.round(x_check, 4))