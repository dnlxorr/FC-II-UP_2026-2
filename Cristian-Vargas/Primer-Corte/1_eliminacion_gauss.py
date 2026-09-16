"""
=============================================================================
 TEMA: Eliminacion de Gauss (Eliminacion hacia adelante)
 CURSO: Fisica Computacional II
=============================================================================

EJEMPLO FISICO: Analisis de mallas (Kirchhoff) en un circuito resistivo
-----------------------------------------------------------------------
Se tiene un circuito de tres mallas con resistencias y dos fuentes de
voltaje. Aplicando la Ley de Voltajes de Kirchhoff (LVK) a cada malla se
obtiene un sistema lineal 3x3 para las corrientes de malla I1, I2, I3.

Circuito (valores en Ohm y Voltios):

        R1=10        R2=20        R4=15
   +---/\/\/---+---/\/\/---+---/\/\/---+
   |    ->     |    ->     |    ->     |
  V1=50        R3=25      R5=30       V2=20
  (malla 1)  (comun 1-2) (comun 2-3) (malla 3, polaridad opuesta)
   |           |           |           |
   +-----------+-----------+-----------+

Ecuaciones de malla (sentido horario para I1, I2, I3):

  Malla 1:  (R1+R3) I1  - R3 I2            = V1
  Malla 2:   -R3 I1  + (R2+R3+R5) I2 - R5 I3 = 0
  Malla 3:            - R5 I2 + (R4+R5) I3 = -V2

Este es exactamente el tipo de sistema Ax = b que aparece constantemente
en electromagnetismo y circuitos: la matriz A representa el "acoplamiento"
resistivo entre mallas, y b representa las fuentes.

"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


def eliminacion_gauss(A, b, verbose=True):
    """
    Reduce el sistema [A|b] a forma triangular superior [U|c]
    usando eliminacion de Gauss SIN pivoteo (pivoteo se trata en el
    script 3_pivoteo.py).

    Parametros
    ----------
    A : ndarray (n,n)  matriz de coeficientes
    b : ndarray (n,)   vector de terminos independientes

    Retorna
    -------
    U : ndarray (n,n)  matriz triangular superior
    c : ndarray (n,)   vector transformado
    """
    n = len(b)
    U = A.astype(float).copy()
    c = b.astype(float).copy()

    for k in range(n - 1):                      # columna pivote
        if verbose:
            print(f"\n--- Paso de eliminacion k={k} (pivote a_{k}{k}={U[k,k]:.4f}) ---")
        if abs(U[k, k]) < 1e-14:
            raise ZeroDivisionError(
                f"Pivote nulo en posicion ({k},{k}); se requiere pivoteo."
            )
        for i in range(k + 1, n):                # filas por debajo del pivote
            m_ik = U[i, k] / U[k, k]              # factor de eliminacion
            U[i, k:] = U[i, k:] - m_ik * U[k, k:]  # fila_i -= m_ik * fila_k
            c[i] = c[i] - m_ik * c[k]
            if verbose:
                print(f"  fila {i} <- fila {i} - ({m_ik:.4f}) * fila {k}")
    return U, c


def sustitucion_simple(U, c):
    """Back-substitution basica solo para verificar la solucion aqui.
    El desarrollo completo del metodo se explica en el script 2."""
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# -----------------------------------------------------------------------
# 1) Definicion del sistema fisico (circuito de 3 mallas)
# -----------------------------------------------------------------------
R1, R2, R3, R4, R5 = 10.0, 20.0, 25.0, 15.0, 30.0
V1, V2 = 50.0, 20.0

A = np.array([
    [R1 + R3,      -R3,          0.0],
    [-R3,      R2 + R3 + R5,    -R5],
    [0.0,          -R5,      R4 + R5],
])
b = np.array([V1, 0.0, -V2])

print("Sistema fisico (analisis de mallas):")
print("A =\n", A)
print("b =", b)

# -----------------------------------------------------------------------
# 2) Eliminacion de Gauss paso a paso
# -----------------------------------------------------------------------
U, c = eliminacion_gauss(A, b, verbose=True)
print("\nMatriz triangular superior U:\n", U)
print("Vector transformado c:", c)

x_gauss = sustitucion_simple(U, c)

# -----------------------------------------------------------------------
# 3) Verificacion / analisis de error
# -----------------------------------------------------------------------
# Solucion de referencia con un algoritmo confiable (numpy = LAPACK, usa
# pivoteo parcial internamente), para poder cuantificar el error de
# NUESTRA implementacion didactica sin pivoteo.
x_ref = np.linalg.solve(A, b)

error_abs = np.abs(x_gauss - x_ref)
residuo = A @ x_gauss - b                      # que tan bien satisface Ax=b
norma_residuo = np.linalg.norm(residuo, ord=2)

print("\n--- Resultados ---")
print(f"Corrientes de malla (Gauss manual):  I = {x_gauss} A")
print(f"Corrientes de malla (numpy.solve):   I = {x_ref} A")
print(f"Error absoluto por componente:       {error_abs}")
print(f"Norma del residuo ||Ax-b||_2:        {norma_residuo:.3e}")
print(f"Numero de condicion de A:            {np.linalg.cond(A):.3f}")

# -----------------------------------------------------------------------
# 4) Graficas
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Comparacion de corrientes: Gauss manual vs numpy
etiquetas = [r"$I_1$", r"$I_2$", r"$I_3$"]
xpos = np.arange(3)
ancho = 0.35
axes[0].bar(xpos - ancho / 2, x_gauss, ancho, label="Gauss (manual)", color="#2E86AB")
axes[0].bar(xpos + ancho / 2, x_ref, ancho, label="numpy.linalg.solve", color="#A23B72")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(etiquetas)
axes[0].set_ylabel("Corriente (A)")
axes[0].set_title("Corrientes de malla: metodo propio vs referencia")
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) Error absoluto por componente (escala log para ver ordenes de magnitud)
axes[1].bar(etiquetas, error_abs + 1e-18, color="#F18F01")
axes[1].set_yscale("log")
axes[1].set_ylabel("Error absoluto |I_gauss - I_ref|  (A)")
axes[1].set_title("Error de la eliminacion de Gauss (sin pivoteo)")
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("1_eliminacion_gauss.png", dpi=150)
print("\nGrafica guardada en 1_eliminacion_gauss.png")
