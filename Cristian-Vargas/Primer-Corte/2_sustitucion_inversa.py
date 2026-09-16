r"""
=============================================================================
 TEMA: Sustitucion inversa (back substitution)
 CURSO: Fisica Computacional II
=============================================================================

CONTEXTO: la sustitucion inversa es la SEGUNDA mitad de la eliminacion de
Gauss (ver script 1_eliminacion_gauss.py). Una vez que el sistema Ax = b
se redujo a la forma triangular superior

        U x = c ,   con U triangular superior,

la incognita x_{n-1} (la ULTIMA) queda despejada de forma inmediata porque
la ultima fila de U solo tiene un termino no nulo en la diagonal:

        U[n-1,n-1] * x_{n-1} = c_{n-1}   =>   x_{n-1} = c_{n-1} / U[n-1,n-1]

y luego, conocido x_{n-1}, la penultima fila tiene solo dos incognitas
(x_{n-2} y x_{n-1}, ya conocida), por lo que se despeja x_{n-2}, y asi
sucesivamente HACIA ATRAS (de ahi el nombre). En general:

        x_i = ( c_i - sum_{j=i+1}^{n-1} U[i,j] x_j ) / U[i,i]


"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


def eliminacion_gauss_resumida(A, b):
    """Misma eliminacion del script 1, sin impresiones (ya se explico alli)."""
    n = len(b)
    U = A.astype(float).copy()
    c = b.astype(float).copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            m_ik = U[i, k] / U[k, k]
            U[i, k:] -= m_ik * U[k, k:]
            c[i] -= m_ik * c[k]
    return U, c


def sustitucion_inversa(U, c, verbose=True):
    """
    Resuelve U x = c para U triangular superior, recorriendo las filas
    de abajo hacia arriba (por eso "inversa" / "hacia atras").
    """
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):                 # de la ultima fila a la primera
        if abs(U[i, i]) < 1e-14:
            raise ZeroDivisionError(f"Elemento diagonal nulo en U[{i},{i}]")
        suma = U[i, i + 1:] @ x[i + 1:]             # aporte de incognitas ya conocidas
        x[i] = (c[i] - suma) / U[i, i]
        if verbose:
            print(f"  x[{i}] = (c[{i}] - suma_conocidos) / U[{i},{i}] "
                  f"= ({c[i]:.4f} - {suma:.4f}) / {U[i,i]:.4f} = {x[i]:.6f}")
    return x


# -----------------------------------------------------------------------
# 1) Mismo sistema fisico del script 1: circuito de 3 mallas (LVK)
# -----------------------------------------------------------------------
R1, R2, R3, R4, R5 = 10.0, 20.0, 25.0, 15.0, 30.0
V1, V2 = 50.0, 20.0

A = np.array([
    [R1 + R3,      -R3,          0.0],
    [-R3,      R2 + R3 + R5,    -R5],
    [0.0,          -R5,      R4 + R5],
])
b = np.array([V1, 0.0, -V2])

U, c = eliminacion_gauss_resumida(A, b)
print("Sistema ya triangularizado (resultado del script 1):")
print("U =\n", U)
print("c =", c)

print("\nSustitucion inversa paso a paso:")
x = sustitucion_inversa(U, c, verbose=True)
I1, I2, I3 = x
print(f"\nCorrientes de malla: I1={I1:.4f} A, I2={I2:.4f} A, I3={I3:.4f} A")

# -----------------------------------------------------------------------
# 2) Verificacion / analisis de error
# -----------------------------------------------------------------------
x_ref = np.linalg.solve(A, b)          # referencia: sistema ORIGINAL con LAPACK
residuo_U = U @ x - c                  # que tan bien resuelve el sistema triangular
residuo_A = A @ x - b                  # que tan bien resuelve el sistema fisico original
error_abs = np.abs(x - x_ref)

print(f"\n||U x - c||_2 (error del algoritmo de sustitucion):  {np.linalg.norm(residuo_U):.3e}")
print(f"||A x - b||_2 (error respecto al sistema fisico):     {np.linalg.norm(residuo_A):.3e}")
print(f"Error absoluto vs numpy.linalg.solve: {error_abs}")

# -----------------------------------------------------------------------
# 3) Graficas
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

etiquetas = [r"$I_1$", r"$I_2$", r"$I_3$"]
axes[0].bar(etiquetas, x, color="#2E86AB", label="Sustitucion inversa")
axes[0].plot(etiquetas, x_ref, "o--", color="#A23B72", label="numpy.linalg.solve")
axes[0].set_ylabel("Corriente de malla (A)")
axes[0].set_title("Corrientes obtenidas por sustitucion inversa")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Evolucion del residuo fila a fila: muestra que cada fila queda
# EXACTAMENTE satisfecha en el momento en que se resuelve (el unico error
# remanente es de redondeo de punto flotante, no error de metodo)
residuos_parciales = []
n = len(c)
x_parcial = np.zeros(n)
for i in range(n - 1, -1, -1):
    suma = U[i, i + 1:] @ x_parcial[i + 1:]
    x_parcial[i] = (c[i] - suma) / U[i, i]
    fila_actual = U[i, :] @ x_parcial - c[i]
    residuos_parciales.append(abs(fila_actual))

axes[1].semilogy(range(n, 0, -1), np.array(residuos_parciales) + 1e-18,
                  "o-", color="#F18F01")
axes[1].set_xlabel("Fila resuelta (orden: n -> 1)")
axes[1].set_ylabel("|residuo de esa fila| (log)")
axes[1].set_title("Residuo por fila al momento de resolverla")
axes[1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("2_sustitucion_inversa.png", dpi=150)
print("\nGrafica guardada en 2_sustitucion_inversa.png")
