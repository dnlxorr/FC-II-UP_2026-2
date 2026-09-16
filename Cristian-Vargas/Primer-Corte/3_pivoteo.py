r"""
=============================================================================
 TEMA: Pivoteo (pivoteo parcial en eliminacion de Gauss)
 CURSO: Fisica Computacional II
=============================================================================


EJEMPLO FISICO 1 (division por cero por orden de las ecuaciones)
---------------------------------------------------------------------------
Tres ecuaciones de balance de un circuito, escritas en un orden en el que,
por construccion, la primera ecuacion no involucra la primera incognita
(coeficiente cero). Esto es habitual: el orden en que un estudiante escribe
las ecuaciones de malla/nodo es arbitrario, no viene "ordenado" para el
algoritmo.

EJEMPLO FISICO 2 (pivote pequeno con problema BIEN condicionado)
---------------------------------------------------------------------------
Un elemento termoelectrico (efecto Seebeck/Peltier) descrito por dos
ecuaciones acopladas, una electrica y una termica, en las incognitas
DT (diferencia de temperatura) e I (corriente). El coeficiente Seebeck S
es varios ordenes de magnitud menor que la conductancia termica K, y
ocupa la posicion diagonal (0,0): el factor de eliminacion sin pivoteo es
m = K/S, enorme. Se hace un barrido de S para mostrar que el numero de
condicion del problema NO cambia (el problema fisico es estable) pero el
error del algoritmo sin pivoteo crece proporcional a m. Esto separa
claramente "mal condicionamiento del problema" de "inestabilidad del
algoritmo".
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


def gauss_con_pivoteo(A, b, pivotear=True, verbose=True):
    """
    Eliminacion de Gauss con pivoteo parcial opcional.
    Si pivotear=False, se comporta como el script 1 (para poder comparar).
    """
    n = len(b)
    U = A.astype(float).copy()
    c = b.astype(float).copy()

    for k in range(n - 1):
        if pivotear:
            # Busca, en la columna k, la fila (desde k en adelante) con
            # el mayor valor absoluto: ese sera el nuevo pivote.
            fila_max = k + np.argmax(np.abs(U[k:, k]))
            if fila_max != k:
                U[[k, fila_max]] = U[[fila_max, k]]   # intercambio de filas
                c[[k, fila_max]] = c[[fila_max, k]]
                if verbose:
                    print(f"  Pivoteo: se intercambia fila {k} <-> fila {fila_max} "
                          f"(nuevo pivote = {U[k,k]:.6g})")
        if abs(U[k, k]) < 1e-14:
            msg = "incluso tras pivotear" if pivotear else "(se requiere pivoteo)"
            raise ZeroDivisionError(f"Pivote nulo en columna {k} {msg}")
        for i in range(k + 1, n):
            m_ik = U[i, k] / U[k, k]
            U[i, k:] -= m_ik * U[k, k:]
            c[i] -= m_ik * c[k]
    return U, c


def sustitucion_inversa(U, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# =========================================================================
# EJEMPLO 1: division por cero evitable con pivoteo
# =========================================================================
print("=" * 70)
print("EJEMPLO 1: pivote nulo por orden de las ecuaciones")
print("=" * 70)

# Sistema fisico valido (tiene solucion unica), pero la ecuacion 1 no
# contiene la incognita 1 por como se planteo el balance:
A1 = np.array([
    [0.0, 2.0, 3.0],
    [4.0, 6.0, 1.0],
    [2.0, -1.0, 5.0],
])
b1 = np.array([5.0, 13.0, 9.0])

print("Intentando SIN pivoteo (columna 0, fila 0 = 0):")
try:
    gauss_con_pivoteo(A1, b1, pivotear=False, verbose=True)
except ZeroDivisionError as err:
    print(f"  -> FALLA como se esperaba: {err}")

print("\nCon pivoteo parcial:")
U1, c1 = gauss_con_pivoteo(A1, b1, pivotear=True, verbose=True)
x1 = sustitucion_inversa(U1, c1)
x1_ref = np.linalg.solve(A1, b1)
print(f"Solucion: x = {x1}")
print(f"Error vs numpy: {np.abs(x1 - x1_ref)}")


# =========================================================================
# EJEMPLO 2: pivote pequeno con matriz BIEN condicionada
#            (acoplamiento termoelectrico debil)
# =========================================================================
print("\n" + "=" * 70)
print("EJEMPLO 2: pivote pequeno por acoplamiento termoelectrico debil")
print("=" * 70)
print("""
Modelo de dos ecuaciones acopladas en un elemento termoelectrico (efecto
Seebeck/Peltier). Incognitas: la diferencia de temperatura DT (K) entre
las caras y la corriente I (A) que circula:

  Ec. electrica:   S*DT  -  R*I   = V_medido
  Ec. termica:     K*DT  +  Pi*I  = Q_disipado

donde S es el coeficiente Seebeck (V/K), R la resistencia electrica (Ohm),
K la conductancia termica (W/K) y Pi = S*T el coeficiente Peltier (V).

El punto clave: S es un numero MUY pequeno comparado con R y K (para un
metal comun S ~ 1e-6 V/K; para un semiconductor termoelectrico ~ 1e-4 V/K),
pero aparece en la posicion DIAGONAL (0,0) de la matriz. Si se hace la
eliminacion sin pivotear, el factor de eliminacion es m = K/S, que puede
ser de orden 1e6 o mas: exactamente la situacion de amplificacion del
error de redondeo descrita arriba.
""")

R_e = 5.0         # resistencia electrica del elemento (Ohm)
K_t = 0.5         # conductancia termica (W/K)
Pi = 0.06         # coeficiente Peltier (V)

# Solucion "verdadera" impuesta (para conocer el error exacto):
# DT = 3 K, I = 0.4 A. Se construye b = A x_verdadero, de modo que
# cualquier desviacion respecto a x_verdadero es error puramente numerico.
x_verdadero = np.array([3.0, 0.4])

# Barrido del coeficiente Seebeck. Los valores grandes (1e-2..1e-6)
# corresponden a materiales reales (termoelectricos y metales); los
# valores mas pequenos (1e-10..1e-14) ya NO corresponden a un material
# fisico real, y se incluyen solo para exponer con claridad la TENDENCIA
# del error del algoritmo: son un experimento numerico controlado, no una
# prediccion fisica.
valores_S = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])

errores_con_pivoteo = []
errores_sin_pivoteo = []
condiciones = []
factores_m = []

for S in valores_S:
    A2 = np.array([
        [S,    -R_e],
        [K_t,   Pi ],
    ])
    b2 = A2 @ x_verdadero

    x_p = sustitucion_inversa(*gauss_con_pivoteo(A2, b2, pivotear=True, verbose=False))
    x_np = sustitucion_inversa(*gauss_con_pivoteo(A2, b2, pivotear=False, verbose=False))

    norma = np.linalg.norm(x_verdadero)
    errores_con_pivoteo.append(np.linalg.norm(x_p - x_verdadero) / norma)
    errores_sin_pivoteo.append(np.linalg.norm(x_np - x_verdadero) / norma)
    condiciones.append(np.linalg.cond(A2))
    factores_m.append(K_t / S)          # factor de eliminacion sin pivoteo

errores_con_pivoteo = np.array(errores_con_pivoteo)
errores_sin_pivoteo = np.array(errores_sin_pivoteo)
condiciones = np.array(condiciones)
factores_m = np.array(factores_m)

print(f"{'S (V/K)':>10} | {'cond(A)':>10} | {'m=K/S':>10} | "
      f"{'err CON pivoteo':>16} | {'err SIN pivoteo':>16}")
for S, cnd, m, ep, enp in zip(valores_S, condiciones, factores_m,
                              errores_con_pivoteo, errores_sin_pivoteo):
    print(f"{S:10.0e} | {cnd:10.3e} | {m:10.2e} | {ep:16.3e} | {enp:16.3e}")

# -----------------------------------------------------------------------
# Graficas
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].loglog(valores_S, errores_sin_pivoteo + 1e-20, "o-",
               color="#C1121F", label="Sin pivoteo")
axes[0].loglog(valores_S, errores_con_pivoteo + 1e-20, "s-",
               color="#2E86AB", label="Con pivoteo parcial")
axes[0].axhline(np.finfo(float).eps, ls=":", color="gray",
                label=r"$\epsilon_{maquina}\approx 2.2\times10^{-16}$")
axes[0].set_xlabel(r"Coeficiente Seebeck $S$ (V/K)  [pivote diagonal]")
axes[0].set_ylabel("Error relativo de la solucion")
axes[0].set_title("Efecto del pivoteo al reducirse el pivote")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3, which="both")
axes[0].invert_xaxis()      # pivote mas pequeno (peor caso) hacia la derecha

axes[1].loglog(valores_S, condiciones, "d-", color="#F18F01",
               label=r"cond$(A)$")
axes[1].loglog(valores_S, factores_m, "v-", color="#6A4C93",
               label=r"factor $m=K/S$ (sin pivoteo)")
axes[1].set_xlabel(r"Coeficiente Seebeck $S$ (V/K)")
axes[1].set_ylabel("Magnitud")
axes[1].set_title("El PROBLEMA sigue bien condicionado;\nlo que explota es el ALGORITMO")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")
axes[1].invert_xaxis()

plt.tight_layout()
plt.savefig("3_pivoteo.png", dpi=150)
print("\nGrafica guardada en 3_pivoteo.png")

print("""
CONCLUSION IMPORTANTE (leer con atencion):
El numero de condicion de esta matriz se mantiene practicamente CONSTANTE
(~10) en todo el barrido: el PROBLEMA FISICO esta bien condicionado y su
solucion es perfectamente estable frente a perturbaciones de los datos.
Sin embargo, el error de la eliminacion SIN pivoteo crece proporcional al
factor m = K/S, hasta perder casi todas las cifras significativas. Con
pivoteo parcial el error se mantiene al nivel del epsilon de maquina.

Esto separa dos conceptos que suelen confundirse:
  - Mal condicionamiento  -> defecto del PROBLEMA (ninguna algoritmo lo arregla).
  - Inestabilidad numerica -> defecto del ALGORITMO (el pivoteo SI lo arregla).
Aqui el problema es bueno y el algoritmo sin pivoteo es malo: por eso el
pivoteo parcial es el estandar en toda libreria seria (LAPACK, y por tanto
numpy.linalg.solve, lo usan siempre por defecto).
""")
