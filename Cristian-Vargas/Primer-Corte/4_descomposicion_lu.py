r"""
=============================================================================
 TEMA: Descomposicion LU (metodo de Doolittle, con pivoteo parcial PA = LU)
 CURSO: Fisica Computacional II
=============================================================================


EJEMPLO FISICO: armadura plana (cercha) en equilibrio estatico
---------------------------------------------------------------------------
Una armadura de barras articuladas en equilibrio: en cada nodo libre, la
suma vectorial de las fuerzas axiales de las barras que concurren mas las
cargas externas debe ser cero (equilibrio estatico, 2 ecuaciones por nodo:
sum Fx = 0, sum Fy = 0). Las incognitas son las tensiones/compresiones T_j
en cada barra y las reacciones en los apoyos. Esto da un sistema A T = b
donde:
  - A (matriz de geometria) contiene los cosenos directores de las barras:
    depende SOLO de la geometria de la estructura, que no cambia.
  - b contiene las cargas externas aplicadas: cambia con cada caso de carga.

Este es el escenario ideal para LU: una estructura fija sometida a muchos
casos de carga (peso propio, viento de izquierda, viento de derecha, carga
de nieve, combinaciones reglamentarias...). Se factoriza la geometria una
sola vez y se resuelven todos los casos de carga con sustituciones baratas.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)


# =========================================================================
# 1) Implementacion de la descomposicion LU con pivoteo parcial (Doolittle)
# =========================================================================
def descomposicion_lu(A, verbose=False):
    """
    Calcula P, L, U tales que P @ A = L @ U.

    L: triangular inferior con 1 en la diagonal (guarda los factores m_ik).
    U: triangular superior (resultado de la eliminacion).
    P: matriz de permutacion que registra los intercambios de filas.
    """
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        # --- pivoteo parcial (misma justificacion del script 3) ---
        fila_max = k + np.argmax(np.abs(U[k:, k]))
        if fila_max != k:
            U[[k, fila_max], :] = U[[fila_max, k], :]
            P[[k, fila_max], :] = P[[fila_max, k], :]
            # IMPORTANTE: los factores ya calculados (columnas < k) tambien
            # deben permutarse, porque pertenecen a las filas intercambiadas.
            L[[k, fila_max], :k] = L[[fila_max, k], :k]
            if verbose:
                print(f"  intercambio filas {k} <-> {fila_max}")

        if abs(U[k, k]) < 1e-14:
            raise np.linalg.LinAlgError("Matriz singular: no admite factorizacion LU")

        # --- eliminacion, GUARDANDO los factores en L ---
        for i in range(k + 1, n):
            m_ik = U[i, k] / U[k, k]
            L[i, k] = m_ik              # <-- unica diferencia con Gauss puro
            U[i, k:] -= m_ik * U[k, k:]
        U[k + 1:, k] = 0.0              # limpieza de ceros numericos

    return P, L, U


def sustitucion_adelante(L, b):
    """Resuelve L y = b con L triangular INFERIOR y diagonal unitaria.
    Se recorre de la primera fila a la ultima: la fila 0 tiene una sola
    incognita (y_0), y cada fila siguiente solo involucra incognitas ya
    calculadas. Es el espejo exacto de la sustitucion inversa."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def sustitucion_atras(U, y):
    """Resuelve U x = y (identica al script 2)."""
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def resolver_con_lu(P, L, U, b):
    """Resuelve A x = b usando la factorizacion P A = L U ya calculada."""
    return sustitucion_atras(U, sustitucion_adelante(L, P @ b))


# =========================================================================
# 2) Sistema fisico: armadura plana de 3 nodos libres
# =========================================================================
# Geometria: armadura tipo triangular con apoyos en los extremos.
#
#              N2 (nodo superior)
#             /  \
#            /    \
#          N1------N3
#          ^                ^
#       apoyo fijo       apoyo movil
#
# Barras: b1 = N1-N2, b2 = N2-N3, b3 = N1-N3
# Incognitas: T1, T2, T3 (fuerzas axiales), Ry1, Ry3 (reacciones verticales),
#             Rx1 (reaccion horizontal del apoyo fijo)  -> 6 incognitas
# Ecuaciones: equilibrio en x e y de los 3 nodos -> 6 ecuaciones

L_base = 4.0          # longitud de la base N1-N3 (m)
altura = 3.0          # altura del nodo N2 (m)

# cosenos directores de las barras inclinadas
Lb = np.hypot(L_base / 2, altura)
cos_a = (L_base / 2) / Lb     # componente horizontal
sin_a = altura / Lb           # componente vertical

# Orden de incognitas: [T1, T2, T3, Rx1, Ry1, Ry3]
A = np.array([
    # Nodo 1: sum Fx = 0   (T1 hacia N2, T3 hacia N3)
    [ cos_a,   0.0,   1.0,   1.0,   0.0,   0.0],
    # Nodo 1: sum Fy = 0
    [ sin_a,   0.0,   0.0,   0.0,   1.0,   0.0],
    # Nodo 2: sum Fx = 0
    [-cos_a, cos_a,   0.0,   0.0,   0.0,   0.0],
    # Nodo 2: sum Fy = 0
    [-sin_a,-sin_a,   0.0,   0.0,   0.0,   0.0],
    # Nodo 3: sum Fx = 0
    [   0.0,-cos_a,  -1.0,   0.0,   0.0,   0.0],
    # Nodo 3: sum Fy = 0
    [   0.0, sin_a,   0.0,   0.0,   0.0,   1.0],
])

nombres = ["T1", "T2", "T3", "Rx1", "Ry1", "Ry3"]

print("Matriz de geometria de la armadura (no depende de las cargas):")
print(A)
print(f"\nNumero de condicion: {np.linalg.cond(A):.3f}")

# --- factorizacion UNICA ---
P, L, U = descomposicion_lu(A, verbose=True)

print("\nL =\n", L)
print("\nU =\n", U)
print("\nVerificacion de la factorizacion:")
error_fact = np.linalg.norm(P @ A - L @ U)
print(f"  ||P A - L U||_2 = {error_fact:.3e}")

# =========================================================================
# 3) Multiples casos de carga (la razon de ser de LU)
# =========================================================================
# b contiene el negativo de las cargas externas en cada grado de libertad,
# en el mismo orden de ecuaciones: [N1x, N1y, N2x, N2y, N3x, N3y]
casos_de_carga = {
    "Peso propio (10 kN en N2)":        np.array([0.0, 0.0, 0.0,  10.0, 0.0, 0.0]),
    "Viento izquierda (6 kN en N2)":    np.array([0.0, 0.0, -6.0, 0.0,  0.0, 0.0]),
    "Viento derecha (6 kN en N2)":      np.array([0.0, 0.0,  6.0, 0.0,  0.0, 0.0]),
    "Nieve (15 kN en N2)":              np.array([0.0, 0.0, 0.0,  15.0, 0.0, 0.0]),
    "Combinada (peso+viento+nieve)":    np.array([0.0, 0.0, -6.0, 25.0, 0.0, 0.0]),
}

print("\n" + "=" * 72)
print("RESULTADOS POR CASO DE CARGA (una sola factorizacion LU)")
print("=" * 72)

resultados = {}
errores_residuo = []
for nombre, b in casos_de_carga.items():
    x = resolver_con_lu(P, L, U, b)
    resultados[nombre] = x
    residuo = np.linalg.norm(A @ x - b)
    errores_residuo.append(residuo)
    print(f"\n{nombre}")
    for nom, val in zip(nombres, x):
        estado = ""
        if nom.startswith("T"):
            estado = "  (traccion)" if val > 0 else "  (compresion)"
        print(f"   {nom:>4} = {val:9.4f} kN{estado}")
    print(f"   residuo ||A x - b||_2 = {residuo:.3e}")

# =========================================================================
# 4) Comparacion de costo: LU (1 factorizacion) vs Gauss (m eliminaciones)
# =========================================================================
def gauss_completo(A, b):
    """Eliminacion + sustitucion, repitiendo TODO el trabajo cada vez."""
    n = len(b)
    U_ = A.astype(float).copy()
    c_ = b.astype(float).copy()
    for k in range(n - 1):
        fm = k + np.argmax(np.abs(U_[k:, k]))
        if fm != k:
            U_[[k, fm]] = U_[[fm, k]]
            c_[[k, fm]] = c_[[fm, k]]
        for i in range(k + 1, n):
            m = U_[i, k] / U_[k, k]
            U_[i, k:] -= m * U_[k, k:]
            c_[i] -= m * c_[k]
    return sustitucion_atras(U_, c_)


print("\n" + "=" * 72)
print("COSTO COMPUTACIONAL: LU vs Gauss repetido")
print("=" * 72)

tamanos = [20, 40, 60, 80, 100]
m_cargas = 40          # numero de casos de carga a resolver
t_lu, t_gauss = [], []

rng = np.random.default_rng(0)
for n in tamanos:
    # Matriz de prueba diagonalmente dominante (bien condicionada, como
    # suelen ser las matrices de rigidez/geometria de estructuras reales)
    M = rng.random((n, n)) + n * np.eye(n)
    Bs = rng.random((n, m_cargas))

    t0 = time.perf_counter()
    Pn, Ln, Un = descomposicion_lu(M)            # UNA vez
    for j in range(m_cargas):
        resolver_con_lu(Pn, Ln, Un, Bs[:, j])    # barato
    t_lu.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    for j in range(m_cargas):
        gauss_completo(M, Bs[:, j])              # eliminacion COMPLETA cada vez
    t_gauss.append(time.perf_counter() - t0)

    print(f"  n={n:4d}: LU = {t_lu[-1]*1000:8.2f} ms | "
          f"Gauss x{m_cargas} = {t_gauss[-1]*1000:8.2f} ms | "
          f"speedup = {t_gauss[-1]/t_lu[-1]:.1f}x")

# =========================================================================
# 5) Graficas
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Fuerzas en las barras por caso de carga
ancho = 0.15
xpos = np.arange(3)
for idx, (nombre, x) in enumerate(resultados.items()):
    axes[0].bar(xpos + idx * ancho, x[:3], ancho, label=nombre.split("(")[0].strip())
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xticks(xpos + 2 * ancho)
axes[0].set_xticklabels(["Barra 1", "Barra 2", "Barra 3"])
axes[0].set_ylabel("Fuerza axial (kN)  [+traccion / -compresion]")
axes[0].set_title("Fuerzas en las barras por caso de carga")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

# (b) Error residual de cada caso resuelto con la MISMA factorizacion
axes[1].semilogy(range(1, len(errores_residuo) + 1),
                 np.array(errores_residuo) + 1e-18, "o-", color="#F18F01")
axes[1].axhline(np.finfo(float).eps, ls=":", color="gray",
                label=r"$\epsilon_{maquina}$")
axes[1].set_xlabel("Caso de carga")
axes[1].set_ylabel(r"$\|Ax-b\|_2$")
axes[1].set_title("Error residual (una sola factorizacion)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

# (c) Tiempos
axes[2].plot(tamanos, np.array(t_gauss) * 1000, "o-", color="#C1121F",
             label=f"Gauss repetido x{m_cargas}")
axes[2].plot(tamanos, np.array(t_lu) * 1000, "s-", color="#2E86AB",
             label="LU (1 factorizacion)")
axes[2].set_xlabel("Tamano del sistema n")
axes[2].set_ylabel("Tiempo (ms)")
axes[2].set_title(f"Costo para {m_cargas} casos de carga")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("4_descomposicion_lu.png", dpi=150)
print("\nGrafica guardada en 4_descomposicion_lu.png")
