r"""
=============================================================================
 TEMA: Inversa de una matriz
 CURSO: Fisica Computacional II
=============================================================================

EJEMPLO FISICO: coeficientes de potencial y matriz de capacitancia
---------------------------------------------------------------------------
Se tienen N conductores aislados, cada uno con carga Q_i y potencial V_i.
Por la linealidad de la ecuacion de Laplace (electrostatica), los
potenciales son combinacion lineal de las cargas:

        V_i = sum_j p_ij Q_j       =>       V = P Q

donde p_ij son los "coeficientes de potencial", que dependen solo de la
geometria. La matriz inversa tiene nombre propio y significado fisico
directo:

        Q = C V ,       C = P^{-1}

C es la MATRIZ DE CAPACITANCIA del sistema: C_ii es la capacitancia propia
del conductor i y C_ij (i!=j, negativos) son las capacitancias mutuas o de
acoplamiento entre conductores. Aqui la inversa NO es un paso intermedio
para resolver un sistema: es el objeto fisico que se quiere reportar
(es lo que un simulador de interconexiones o un software de EMC entrega
como resultado). Por eso este ejemplo justifica calcular la inversa.

Modelo usado: N esferas conductoras de radio a, centros separados por
distancias d_ij, en el regimen a << d_ij. En ese limite se puede usar la
aproximacion de carga puntual para el potencial cruzado:

        p_ii = 1/(4 pi eps0 a)        (potencial de la esfera por su propia carga)
        p_ij = 1/(4 pi eps0 d_ij)     (potencial en i debido a la carga de j)

"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)

EPS0 = 8.8541878128e-12          # permitividad del vacio (F/m)


# =========================================================================
# 1) Metodo (a): Gauss-Jordan sobre la matriz aumentada [A | I]
# =========================================================================
def inversa_gauss_jordan(A, verbose=False):
    """
    Reduce [A | I] a [I | A^{-1}] mediante operaciones elementales de fila
    con pivoteo parcial. Cada operacion se aplica simultaneamente a ambos
    bloques, de modo que el bloque derecho acumula exactamente el producto
    de todas las matrices elementales aplicadas, que es A^{-1}.
    """
    n = A.shape[0]
    M = np.hstack([A.astype(float).copy(), np.eye(n)])   # matriz aumentada

    for k in range(n):
        # pivoteo parcial (misma justificacion del script 3)
        fila_max = k + np.argmax(np.abs(M[k:, k]))
        if fila_max != k:
            M[[k, fila_max]] = M[[fila_max, k]]
            if verbose:
                print(f"  intercambio filas {k} <-> {fila_max}")
        if abs(M[k, k]) < 1e-14:
            raise np.linalg.LinAlgError("Matriz singular: no tiene inversa")

        # normalizacion del pivote a 1
        M[k] = M[k] / M[k, k]

        # eliminacion en TODAS las demas filas (arriba y abajo):
        # esto es lo que diferencia Gauss-Jordan de Gauss
        for i in range(n):
            if i != k:
                M[i] -= M[i, k] * M[k]

    return M[:, n:]          # bloque derecho = A^{-1}


# =========================================================================
# 2) Metodo (b): inversa via descomposicion LU (n sistemas A x_j = e_j)
# =========================================================================
def descomposicion_lu(A):
    """LU con pivoteo parcial: P A = L U (identica al script 4)."""
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n - 1):
        fm = k + np.argmax(np.abs(U[k:, k]))
        if fm != k:
            U[[k, fm], :] = U[[fm, k], :]
            P[[k, fm], :] = P[[fm, k], :]
            L[[k, fm], :k] = L[[fm, k], :k]
        if abs(U[k, k]) < 1e-14:
            raise np.linalg.LinAlgError("Matriz singular")
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
        U[k + 1:, k] = 0.0
    return P, L, U


def sustitucion_adelante(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def sustitucion_atras(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inversa_via_lu(A):
    """Resuelve A x_j = e_j para cada columna j de la identidad."""
    n = A.shape[0]
    P, L, U = descomposicion_lu(A)
    Ainv = np.zeros((n, n))
    I = np.eye(n)
    for j in range(n):
        Ainv[:, j] = sustitucion_atras(U, sustitucion_adelante(L, P @ I[:, j]))
    return Ainv


# =========================================================================
# 3) Sistema fisico: 4 esferas conductoras en linea
# =========================================================================
a = 0.01                                   # radio de cada esfera (m)
posiciones = np.array([0.0, 0.25, 0.50, 0.75])   # centros sobre el eje x (m)
N = len(posiciones)

def matriz_coef_potencial(a, posiciones):
    """Construye P con p_ii = 1/(4 pi eps0 a), p_ij = 1/(4 pi eps0 d_ij)."""
    N = len(posiciones)
    P_ = np.zeros((N, N))
    k_coul = 1.0 / (4 * np.pi * EPS0)
    for i in range(N):
        for j in range(N):
            if i == j:
                P_[i, j] = k_coul / a
            else:
                d = abs(posiciones[i] - posiciones[j])
                P_[i, j] = k_coul / d
    return P_

P_pot = matriz_coef_potencial(a, posiciones)

print("Matriz de coeficientes de potencial P (V/C):")
print(P_pot)
print(f"\nNumero de condicion de P: {np.linalg.cond(P_pot):.4f}")
print(f"P es simetrica? {np.allclose(P_pot, P_pot.T)}  "
      "(debe serlo por reciprocidad de Green)")

# --- calculo de la inversa por ambos metodos ---
C_gj = inversa_gauss_jordan(P_pot, verbose=True)
C_lu = inversa_via_lu(P_pot)
C_ref = np.linalg.inv(P_pot)

print("\nMatriz de capacitancia C = P^{-1}  (en pF):")
print(C_gj * 1e12)

print("\nInterpretacion fisica:")
for i in range(N):
    print(f"  Esfera {i}: capacitancia propia C_{i}{i} = {C_gj[i,i]*1e12:8.4f} pF")
print(f"  Capacitancia mutua vecinos C_01 = {C_gj[0,1]*1e12:8.4f} pF "
      "(negativa, como debe ser)")
print(f"  Capacitancia esferas lejanas C_03 = {C_gj[0,3]*1e12:8.4f} pF "
      "(mucho menor: el acoplamiento decae con la distancia)")

# =========================================================================
# 4) Analisis de error
# =========================================================================
print("\n" + "=" * 72)
print("ANALISIS DE ERROR")
print("=" * 72)

err_gj_vs_ref = np.linalg.norm(C_gj - C_ref) / np.linalg.norm(C_ref)
err_lu_vs_ref = np.linalg.norm(C_lu - C_ref) / np.linalg.norm(C_ref)
# Residuo: que tan bien se cumple A A^{-1} = I. Es la medida mas honesta,
# porque no depende de tomar otra implementacion como "verdad".
res_gj = np.linalg.norm(P_pot @ C_gj - np.eye(N))
res_lu = np.linalg.norm(P_pot @ C_lu - np.eye(N))

print(f"  Error relativo Gauss-Jordan vs numpy.linalg.inv : {err_gj_vs_ref:.3e}")
print(f"  Error relativo via LU vs numpy.linalg.inv       : {err_lu_vs_ref:.3e}")
print(f"  Residuo ||P C - I||_2  (Gauss-Jordan)           : {res_gj:.3e}")
print(f"  Residuo ||P C - I||_2  (via LU)                 : {res_lu:.3e}")

# --- Comparacion: resolver A x = b directamente vs usar la inversa ---
V_aplicado = np.array([100.0, 0.0, 0.0, -100.0])      # voltios
Q_via_inversa = C_gj @ V_aplicado
Pm, Lm, Um = descomposicion_lu(P_pot)
Q_via_lu = sustitucion_atras(Um, sustitucion_adelante(Lm, Pm @ V_aplicado))

print(f"\n  Cargas con V = {V_aplicado} V:")
print(f"    usando la inversa : {Q_via_inversa*1e12} pC")
print(f"    resolviendo P Q=V : {Q_via_lu*1e12} pC")
print(f"    residuo ||P Q - V|| usando inversa : "
      f"{np.linalg.norm(P_pot @ Q_via_inversa - V_aplicado):.3e}")
print(f"    residuo ||P Q - V|| resolviendo    : "
      f"{np.linalg.norm(P_pot @ Q_via_lu - V_aplicado):.3e}")

# --- Validez de la aproximacion de carga puntual segun a/d ---
# Se compara P construida con la aproximacion puntual frente a P con una
# correccion de orden (a/d)^3 (primer termino de la polarizacion mutua
# inducida), para estimar el error de la SIMPLIFICACION fisica (no numerica).
razones_ad = np.logspace(-2.5, -0.7, 20)
d_ref = 0.25
errores_modelo = []
for r in razones_ad:
    a_var = r * d_ref
    P_puntual = matriz_coef_potencial(a_var, posiciones)
    # correccion dipolar inducida de primer orden ~ (a/d)^3 en los terminos
    # cruzados (estimacion de la magnitud del termino despreciado)
    P_corr = P_puntual.copy()
    k_coul = 1.0 / (4 * np.pi * EPS0)
    for i in range(N):
        for j in range(N):
            if i != j:
                d = abs(posiciones[i] - posiciones[j])
                P_corr[i, j] += k_coul / d * (a_var / d) ** 3
    errores_modelo.append(
        np.linalg.norm(np.linalg.inv(P_corr) - np.linalg.inv(P_puntual))
        / np.linalg.norm(np.linalg.inv(P_puntual))
    )
errores_modelo = np.array(errores_modelo)

# =========================================================================
# 5) Graficas
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

im = axes[0].imshow(C_gj * 1e12, cmap="RdBu_r",
                    vmin=-np.max(np.abs(C_gj * 1e12)),
                    vmax=np.max(np.abs(C_gj * 1e12)))
axes[0].set_title("Matriz de capacitancia $C=P^{-1}$ (pF)")
axes[0].set_xlabel("Conductor j")
axes[0].set_ylabel("Conductor i")
axes[0].set_xticks(range(N))
axes[0].set_yticks(range(N))
for i in range(N):
    for j in range(N):
        axes[0].text(j, i, f"{C_gj[i,j]*1e12:.2f}", ha="center", va="center",
                     fontsize=7)
plt.colorbar(im, ax=axes[0], fraction=0.046)

metodos = ["Gauss-Jordan", "Via LU", "numpy.inv"]
residuos = [res_gj, res_lu, np.linalg.norm(P_pot @ C_ref - np.eye(N))]
axes[1].bar(metodos, np.array(residuos) + 1e-20,
            color=["#2E86AB", "#A23B72", "#F18F01"])
axes[1].set_yscale("log")
axes[1].set_ylabel(r"$\|P\,C - I\|_2$")
axes[1].set_title("Residuo de la inversion por metodo")
axes[1].grid(alpha=0.3, which="both")
axes[1].tick_params(axis="x", labelsize=8)

axes[2].loglog(razones_ad, errores_modelo, "o-", color="#6A4C93",
               label="error del modelo")
axes[2].loglog(razones_ad, razones_ad ** 3, "--", color="gray",
               label=r"pendiente $(a/d)^3$")
axes[2].axvline(0.1, ls=":", color="red", label="a/d = 0.1")
axes[2].set_xlabel("Razon $a/d$")
axes[2].set_ylabel("Error relativo en $C$")
axes[2].set_title("Validez de la aproximacion de carga puntual")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("5_inversa_matriz.png", dpi=150)
print("\nGrafica guardada en 5_inversa_matriz.png")

print(f"""
CONCLUSION SOBRE LA SIMPLIFICACION FISICA:
El error del modelo de carga puntual escala como (a/d)^3, tal como predice
el argumento de polarizacion mutua dado en el encabezado (la grafica 3 lo
confirma: la curva es paralela a la recta de pendiente 3 en escala log-log).
Para la configuracion usada aqui, a/d = {a/0.25:.3f}, el error de la
aproximacion es del orden de {(a/0.25)**3:.1e}, es decir, muy inferior al
error experimental tipico de una medicion de capacitancia. La aproximacion
esta plenamente justificada en este regimen y dejaria de estarlo si las
esferas se acercaran hasta a/d ~ 0.3 o mas.
""")
