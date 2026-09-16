r"""
=============================================================================
 TEMA: Matrices tridiagonales y de banda (algoritmo de Thomas)
 CURSO: Fisica Computacional II
=============================================================================
EJEMPLO FISICO: conduccion de calor estacionaria en una barra con fuente
---------------------------------------------------------------------------
Barra de longitud L, conductividad termica k, con generacion de calor
interna q(x) (por ejemplo, calentamiento ohmico) y temperaturas fijas en
los extremos. La ecuacion de calor en estado estacionario es:

        k d2T/dx2 + q(x) = 0 ,     T(0) = T_izq ,   T(L) = T_der

Se elige q(x) = q0 sin(pi x / L), que admite SOLUCION ANALITICA EXACTA:

        T(x) = T_izq + (T_der - T_izq) x/L + (q0 L^2)/(k pi^2) sin(pi x/L)

(se verifica derivando dos veces: d2/dx2 del termino lineal es cero y
 d2/dx2 [sin(pi x/L)] = -(pi/L)^2 sin(pi x/L), lo que cancela exactamente
 q(x)/k). Tener la solucion exacta permite medir el error REAL del metodo
contra la fisica, no solo contra otra rutina numerica.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)


# =========================================================================
# 1) Algoritmo de Thomas
# =========================================================================
def thomas(a, b, c, d, verificar_dominancia=True):
    """
    Resuelve un sistema tridiagonal en O(n).

    Parametros
    ----------
    a : subdiagonal, longitud n (a[0] no se usa)
    b : diagonal principal, longitud n
    c : superdiagonal, longitud n (c[n-1] no se usa)
    d : lado derecho, longitud n

    Los arreglos NO se modifican (se trabaja sobre copias).
    """
    n = len(b)
    a, b, c, d = (np.asarray(v, dtype=float).copy() for v in (a, b, c, d))

    if verificar_dominancia:
        fuera_de_diagonal = np.abs(a) + np.abs(c)
        fuera_de_diagonal[0] = abs(c[0])
        fuera_de_diagonal[-1] = abs(a[-1])
        if not np.all(np.abs(b) >= fuera_de_diagonal - 1e-12):
            raise ValueError(
                "La matriz NO es diagonalmente dominante: Thomas sin pivoteo "
                "puede ser inestable aqui. Use eliminacion con pivoteo."
            )

    # --- barrido hacia adelante ---
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # --- barrido hacia atras ---
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


# =========================================================================
# 2) Construccion del problema fisico discretizado
# =========================================================================
L_barra = 1.0          # longitud de la barra (m)
k_cond = 50.0          # conductividad termica (W/m K), tipico de un acero
q0 = 5.0e4             # amplitud de la fuente de calor (W/m^3)
T_izq, T_der = 300.0, 400.0     # temperaturas fijas en los extremos (K)


def solucion_analitica(x):
    """T(x) exacta, derivada en el encabezado."""
    return (T_izq + (T_der - T_izq) * x / L_barra
            + q0 * L_barra**2 / (k_cond * np.pi**2) * np.sin(np.pi * x / L_barra))


def construir_sistema(n_int):
    """
    Discretiza k T'' + q = 0 en n_int nodos INTERIORES.
    Malla: x_i = (i+1) h, con h = L/(n_int+1); los extremos son frontera.

    En cada nodo interior:
        k (T_{i-1} - 2T_i + T_{i+1})/h^2 = -q_i
    Multiplicando por -h^2/k:
        -T_{i-1} + 2 T_i - T_{i+1} = h^2 q_i / k
    Las condiciones de frontera (T conocida en los extremos) se pasan al
    lado derecho: en la primera fila T_{i-1}=T_izq es un dato conocido, no
    una incognita, asi que su termino se suma a d[0]. Analogo en la ultima.
    """
    h = L_barra / (n_int + 1)
    x = np.linspace(h, L_barra - h, n_int)

    a = np.full(n_int, -1.0)      # subdiagonal
    b = np.full(n_int, 2.0)       # diagonal
    c = np.full(n_int, -1.0)      # superdiagonal
    a[0] = 0.0
    c[-1] = 0.0

    q = q0 * np.sin(np.pi * x / L_barra)
    d = h**2 * q / k_cond
    d[0] += T_izq                 # condicion de frontera izquierda
    d[-1] += T_der                # condicion de frontera derecha

    return a, b, c, d, x, h


# --- resolucion con una malla de tamano moderado ---
n_int = 49
a, b, c, d, x, h = construir_sistema(n_int)

print(f"Malla: {n_int} nodos interiores, h = {h:.5f} m")
print(f"Dominancia diagonal: |b|=2, |a|+|c|=2 en filas interiores "
      f"(se cumple la igualdad; en las filas de frontera |b| > |a|+|c|)")

T_num = thomas(a, b, c, d)
T_exa = solucion_analitica(x)

error_max = np.max(np.abs(T_num - T_exa))
error_rms = np.sqrt(np.mean((T_num - T_exa) ** 2))
print(f"\nError maximo   |T_num - T_exacta| = {error_max:.6e} K")
print(f"Error RMS                          = {error_rms:.6e} K")
print(f"Temperatura maxima calculada       = {T_num.max():.4f} K")

# --- verificacion contra la matriz densa (mismo sistema, otro metodo) ---
A_densa = (np.diag(b) + np.diag(c[:-1], 1) + np.diag(a[1:], -1))
T_densa = np.linalg.solve(A_densa, d)
print(f"Diferencia Thomas vs numpy (matriz densa) = "
      f"{np.max(np.abs(T_num - T_densa)):.3e} K")

# =========================================================================
# 3) Convergencia: verificacion de que el error escala como O(h^2)
# =========================================================================
print("\n" + "=" * 72)
print("ESTUDIO DE CONVERGENCIA (debe dar pendiente ~2)")
print("=" * 72)

ns = np.array([9, 19, 39, 79, 159, 319, 639, 1279])
hs, errores = [], []
for n_ in ns:
    a_, b_, c_, d_, x_, h_ = construir_sistema(n_)
    T_ = thomas(a_, b_, c_, d_)
    hs.append(h_)
    errores.append(np.max(np.abs(T_ - solucion_analitica(x_))))
hs, errores = np.array(hs), np.array(errores)

# pendiente en escala log-log = orden de convergencia observado
orden = np.polyfit(np.log(hs), np.log(errores), 1)[0]
print(f"  Orden de convergencia medido: p = {orden:.4f}  (teorico: 2)")
for h_, e_ in zip(hs, errores):
    print(f"    h = {h_:.6f}  ->  error_max = {e_:.4e} K")

# =========================================================================
# 4) Costo computacional: Thomas O(n) vs Gauss denso O(n^3)
# =========================================================================
print("\n" + "=" * 72)
print("COSTO: Thomas O(n) vs solver denso O(n^3)")
print("=" * 72)

tamanos = [50, 100, 200, 400, 800]
t_thomas, t_denso = [], []
mem_thomas, mem_denso = [], []

for n_ in tamanos:
    a_, b_, c_, d_, _, _ = construir_sistema(n_)

    t0 = time.perf_counter()
    for _ in range(5):
        thomas(a_, b_, c_, d_, verificar_dominancia=False)
    t_thomas.append((time.perf_counter() - t0) / 5)

    A_ = np.diag(b_) + np.diag(c_[:-1], 1) + np.diag(a_[1:], -1)
    t0 = time.perf_counter()
    for _ in range(5):
        np.linalg.solve(A_, d_)
    t_denso.append((time.perf_counter() - t0) / 5)

    mem_thomas.append(4 * n_ * 8 / 1024)          # 4 vectores, float64, en KB
    mem_denso.append(n_ * n_ * 8 / 1024)          # matriz completa, en KB

    print(f"  n={n_:5d}: Thomas = {t_thomas[-1]*1e6:9.1f} us | "
          f"denso = {t_denso[-1]*1e6:9.1f} us | "
          f"memoria {mem_thomas[-1]:.1f} KB vs {mem_denso[-1]:.1f} KB")

print("""
NOTA HONESTA SOBRE LA COMPARACION DE TIEMPOS: el algoritmo de Thomas aqui
esta escrito en Python puro (bucles interpretados), mientras que
numpy.linalg.solve llama a LAPACK compilado en Fortran/C. Por eso, para n
pequeno, el "denso" puede incluso ganar pese a su peor complejidad: se esta
comparando O(n) interpretado contra O(n^3) compilado. Lo que importa es la
PENDIENTE de cada curva (como crece el tiempo con n), no el valor absoluto.
La ventaja real y no discutible de Thomas es la MEMORIA: crece como O(n)
frente a O(n^2), y esa comparacion no depende del lenguaje.
""")

# =========================================================================
# 5) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) Perfil de temperatura
x_fino = np.linspace(0, L_barra, 400)
axes[0, 0].plot(x_fino, solucion_analitica(x_fino), "-", color="#A23B72",
                lw=2, label="Solucion analitica")
axes[0, 0].plot(x, T_num, "o", color="#2E86AB", ms=4,
                label=f"Thomas (n={n_int})")
axes[0, 0].plot([0, L_barra], [T_izq, T_der], "ks", ms=7,
                label="Condiciones de frontera")
axes[0, 0].set_xlabel("Posicion x (m)")
axes[0, 0].set_ylabel("Temperatura T (K)")
axes[0, 0].set_title("Perfil de temperatura en la barra con fuente de calor")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# (b) Error nodo a nodo
axes[0, 1].plot(x, np.abs(T_num - T_exa), "o-", color="#F18F01", ms=3)
axes[0, 1].set_xlabel("Posicion x (m)")
axes[0, 1].set_ylabel(r"$|T_{num} - T_{exacta}|$  (K)")
axes[0, 1].set_title(f"Error de discretizacion (h = {h:.4f} m)")
axes[0, 1].grid(alpha=0.3)

# (c) Convergencia O(h^2)
axes[1, 0].loglog(hs, errores, "o-", color="#2E86AB", label="error medido")
axes[1, 0].loglog(hs, errores[0] * (hs / hs[0]) ** 2, "--", color="gray",
                  label=r"referencia $O(h^2)$")
axes[1, 0].set_xlabel("Paso de malla h (m)")
axes[1, 0].set_ylabel("Error maximo (K)")
axes[1, 0].set_title(f"Convergencia: pendiente medida p = {orden:.3f}")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, which="both")

# (d) Memoria requerida
axes[1, 1].loglog(tamanos, mem_denso, "o-", color="#C1121F",
                  label=r"Matriz densa $O(n^2)$")
axes[1, 1].loglog(tamanos, mem_thomas, "s-", color="#2E86AB",
                  label=r"Thomas (3 vectores) $O(n)$")
axes[1, 1].set_xlabel("Tamano del sistema n")
axes[1, 1].set_ylabel("Memoria (KB)")
axes[1, 1].set_title("Memoria: la ventaja estructural de Thomas")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("6_tridiagonales_banda.png", dpi=150)
print("Grafica guardada en 6_tridiagonales_banda.png")

# =========================================================================
# 6) Extension a matrices de BANDA (ancho de banda mayor que 1)
# =========================================================================
print("\n" + "=" * 72)
print("EXTENSION: matrices de banda")
print("=" * 72)
print("""
Una matriz de banda con semiancho p tiene entradas no nulas solo si
|i - j| <= p (la tridiagonal es el caso p = 1). Surgen, por ejemplo, al
usar diferencias finitas de MAYOR ORDEN: la formula de 5 puntos para T''

    T'' = (-T_{i-2} + 16T_{i-1} - 30T_i + 16T_{i+1} - T_{i+2}) / (12 h^2)

tiene error O(h^4) en lugar de O(h^2), pero conecta cada nodo con CUATRO
vecinos, dando p = 2 (pentadiagonal). El mismo argumento del algoritmo de
Thomas se generaliza: la eliminacion solo necesita tocar las p filas
siguientes y las p columnas siguientes, con costo O(n p^2) en vez de
O(n^3). Se comprueba abajo que la version pentadiagonal efectivamente
converge mas rapido.
""")

def construir_pentadiagonal(n_int):
    """Discretizacion de 5 puntos (O(h^4)) para el mismo problema fisico.
    En los nodos adyacentes a la frontera no hay suficientes vecinos para
    la formula de 5 puntos, asi que ahi se usa la de 3 puntos: por eso el
    orden global observado queda entre 2 y 4, no exactamente 4."""
    h = L_barra / (n_int + 1)
    x = np.linspace(h, L_barra - h, n_int)
    A = np.zeros((n_int, n_int))
    q = q0 * np.sin(np.pi * x / L_barra)
    d = h**2 * q / k_cond

    for i in range(n_int):
        if 2 <= i <= n_int - 3:
            A[i, i-2], A[i, i-1], A[i, i] = 1/12, -16/12, 30/12
            A[i, i+1], A[i, i+2] = -16/12, 1/12
        else:
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < n_int - 1:
                A[i, i+1] = -1.0
    d[0] += T_izq
    d[-1] += T_der
    return A, d, x

err_penta = []
for n_ in ns[:6]:
    A_, d_, x_ = construir_pentadiagonal(n_)
    T_ = np.linalg.solve(A_, d_)
    err_penta.append(np.max(np.abs(T_ - solucion_analitica(x_))))
orden_penta = np.polyfit(np.log(hs[:6]), np.log(err_penta), 1)[0]
print(f"  Orden de convergencia tridiagonal (3 puntos):   p = {orden:.3f}")
print(f"  Orden de convergencia pentadiagonal (5 puntos): p = {orden_penta:.3f}")
