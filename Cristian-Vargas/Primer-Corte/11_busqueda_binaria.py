r"""
=============================================================================
 TEMA: Busqueda binaria (metodo de biseccion)
 CURSO: Fisica Computacional II
=============================================================================

IDEA DEL METODO Y SU JUSTIFICACION
-----------------------------------------------------------------
Teorema del valor intermedio: si f es continua en [a, b] y f(a) f(b) < 0,
existe al menos una raiz de f en (a, b). La busqueda binaria explota esto:

   1. c = (a + b)/2.
   2. Si f(a) f(c) < 0 la raiz esta en [a, c]  -> b = c ; si no, en [c, b] -> a = c.
   3. Se repite hasta que el intervalo sea menor que la tolerancia.

Cada paso CONSERVA el cambio de signo, asi que la raiz queda siempre atrapada
(por eso el metodo no puede fallar si f es continua y el intervalo inicial
es valido) y el intervalo se reduce EXACTAMENTE a la mitad:

        (b_n - a_n) = (b_0 - a_0) / 2^n .

Tomando como estimacion el punto medio, |c_n - x*| <= (b_0 - a_0)/2^{n+1}.
Para garantizar una precision eps hacen falta

        N = ceil( log2( (b_0 - a_0) / eps ) )

iteraciones, numero que se conoce ANTES de empezar (no depende de f).
Es convergencia lineal con razon 1/2: se gana ~0.3 cifras decimales
(log10 2) por iteracion, mas lenta que Newton, pero totalmente robusta.

EJEMPLO FISICO: niveles de energia de un pozo de potencial cuadrado finito
-----------------------------------------------------------------
Un electron en un pozo de profundidad V0 y ancho w (V = 0 para |x| < a,
a = w/2; V = V0 fuera). Los estados ligados (0 < E < V0) se obtienen
empalmando psi y psi' en x = a. Con

    z = a k ,   k = sqrt(2 m E)/hbar ,    z0 = (a/hbar) sqrt(2 m V0) ,

las condiciones de empalme dan, para estados de paridad par e impar:

    par   :  z tan z       = sqrt(z0^2 - z^2)
    impar : -z cot z       = sqrt(z0^2 - z^2)

Son ecuaciones TRASCENDENTES: no tienen solucion algebraica, y hay que
resolverlas numericamente. La bisección es ideal porque cada estado ligado
tiene un intervalo natural: el n-esimo estado (n = 0, 1, 2, ...) tiene su
raiz en  n pi/2 < z < (n+1) pi/2  (recortado a z <= z0).

TRUCO PARA EVITAR LOS POLOS (importante): tan z y cot z divergen en los
extremos de esos intervalos. Un cambio de signo producido por un POLO (de
+inf a -inf) NO es una raiz, y la biseccion lo confundiria con una. Se
multiplican las ecuaciones por cos z (o sin z) para obtener funciones
continuas sin polos que tienen exactamente las mismas raices en el intervalo:

    par   : f(z) = z sin z - sqrt(z0^2 - z^2) cos z = 0
    impar : f(z) = z cos z + sqrt(z0^2 - z^2) sin z = 0

Se verifica que f cambia de signo en cada intervalo [n pi/2, (n+1) pi/2]
(evaluando en los extremos, donde sin o cos se anulan). Abajo se demuestra
tambien que la forma con tan SI falla si el intervalo contiene un polo.

Se bisecta directamente en la ENERGIA E (en eV), para que la tolerancia sea
fisica (1e-6 eV).
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=8, suppress=True)

# -----------------------------------------------------------------------
# Parametros fisicos
# -----------------------------------------------------------------------
HBAR2_2M = 0.0380998212        # hbar^2 / (2 m_e)  en eV nm^2
V0 = 20.0                      # eV
W = 1.0                        # ancho del pozo (nm)
A_HALF = W / 2                 # semiancho a (nm)
Z0 = A_HALF * np.sqrt(V0 / HBAR2_2M)


def z_de_E(E):
    return A_HALF * np.sqrt(E / HBAR2_2M)


def E_de_z(z):
    return HBAR2_2M * (z / A_HALF) ** 2


def f_par(E):
    z = z_de_E(E)
    return z * np.sin(z) - np.sqrt(max(Z0 ** 2 - z ** 2, 0.0)) * np.cos(z)


def f_impar(E):
    z = z_de_E(E)
    return z * np.cos(z) + np.sqrt(max(Z0 ** 2 - z ** 2, 0.0)) * np.sin(z)


def busqueda_binaria(f, a, b, tol=1e-6, max_iter=200):
    """Biseccion. Devuelve (raiz, historial de puntos medios, de anchos y de
    |f(c)|). Exige cambio de signo en [a, b]."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a) y f(b) tienen el mismo signo: [{a:.4f}, {b:.4f}] "
                         "no garantiza una raiz")
    medios, anchos, residuos = [], [], []
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        medios.append(c); anchos.append(b - a); residuos.append(abs(fc))
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        if (b - a) < tol:
            break
    c = 0.5 * (a + b)
    medios.append(c); anchos.append(b - a); residuos.append(abs(f(c)))
    return c, np.array(medios), np.array(anchos), np.array(residuos)


# =========================================================================
# 1) Todos los estados ligados
# =========================================================================
n_estados = int(np.floor(2 * Z0 / np.pi)) + 1
tol = 1e-6

print(f"z0 = {Z0:.5f}  ->  numero de estados ligados = floor(2 z0/pi)+1 = {n_estados}")
print(f"\n{'n':>2} {'paridad':>8} {'intervalo en E (eV)':>26} {'E (eV)':>14} {'iter':>5} "
      f"{'N teorico':>10} {'|f(E)|':>10}")

niveles, historias = [], []
for n in range(n_estados):
    z_lo = n * np.pi / 2
    z_hi = min((n + 1) * np.pi / 2, Z0)
    E_lo, E_hi = E_de_z(z_lo), E_de_z(z_hi)
    f = f_par if n % 2 == 0 else f_impar
    # el extremo inferior puede ser exactamente E=0 (n=0); se evita E=0 puro
    E_lo = max(E_lo, 1e-9)
    E_raiz, medios, anchos, resid = busqueda_binaria(f, E_lo, E_hi, tol=tol)
    N_teo = int(np.ceil(np.log2((E_hi - E_lo) / tol)))
    niveles.append(E_raiz)
    historias.append((medios, anchos, resid, E_lo, E_hi))
    print(f"{n:2d} {'par' if n % 2 == 0 else 'impar':>8} "
          f"[{E_lo:9.4f}, {E_hi:9.4f}] {E_raiz:14.8f} {len(anchos)-1:5d} "
          f"{N_teo:10d} {resid[-1]:10.2e}")

niveles = np.array(niveles)

# =========================================================================
# 2) Verificacion independiente (brentq de scipy, si esta disponible)
# =========================================================================
print("\nVerificacion con scipy.optimize.brentq (solo como referencia):")
try:
    from scipy.optimize import brentq
    ref = []
    for n in range(n_estados):
        z_lo = n * np.pi / 2
        z_hi = min((n + 1) * np.pi / 2, Z0)
        f = f_par if n % 2 == 0 else f_impar
        ref.append(brentq(f, max(E_de_z(z_lo), 1e-9), E_de_z(z_hi), xtol=1e-14))
    ref = np.array(ref)
    print(f"   max |E_biseccion - E_brentq| = {np.max(np.abs(niveles - ref)):.2e} eV "
          f"(tolerancia pedida {tol:.0e} eV)")
except ImportError:
    ref = None
    print("   scipy no disponible; se omite.")

# --- comparacion con el pozo infinito del mismo ancho ---
n_inf = np.arange(1, n_estados + 1)
E_inf = HBAR2_2M * (n_inf * np.pi / W) ** 2
print("\n   n    E finito (eV)   E pozo infinito (eV)   diferencia relativa")
for i in range(n_estados):
    print(f"  {i+1:2d}   {niveles[i]:12.5f}   {E_inf[i]:18.5f}   "
          f"{(E_inf[i]-niveles[i])/E_inf[i]:14.3%}")
print("(el pozo finito tiene niveles mas BAJOS: la funcion de onda se filtra a "
      "la region prohibida y 've' un pozo efectivamente mas ancho)")

# =========================================================================
# 3) Anti-ejemplo: la forma con tan tiene un polo en el intervalo
# =========================================================================
print("\n" + "=" * 70)
print("ANTI-EJEMPLO: bisecar la forma con tan sobre un intervalo con polo")
print("=" * 70)
g_tan = lambda E: z_de_E(E) * np.tan(z_de_E(E)) - np.sqrt(max(Z0 ** 2 - z_de_E(E) ** 2, 0))
E_a, E_b = E_de_z(1.5), E_de_z(1.7)         # contiene SOLO el polo z = pi/2 (la raiz esta en z~1.44)
E_falsa, _, _, _ = busqueda_binaria(g_tan, E_a, E_b, tol=1e-9)
print(f"   cambio de signo en [{E_a:.3f}, {E_b:.3f}] eV (z de 1.5 a 1.7: contiene el polo z=pi/2 y ninguna raiz)")
print(f"   la biseccion converge a E = {E_falsa:.6f} eV (z = {z_de_E(E_falsa):.6f}; pi/2 = {np.pi/2:.6f})")
print(f"   |g(E)| = {abs(g_tan(E_falsa)):.2e}  (enorme: NO es una raiz)")
print(f"   con la forma sin polos, |f(E)| en la raiz verdadera = {historias[0][2][-1]:.2e}")

# =========================================================================
# 4) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) solucion grafica
zz = np.linspace(0.01, Z0, 4000)
s = np.sqrt(np.maximum(Z0 ** 2 - zz ** 2, 0))
with np.errstate(all="ignore"):
    par = zz * np.tan(zz)
    imp = -zz / np.tan(zz)
par[np.abs(par) > 40] = np.nan
imp[np.abs(imp) > 40] = np.nan
axes[0, 0].plot(zz, par, color="#2E86AB", label=r"$z\tan z$ (par)")
axes[0, 0].plot(zz, imp, color="#C1121F", label=r"$-z\cot z$ (impar)")
axes[0, 0].plot(zz, s, "k", lw=2, label=r"$\sqrt{z_0^2-z^2}$")
for E_n in niveles:
    axes[0, 0].axvline(z_de_E(E_n), color="gray", ls=":", lw=0.8)
axes[0, 0].set_ylim(0, 16)
axes[0, 0].set_xlabel("z")
axes[0, 0].set_title(f"Solucion grafica: {n_estados} estados ligados ($z_0$={Z0:.2f})")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# (b) diagrama de niveles
axes[0, 1].hlines(0, -0.5 * W, 0.5 * W, color="k", lw=3)
axes[0, 1].vlines([-0.5 * W, 0.5 * W], 0, V0, color="k", lw=3)
axes[0, 1].hlines(V0, -1.0, -0.5 * W, color="k", lw=3)
axes[0, 1].hlines(V0, 0.5 * W, 1.0, color="k", lw=3)
for i, E_n in enumerate(niveles):
    axes[0, 1].hlines(E_n, -0.5 * W, 0.5 * W, color="#2E86AB")
    axes[0, 1].text(0.52 * W, E_n, f"n={i+1}: {E_n:.3f} eV", fontsize=8, va="center")
    axes[0, 1].hlines(E_inf[i], -0.5 * W, 0.5 * W, color="#C1121F", ls=":", lw=0.9)
axes[0, 1].plot([], [], color="#2E86AB", label="pozo finito (biseccion)")
axes[0, 1].plot([], [], color="#C1121F", ls=":", label="pozo infinito")
axes[0, 1].set_xlim(-1.0, 1.9)
axes[0, 1].set_ylim(-1, V0 + 2)
axes[0, 1].set_xlabel("x (nm)")
axes[0, 1].set_ylabel("Energia (eV)")
axes[0, 1].set_title("Niveles de energia del electron en el pozo")
axes[0, 1].legend(fontsize=8, loc="upper left")
axes[0, 1].grid(alpha=0.3)

# (c) ancho del intervalo vs iteracion
for i in [0, 1, 2]:
    _, anchos, _, E_lo, E_hi = historias[i]
    k = np.arange(len(anchos))
    axes[1, 0].semilogy(k, anchos, "o-", ms=3, label=f"estado n={i+1}")
axes[1, 0].semilogy(np.arange(30), (historias[0][4] - historias[0][3]) / 2.0 ** np.arange(30),
                    "k:", label=r"$(b_0-a_0)/2^k$")
axes[1, 0].axhline(tol, color="gray", ls="--", lw=0.8)
axes[1, 0].set_xlabel("Iteracion k")
axes[1, 0].set_ylabel("Ancho del intervalo (eV)")
axes[1, 0].set_title("El intervalo se reduce exactamente a la mitad")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, which="both")

# (d) error del punto medio y residuo
medios, anchos, resid, E_lo, E_hi = historias[0]
err = np.abs(medios - niveles[0])
k = np.arange(len(medios))
axes[1, 1].semilogy(k, err + 1e-16, "o-", ms=3, label=r"$|c_k-E^*|$")
axes[1, 1].semilogy(k, anchos / 2, "k:", label=r"cota $(b_k-a_k)/2$")
axes[1, 1].semilogy(k, resid + 1e-16, "s-", ms=3, label=r"$|f(c_k)|$ (residuo)")
axes[1, 1].set_xlabel("Iteracion k")
axes[1, 1].set_title("Estado base: error real, cota teorica y residuo")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("11_busqueda_binaria.png", dpi=150)
print("\nGrafica guardada en 11_busqueda_binaria.png")