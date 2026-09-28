"""
Actividad FC2 - Equilibrio y puntos prometidos en un sistema de cargas
electricas no lineal.

Sistema:
    Q1 = +4 uC en (0,0)      Q2 = +1 uC en (2,0)      q = +1 uC (movil)
    Potencial externo no lineal  Vext(x,y) = A (x^4 + y^4),  A = 0.5 V/m^4

Fuerza neta (dividida por q, asi q no interviene en el equilibrio):
    f1(x,y) = k [ Q1 x / r1^3 + Q2 (x-2) / r2^3 ] - 4 A x^3
    f2(x,y) = k [ Q1 y / r1^3 + Q2  y    / r2^3 ] - 4 A y^3
    con r1 = sqrt(x^2+y^2),  r2 = sqrt((x-2)^2+y^2)

Parte A: relajacion (punto fijo) en 2 variables, desde (1.0, 0.5), tol 1e-6.
Parte B: biseccion en el eje x sobre U(x) - U0 = 0, intervalo [0.1, 1.9],
         precision 1e-5.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Las 4 figuras se guardan en la carpeta Figure/ (la que usa el .tex)
os.makedirs("Figure", exist_ok=True)

# ---------------------------------------------------------------
# 0. Constantes fisicas
# ---------------------------------------------------------------
k = 8.99e9        # constante de Coulomb [N m^2 / C^2]
Q1, Q2 = 4e-6, 1e-6
q = 1e-6
A = 0.5           # [V/m^4]


def radios(x, y):
    return np.hypot(x, y), np.hypot(x - 2.0, y)


def f_sistema(x, y):
    """Fuerza neta por unidad de carga: (f1, f2). Equilibrio: f1 = f2 = 0."""
    r1, r2 = radios(x, y)
    f1 = k * (Q1 * x / r1**3 + Q2 * (x - 2.0) / r2**3) - 4 * A * x**3
    f2 = k * (Q1 * y / r1**3 + Q2 * y / r2**3) - 4 * A * y**3
    return f1, f2


def U_total(x, y):
    """Energia potencial total de q: U = q [kQ1/r1 + kQ2/r2 + A(x^4+y^4)]."""
    r1, r2 = radios(x, y)
    return q * (k * Q1 / r1 + k * Q2 / r2 + A * (x**4 + y**4))


# ---------------------------------------------------------------
# PARTE A - Relajacion multivariable
# ---------------------------------------------------------------
# Formulacion 1 (raiz cubica): como el termino no lineal es 4A x^3 (un cubo
# puro), se despeja directamente:  x = cbrt( (k/4A) [ ... ] )
def g_raiz_cubica(x, y):
    r1, r2 = radios(x, y)
    c = k / (4 * A)
    g1 = np.cbrt(c * (Q1 * x / r1**3 + Q2 * (x - 2.0) / r2**3))
    g2 = np.cbrt(c * (Q1 * y / r1**3 + Q2 * y / r2**3))
    return g1, g2


# Formulacion 2 (lineal): x = x + a f1,  y = y - a f2.
# Es una reorganizacion valida (en el punto fijo f1 = f2 = 0). El signo de
# cada componente se elige para que |dg/dx| < 1 cerca del equilibrio.
def crear_g_lineal(alfa):
    def g(x, y):
        f1, f2 = f_sistema(x, y)
        return x + alfa * f1, y - alfa * f2
    return g


def relajacion_2var(g, p0, omega=1.0, tol=1e-6, max_iter=5000):
    """
    Relajacion en 2 variables:
        x_(n+1) = (1-w) x_n + w g1(x_n, y_n)
        y_(n+1) = (1-w) y_n + w g2(x_n, y_n)
    Criterio de paro: max(|dx|, |dy|) < tol.
    Devuelve trayectoria (N,2), errores y bandera de convergencia.
    """
    x, y = p0
    tray = [(x, y)]
    errores = []
    for _ in range(max_iter):
        gx, gy = g(x, y)
        xn = (1 - omega) * x + omega * gx
        yn = (1 - omega) * y + omega * gy
        err = max(abs(xn - x), abs(yn - y))
        errores.append(err)
        x, y = xn, yn
        tray.append((x, y))
        if not np.isfinite(err) or abs(x) > 1e6 or abs(y) > 1e6:
            return np.array(tray), errores, False
        if err < tol:
            return np.array(tray), errores, True
    return np.array(tray), errores, False


def clasificar(x, y, h=1e-4):
    """Clasifica el equilibrio con los valores propios del hessiano de U."""
    def Ug(a, b):
        return U_total(a, b)
    Uxx = (Ug(x + h, y) - 2 * Ug(x, y) + Ug(x - h, y)) / h**2
    Uyy = (Ug(x, y + h) - 2 * Ug(x, y) + Ug(x, y - h)) / h**2
    Uxy = (Ug(x + h, y + h) - Ug(x + h, y - h)
           - Ug(x - h, y + h) + Ug(x - h, y - h)) / (4 * h**2)
    ev = np.linalg.eigvalsh(np.array([[Uxx, Uxy], [Uxy, Uyy]]))
    if ev.min() > 0:
        return "minimo (estable)"
    if ev.max() < 0:
        return "maximo"
    return "punto silla (inestable)"


p0 = (1.0, 0.5)
tol_A = 1e-6

print("=" * 66)
print("PARTE A - Relajacion en 2 variables, inicio (1.0, 0.5), tol 1e-6")
print("=" * 66)

resultados_A = {}
for nombre, g, omega in [
    ("Raiz cubica, w=1.0", g_raiz_cubica, 1.0),
    ("Raiz cubica, w=0.5", g_raiz_cubica, 0.5),
    ("Raiz cubica, w=0.2", g_raiz_cubica, 0.2),
    ("Lineal a=1e-5, w=1", crear_g_lineal(1e-5), 1.0),
    ("Lineal a=3e-5, w=1", crear_g_lineal(3e-5), 1.0),
]:
    tray, err, ok = relajacion_2var(g, p0, omega, tol_A)
    resultados_A[nombre] = (tray, err, ok)
    if ok:
        xf, yf = tray[-1]
        print(f"{nombre:22s} converge en {len(err):4d} iter -> "
              f"({xf:.6f}, {yf:.6f})  [{clasificar(xf, abs(yf))}]")
    else:
        print(f"{nombre:22s} NO converge (diverge / max_iter)")

# ---------------------------------------------------------------
# PARTE B - Biseccion sobre el eje x
# ---------------------------------------------------------------
def U_eje(x):
    return U_total(x, 0.0)


def biseccion(g, a, b, tol=1e-5, max_iter=200):
    """Biseccion. Exige g(a)*g(b) < 0; si no, no hay raiz garantizada."""
    if g(a) * g(b) > 0:
        raise ValueError("g(a) y g(b) tienen el mismo signo: no hay "
                         "cambio de signo en el intervalo")
    anchos = []
    for it in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        anchos.append(b - a)
        if g(a) * g(c) < 0:
            b = c
        else:
            a = c
        if (b - a) < tol:
            break
    return 0.5 * (a + b), anchos, it


print()
print("=" * 66)
print("PARTE B - Biseccion sobre el eje x, intervalo [0.1, 1.9], tol 1e-5")
print("=" * 66)

xs = np.linspace(0.1, 1.9, 20001)
Umin = U_eje(xs).min()
print(f"U(x) en [0.1, 1.9]: minimo = {Umin*1e3:.2f} mJ, "
      f"U(0.1) = {U_eje(0.1)*1e3:.2f} mJ, U(1.9) = {U_eje(1.9)*1e3:.2f} mJ")

# Caso 1: el U0 del enunciado
U0 = 15e-3
try:
    biseccion(lambda x: U_eje(x) - U0, 0.1, 1.9)
except ValueError as e:
    print(f"\nU0 = 15 mJ  ->  {e}")
    print(f"  U(x) >= {Umin*1e3:.2f} mJ en todo el intervalo: el objetivo de "
          f"15 mJ NO es alcanzable.")

# Caso 2: U0 alcanzable (entre U(1.9)=108.8 mJ y U(0.1)=364.3 mJ)
U0 = 150e-3
x_bis, anchos_B, n_B = biseccion(lambda x: U_eje(x) - U0, 0.1, 1.9)
print(f"\nU0 = 150 mJ ->  x* = {x_bis:.6f} m en {n_B} iteraciones "
      f"(cota teorica: {int(np.ceil(np.log2(1.8/1e-5)))})")
print(f"  Verificacion: U(x*) = {U_eje(x_bis)*1e3:.4f} mJ")

# ---------------------------------------------------------------
# Comparacion de convergencia A vs B
# ---------------------------------------------------------------
print("\nComparacion: iteraciones necesarias")
print(f"  Biseccion (B, tol 1e-5)      : {n_B}")
for nombre, (tray, err, ok) in resultados_A.items():
    if ok:
        print(f"  Relajacion (A) {nombre:20s}: {len(err)}")

# ---------------------------------------------------------------
# GRAFICAS
# ---------------------------------------------------------------
equilibrios = [(4/3, 0.0), (0.3802, 7.3939), (5.9465, 6.2531)]
from scipy.optimize import fsolve
eq_ref = []
for guess in [(1.3, 0.0), (0.4, 7.4), (5.9, 6.2)]:
    s = fsolve(lambda p: f_sistema(*p), guess)
    eq_ref.append((s[0], s[1]))

# --- Figura 1: trayectorias de convergencia en el plano (x,y)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a_, nombre in zip(ax, ["Raiz cubica, w=0.5", "Lineal a=1e-5, w=1"]):
    tray, err, ok = resultados_A[nombre]
    a_.plot(tray[:, 0], tray[:, 1], 'o-', ms=3, lw=1, label="trayectoria")
    a_.plot(*tray[0], 'gs', ms=9, label="inicio (1.0, 0.5)")
    a_.plot(*tray[-1], 'r*', ms=14, label=f"final ({tray[-1,0]:.4f}, {tray[-1,1]:.4f})")
    a_.plot([0, 2], [0, 0], 'k^', ms=9, label="Q1, Q2")
    a_.set_title(f"{nombre}: {len(err)} iteraciones")
    a_.set_xlabel("x [m]"); a_.set_ylabel("y [m]")
    a_.grid(alpha=0.3); a_.legend(fontsize=8)
plt.tight_layout()
plt.savefig("Figure/trayectoria_relajacion.png", dpi=150)

# --- Figura 2: mapa de potencial con equilibrios superpuestos
X, Y = np.meshgrid(np.linspace(-3, 9, 700), np.linspace(-9.5, 9.5, 700))
Z = U_total(X, Y) * 1e3   # mJ
Z = np.clip(Z, 1.0, 300.0)
fig, ax = plt.subplots(figsize=(8, 7))
cs = ax.contourf(X, Y, Z, levels=np.logspace(np.log10(1.0), np.log10(300.0), 60),
                 norm=LogNorm(), cmap="viridis")
cb = fig.colorbar(cs, label="U(x,y) [mJ] (escala log, recortado a 1-300 mJ)")
cb.set_ticks([1, 3, 10, 30, 100, 300]); cb.set_ticklabels(["1", "3", "10", "30", "100", "300"])
ax.plot([0, 2], [0, 0], 'w^', ms=10, label="Q1, Q2")
for (ex, ey) in eq_ref:
    for sgn in ([1, -1] if abs(ey) > 1e-6 else [1]):
        tipo = clasificar(ex, sgn * ey)
        marc = 'r*' if "minimo" in tipo else 'ro'
        ax.plot(ex, sgn * ey, marc, ms=11, mec='w')
ax.plot([], [], 'r*', ms=11, mec='w', label="equilibrio: minimo")
ax.plot([], [], 'ro', ms=8, mec='w', label="equilibrio: silla")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Mapa de energia potencial y puntos de equilibrio")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("Figure/mapa_potencial.png", dpi=150)

# --- Figura 3: convergencia A vs B
plt.figure(figsize=(7.5, 5))
for nombre, (tray, err, ok) in resultados_A.items():
    if ok:
        plt.plot(err, label=f"A: {nombre} ({len(err)} it)")
plt.plot(anchos_B, 's-', ms=3, label=f"B: biseccion ({n_B} it)")
plt.yscale('log'); plt.xlabel("Iteracion"); plt.ylabel("Error / ancho (log)")
plt.title("Convergencia: relajacion (A) vs biseccion (B)")
plt.grid(alpha=0.3, which='both'); plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("Figure/convergencia_A_vs_B.png", dpi=150)

# --- Figura 4: comparacion de formulaciones g (trayectorias sobre el mapa
#     de energia y error por iteracion)
fig, (axm, axe) = plt.subplots(1, 2, figsize=(14, 6))
X2, Y2 = np.meshgrid(np.linspace(-3, 9, 500), np.linspace(-9.5, 9.5, 500))
Z2 = np.clip(U_total(X2, Y2) * 1e3, 1.0, 300.0)
axm.contourf(X2, Y2, Z2, levels=np.logspace(0, np.log10(300.0), 50),
             norm=LogNorm(), cmap="Greys", alpha=0.6)
estilos = [("Raiz cubica, w=0.5", "tab:blue", "-", "Raiz cubica ($\\omega$=0.5)"),
           ("Lineal a=1e-5, w=1", "tab:green", "-", "Lineal $\\alpha$=1e-5"),
           ("Lineal a=3e-5, w=1", "tab:red", ":", "Lineal $\\alpha$=3e-5")]
for clave, color, ls, etiqueta in estilos:
    tray_, err_, ok_ = resultados_A[clave]
    axm.plot(tray_[:, 0], tray_[:, 1], ls, color=color, lw=1.2, marker='o',
             ms=2, label=f"{etiqueta}: {len(err_)} it")
    axm.plot(*tray_[-1], '*', color=color, ms=14, mec='k')
    axe.plot(err_, ls, color=color, label=f"{etiqueta}: {len(err_)} it")
axm.plot(*p0, 'ks', ms=8, label="inicio (1.0, 0.5)")
axm.plot([0, 2], [0, 0], 'k^', ms=8, label="Q1, Q2")
axm.set_xlim(-3, 9); axm.set_ylim(-9.5, 9.5)
axm.set_xlabel("x [m]"); axm.set_ylabel("y [m]")
axm.set_title("Trayectorias segun la formulacion g")
axm.legend(fontsize=8, loc="upper left")
axe.set_yscale('log'); axe.set_xscale('log')
axe.axhline(tol_A, color='k', lw=0.8, ls='--')
axe.set_xlabel("Iteracion"); axe.set_ylabel("Error max(|dx|,|dy|)")
axe.set_title("Error por iteracion (linea discontinua: tol 1e-6)")
axe.grid(alpha=0.3, which='both'); axe.legend(fontsize=8)
plt.tight_layout()
plt.savefig("Figure/comparacion_g.png", dpi=150)

# Mostrar las cuatro figuras en pantalla (sin esto solo se guardan como PNG)
plt.show()

print("\nEquilibrios (fsolve) y su clasificacion:")
for (ex, ey) in eq_ref:
    print(f"  ({ex:.5f}, {ey:.5f})  ->  {clasificar(ex, abs(ey) if ey else 0.0)}")