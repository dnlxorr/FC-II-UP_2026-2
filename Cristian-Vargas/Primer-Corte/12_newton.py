r"""
=============================================================================
 TEMA: Metodo de Newton (Newton-Raphson) en una variable
 CURSO: Fisica Computacional II
=============================================================================

DEDUCCION Y JUSTIFICACION
-----------------------------------------------------------------
Para resolver f(x) = 0 se aproxima f cerca de la estimacion x_k por su
recta tangente (Taylor de primer orden):

        f(x) ~ f(x_k) + f'(x_k) (x - x_k) .

Se toma como nueva estimacion el punto donde esa recta corta el eje x:

        f(x_k) + f'(x_k)(x_{k+1} - x_k) = 0
   =>   x_{k+1} = x_k - f(x_k) / f'(x_k) .

CONVERGENCIA CUADRATICA. Sea e_k = x_k - x*. Taylor de f alrededor de x_k,
evaluado en x*, con resto de segundo orden (xi entre x_k y x*):

    0 = f(x*) = f(x_k) - f'(x_k) e_k + (1/2) f''(xi) e_k^2 .

Dividiendo por f'(x_k) y reordenando:

    x_k - f(x_k)/f'(x_k) - x* = (f''(xi) / (2 f'(x_k))) e_k^2
    =>   e_{k+1} = ( f''(xi) / (2 f'(x_k)) ) e_k^2  ->  C e_k^2 ,   C = f''(x*) / (2 f'(x*)) .

El error se ELEVA AL CUADRADO (por una constante) en cada paso: el numero de
cifras correctas se DUPLICA por iteracion. Esto requiere f'(x*) != 0 (raiz
simple) y un arranque suficientemente cercano (|C e_0| < 1).

EJEMPLO FISICO: punto de Lagrange L1 del sistema Tierra-Luna
-----------------------------------------------------------------
Un satelite de masa despreciable en la linea Tierra-Luna, a distancia r de
la Tierra, orbita con la misma velocidad angular w que la Luna si la fuerza
neta provee la aceleracion centripeta:

        G M / r^2  -  G m / (R - r)^2  =  w^2 r ,

(M: masa de la Tierra, m: masa de la Luna, R: distancia Tierra-Luna). Es el
punto L1: el satelite se queda "fijo" respecto a ambos cuerpos. La ecuacion
es polinomica de quinto grado en r: no tiene solucion cerrada.

Cambio a variable adimensional u = r / R. Justificacion: elimina las
magnitudes enormes (1e8 m) y los coeficientes diminutos (1e-11), de modo que
la raiz queda en (0,1) y la aritmetica es mejor condicionada. Dividiendo la
ecuacion por w^2 R:

    g(u) = alpha / u^2  -  beta / (1-u)^2  -  u = 0 ,
    alpha = G M /(w^2 R^3) ,   beta = G m /(w^2 R^3) .

    g'(u)  = -2 alpha/u^3 - 2 beta/(1-u)^3 - 1
    g''(u) =  6 alpha/u^4 - 6 beta/(1-u)^4

(derivadas analiticas, que Newton necesita; el script 13 muestra como
prescindir de ellas con el metodo de la secante).
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

np.set_printoptions(precision=10, suppress=True)

# -----------------------------------------------------------------------
# Constantes fisicas (SI)
# -----------------------------------------------------------------------
G = 6.674e-11            # m^3 kg^-1 s^-2
M_T = 5.974e24           # kg  (Tierra)
M_L = 7.348e22           # kg  (Luna)
R_TL = 3.844e8           # m   (distancia Tierra-Luna)
W = 2.662e-6             # rad/s (velocidad angular de la Luna)

alpha = G * M_T / (W ** 2 * R_TL ** 3)
beta = G * M_L / (W ** 2 * R_TL ** 3)

g = lambda u: alpha / u ** 2 - beta / (1 - u) ** 2 - u
dg = lambda u: -2 * alpha / u ** 3 - 2 * beta / (1 - u) ** 3 - 1
d2g = lambda u: 6 * alpha / u ** 4 - 6 * beta / (1 - u) ** 4

print(f"alpha = {alpha:.8f}, beta = {beta:.8f}")


def newton(f, df, x0, tol=1e-14, max_iter=100, verbose=True):
    """Newton-Raphson. Devuelve la lista de iterados.
    Criterio: |f(x_k)| < tol  o  |x_{k+1}-x_k| < tol * (1 + |x_k|)."""
    xs = [x0]
    for k in range(max_iter):
        x = xs[-1]
        d = df(x)
        if d == 0:
            raise ZeroDivisionError(f"f'(x)=0 en x={x}: la tangente es horizontal")
        x_new = x - f(x) / d
        xs.append(x_new)
        if verbose:
            print(f"  k={k+1:2d}  u = {x_new:.15f}   g(u) = {f(x_new): .3e}")
        if not np.isfinite(x_new):
            break
        if abs(f(x_new)) < tol or abs(x_new - x) < tol * (1 + abs(x_new)):
            break
    return np.array(xs)


# =========================================================================
# 1) Resolucion en doble precision
# =========================================================================
print("\nNewton, u0 = 0.5:")
us = newton(g, dg, 0.5)
u_star = us[-1]
r_L1 = u_star * R_TL
print(f"\nu* = {u_star:.15f}")
print(f"r (distancia a la Tierra) = {r_L1:.6e} m = {r_L1/1e3:,.0f} km")
print(f"Distancia a la Luna       = {(R_TL - r_L1)/1e3:,.0f} km")
print(f"Residuo |g(u*)| = {abs(g(u_star)):.2e}")

# =========================================================================
# 2) Analisis de error con precision extendida (mpmath, 60 digitos)
#    Se usa SOLO para poder ver la convergencia cuadratica mas alla de los
#    ~16 digitos de double; el algoritmo es el mismo.
# =========================================================================
mp.mp.dps = 60
alpha_mp = mp.mpf(G) * mp.mpf(M_T) / (mp.mpf(W) ** 2 * mp.mpf(R_TL) ** 3)
beta_mp = mp.mpf(G) * mp.mpf(M_L) / (mp.mpf(W) ** 2 * mp.mpf(R_TL) ** 3)
g_mp = lambda u: alpha_mp / u ** 2 - beta_mp / (1 - u) ** 2 - u
dg_mp = lambda u: -2 * alpha_mp / u ** 3 - 2 * beta_mp / (1 - u) ** 3 - 1

u_exacta = mp.findroot(g_mp, mp.mpf("0.85"))        # raiz con 60 digitos
u0 = mp.mpf("0.3")
trayecto = [u0]
for _ in range(9):
    trayecto.append(trayecto[-1] - g_mp(trayecto[-1]) / dg_mp(trayecto[-1]))
errores_mp = [abs(x - u_exacta) for x in trayecto]

C_teorica = abs(d2g(float(u_exacta)) / (2 * dg(float(u_exacta))))
print(f"\nRaiz con 60 digitos: u* = {mp.nstr(u_exacta, 30)}")
print(f"Constante asintotica teorica C = |g''/(2g')| = {C_teorica:.6f}")
print(f"\n{'k':>2} {'error e_k':>14} {'e_{k+1}/e_k^2':>14} {'cifras correctas':>17}")
for k in range(len(errores_mp) - 1):
    ratio = errores_mp[k + 1] / errores_mp[k] ** 2
    cifras = -mp.log10(errores_mp[k]) if errores_mp[k] > 0 else mp.inf
    print(f"{k:2d} {mp.nstr(errores_mp[k], 5):>14} {mp.nstr(ratio, 8):>14} {mp.nstr(cifras, 4):>17}")

# =========================================================================
# 3) Comparacion con biseccion (script 11)
# =========================================================================
def biseccion_u(f, a, b, tol=1e-14):
    xs = []
    fa = f(a)
    while (b - a) > tol:
        c = 0.5 * (a + b)
        xs.append(c)
        if fa * f(c) < 0:
            b = c
        else:
            a, fa = c, f(c)
    return np.array(xs)


us_bis = biseccion_u(g, 0.3, 0.99)
u_ref = float(u_exacta)
err_newton_dp = np.abs(newton(g, dg, 0.3, verbose=False) - u_ref)
err_bis = np.abs(us_bis - u_ref)
print(f"\nPara error < 1e-14:  Newton = {len(err_newton_dp)-1} iteraciones,  "
      f"biseccion = {len(us_bis)} iteraciones")

# =========================================================================
# 4) Cuenca de atraccion en L1 y en el pozo cuadrado (Newton puede fallar)
# =========================================================================
u_inicios = np.linspace(0.02, 0.98, 97)
iter_L1 = []
raiz_L1 = []
for u0_ in u_inicios:
    try:
        tray = newton(g, dg, u0_, verbose=False, max_iter=60)
        ok = np.isfinite(tray[-1]) and abs(g(tray[-1])) < 1e-9
        raiz_L1.append(tray[-1] if ok else np.nan)
        iter_L1.append(len(tray) - 1 if ok else np.nan)
    except ZeroDivisionError:
        raiz_L1.append(np.nan); iter_L1.append(np.nan)
raiz_L1 = np.array(raiz_L1)
print(f"\nL1: Newton desde 97 arranques en (0,1): "
      f"{np.sum(np.abs(raiz_L1 - u_ref) < 1e-8)} convergen a L1, "
      f"{np.sum(np.isnan(raiz_L1))} fallan.")

# Pozo cuadrado (estados pares), variable z; f_par(z) = z sin z - s cos z
Z0 = 11.45575
s_ = lambda z: np.sqrt(Z0 ** 2 - z ** 2)
fz = lambda z: z * np.sin(z) - s_(z) * np.cos(z)
dfz = lambda z: np.sin(z) + z * np.cos(z) + (z / s_(z)) * np.cos(z) + s_(z) * np.sin(z)

z_inicios = np.linspace(0.3, Z0 - 0.05, 400)
z_final = []
for z0_ in z_inicios:
    z = z0_
    try:
        with np.errstate(all="ignore"):
            for _ in range(60):
                z_new = z - fz(z) / dfz(z)
                if not np.isfinite(z_new) or z_new <= 0 or z_new >= Z0:
                    z = np.nan
                    break
                if abs(z_new - z) < 1e-12:
                    z = z_new
                    break
                z = z_new
    except ZeroDivisionError:
        z = np.nan
    z_final.append(z)
z_final = np.array(z_final)

raices_par = np.unique(np.round(z_final[np.isfinite(z_final)], 6))
print("\nPozo cuadrado, ecuacion par, Newton desde 400 arranques en (0, z0):")
print("   raices alcanzadas (z):", raices_par[:12], "...")
print(f"   {np.sum(np.isnan(z_final))} arranques salieron del dominio (0,z0) y fallaron")
print("   -> Newton llega a raices DISTINTAS segun el arranque; la biseccion con el\n"
      "      intervalo adecuado (script 11) siempre da la raiz que se pide.")

# =========================================================================
# 5) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# (a) interpretacion geometrica (tangentes)
uu = np.linspace(0.2, 0.98, 500)
axes[0, 0].plot(uu, g(uu), color="#2E86AB", label=r"$g(u)$")
axes[0, 0].axhline(0, color="k", lw=0.8)
xk = 0.3
for _ in range(3):
    xn = xk - g(xk) / dg(xk)
    axes[0, 0].plot([xk, xk, xn], [0, g(xk), 0], color="#C1121F", lw=1)
    axes[0, 0].plot(xk, g(xk), "o", color="#C1121F", ms=4)
    xk = xn
axes[0, 0].plot(u_ref, 0, "*", color="#F18F01", ms=15, label=r"$u^*$ (L1)")
axes[0, 0].set_ylim(-3, 10)
axes[0, 0].set_xlabel(r"$u=r/R$")
axes[0, 0].set_title("Newton: cada paso sigue la tangente hasta el eje")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# (b) Newton vs biseccion
axes[0, 1].semilogy(np.arange(len(err_bis)), err_bis + 1e-17, label="biseccion")
axes[0, 1].semilogy(np.arange(len(err_newton_dp)), err_newton_dp + 1e-17, "o-", label="Newton")
axes[0, 1].set_ylim(1e-17, 2)
axes[0, 1].set_xlabel("Iteracion k")
axes[0, 1].set_ylabel(r"$|u_k-u^*|$")
axes[0, 1].set_title("Newton (cuadratica) vs biseccion (lineal)")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3, which="both")

# (c) e_{k+1} vs e_k en log-log: pendiente 2
e = np.array([float(x) for x in errores_mp])
axes[0, 2].loglog(e[:-1], e[1:], "o", color="#2E86AB", label="datos (60 digitos)")
ee = np.logspace(-30, np.log10(e[0]), 50)
axes[0, 2].loglog(ee, C_teorica * ee ** 2, "k--", label=rf"$C\,e^2$, $C={C_teorica:.3f}$")
axes[0, 2].set_xlabel(r"$e_k$")
axes[0, 2].set_ylabel(r"$e_{k+1}$")
axes[0, 2].set_title("Pendiente 2: convergencia cuadratica")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3, which="both")

# (d) iteraciones segun el arranque (L1)
axes[1, 0].plot(u_inicios, iter_L1, "o-", ms=3, color="#2E86AB")
axes[1, 0].set_xlabel(r"Arranque $u_0$")
axes[1, 0].set_ylabel("Iteraciones hasta convergencia")
axes[1, 0].set_title("L1: Newton converge desde todo (0,1)")
axes[1, 0].grid(alpha=0.3)

# (e) cifras correctas por iteracion
cifras = np.array([float(-mp.log10(x)) for x in errores_mp if x > 0])
axes[1, 1].plot(np.arange(len(cifras)), cifras, "o-", color="#C1121F")
axes[1, 1].set_xlabel("Iteracion k")
axes[1, 1].set_ylabel("Cifras decimales correctas")
axes[1, 1].set_title("Las cifras correctas se duplican")
axes[1, 1].grid(alpha=0.3)

# (f) raiz alcanzada vs arranque en el pozo
axes[1, 2].plot(z_inicios, z_final, ".", ms=3, color="#6A4C93")
axes[1, 2].set_xlabel(r"Arranque $z_0$")
axes[1, 2].set_ylabel("Raiz alcanzada z")
axes[1, 2].set_title("Pozo cuadrado: Newton salta entre raices")
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("12_newton.png", dpi=150)
print("\nGrafica guardada en 12_newton.png")