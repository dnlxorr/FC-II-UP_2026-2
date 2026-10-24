
"""
Taller de Física Computacional
Análisis estructural y espectral de un pozo de potencial asimétrico (Lennard-Jones 12-6)

    V(x) = x^(-12) - x^(-6),   x > 0      (unidades reducidas: epsilon = sigma = 1)

Regla del taller: los algoritmos clave se programan A MANO.
Solo se usan numpy (arreglos) y matplotlib (gráficas).

Ejecución:  python taller_lennard_jones.py
Salida   :  figuras/*.pdf  y  resultados.json  (los números que usa el informe .tex)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figuras", exist_ok=True)
EPS_MAQ = np.finfo(float).eps          # epsilon de máquina (~2.2e-16)
R = {}                                 # diccionario de resultados -> resultados.json

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "savefig.bbox": "tight"})


# ----------------------------------------------------------------------------
# 0. El potencial y su derivada analítica (la derivada analítica SOLO se usa
#    para verificar resultados y para Newton-Raphson, que la necesita por definición)
# ----------------------------------------------------------------------------
def V(x):
    return x ** (-12.0) - x ** (-6.0)


def dV(x):
    return -12.0 * x ** (-13.0) + 6.0 * x ** (-7.0)


X0_EXACTO = 1.0                        # V(1) = 1 - 1 = 0
XMIN_EXACTO = 2.0 ** (1.0 / 6.0)       # dV/dx = 0  ->  x^6 = 2
VMIN_EXACTO = -0.25                    # V(2^(1/6)) = 1/4 - 1/2


# ----------------------------------------------------------------------------
# 1. ALGORITMOS NUMÉRICOS (implementados desde cero)
# ----------------------------------------------------------------------------
def biseccion(f, a, b, tol=1e-12, max_iter=200):
    """Bisección. Requiere cambio de signo f(a)*f(b) < 0 (teorema de Bolzano).
    Cada iteración reduce el intervalo a la mitad y cuesta UNA evaluación de f."""
    fa, fb = f(a), f(b)
    n_eval = 2
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b]: no se garantiza raíz.")
    historia = []                      # (evaluaciones acumuladas, punto medio)
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        n_eval += 1
        historia.append((n_eval, c))
        if fc == 0.0 or 0.5 * (b - a) < tol:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, n_eval, historia


def newton_raphson(f, df, x, tol=1e-12, max_iter=100):
    """Newton-Raphson: x_{k+1} = x_k - f/f'. Cada iteración cuesta DOS evaluaciones
    (f y f')."""
    n_eval = 0
    historia = []
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        n_eval += 2
        paso = fx / dfx
        x = x - paso
        historia.append((n_eval, x))
        if abs(paso) < tol:
            break
    return x, n_eval, historia


def busqueda_dorada(f, a, b, tol=1e-7, max_iter=500):
    """Búsqueda de la sección áurea para el mínimo de una función unimodal en [a, b].
    Reutiliza una de las dos evaluaciones interiores en cada paso (razón 1/phi)."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0          # 1/phi = 0.618...
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    n_eval = 2
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        if fc < fd:                    # el mínimo queda en [a, d]
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:                          # el mínimo queda en [c, b]
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
        n_eval += 1
    xm = 0.5 * (a + b)
    return xm, f(xm), n_eval + 1


def derivada_centrada(f, x, h):
    """Diferencia finita centrada, orden O(h^2):  f'(x) ~ [f(x+h) - f(x-h)] / (2h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def simpson(y, h):
    """Regla de Simpson 1/3 compuesta. y tiene M+1 puntos con M PAR; paso h.
    I ~ h/3 [ y0 + 4(y1+y3+...) + 2(y2+y4+...) + yM ]"""
    if (len(y) - 1) % 2 != 0:
        raise ValueError("Simpson 1/3 necesita un número PAR de subintervalos.")
    return h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())


def coef_fourier(f, x0, L, N, M=20000):
    """a_n, b_n (n = 0..N) de la serie de Fourier en [x0, x0+L] con Simpson 1/3:
       a_n = (2/L) int f cos(2 pi n x / L) dx,   b_n = (2/L) int f sin(2 pi n x / L) dx."""
    x = np.linspace(x0, x0 + L, M + 1)
    h = L / M
    fx = f(x)
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    for n in range(N + 1):
        k = 2.0 * np.pi * n / L
        a[n] = 2.0 / L * simpson(fx * np.cos(k * x), h)
        b[n] = 2.0 / L * simpson(fx * np.sin(k * x), h)
    return a, b


def serie_fourier(x, a, b, L, N):
    """Suma parcial V_p^(N)(x) = a0/2 + sum_{n=1}^{N} [a_n cos + b_n sin]."""
    s = np.full_like(x, a[0] / 2.0)
    for n in range(1, N + 1):
        k = 2.0 * np.pi * n / L
        s += a[n] * np.cos(k * x) + b[n] * np.sin(k * x)
    return s


# ----------------------------------------------------------------------------
# PARTE A
# ----------------------------------------------------------------------------
print("=" * 70)
print("PARTE A: raíces y mínimos")
print("=" * 70)

# --- A1. Punto de corte x0 (V(x0) = 0) con Bisección --------------------------
# Bracketing: V(0.9) = +1.66 > 0 y V(1.5) = -0.080 < 0  -> cambio de signo.
# (Se evita un intervalo simétrico alrededor de x0 = 1: el punto medio caería
#  exactamente en la raíz y la comparación de costos sería artificialmente favorable.)
A_BRACKET, B_BRACKET = 0.9, 1.5
x0_bis, nev_bis, hist_bis = biseccion(V, A_BRACKET, B_BRACKET, tol=1e-12)
print(f"[A1] Bisección      x0 = {x0_bis:.12f}  (exacto 1)  evaluaciones = {nev_bis}")
R["x0_biseccion"] = x0_bis
R["x0_biseccion_evals"] = nev_bis

# --- A2. Mínimo de energía con Búsqueda Dorada -------------------------------
# Tolerancia 1e-7: cerca del mínimo V es plana (V ~ Vmin + V''/2 dx^2), así que
# comparar valores de V solo discrimina dx hasta ~ sqrt(eps_maq) ~ 1.5e-8.
xmin_num, vmin_num, nev_gs = busqueda_dorada(V, 1.0, 2.0, tol=1e-7)
print(f"[A2] Búsq. Dorada   xmin = {xmin_num:.8f}  (exacto {XMIN_EXACTO:.8f})"
      f"  Vmin = {vmin_num:.10f}  evaluaciones = {nev_gs}")
R.update(xmin_dorada=xmin_num, xmin_exacto=XMIN_EXACTO, vmin_dorada=vmin_num,
         xmin_dorada_evals=nev_gs, xmin_dorada_error=abs(xmin_num - XMIN_EXACTO))

# --- A3. Fuerza F = -dV/dx por diferencias centradas -------------------------
# Estudio del paso h en x = 1 (donde F exacta = 6) para ver truncamiento vs redondeo.
tabla_h = []
print("[A3] Estudio del paso h (F exacta en x=1 es 6):")
for e in range(1, 13):
    h = 10.0 ** (-e)
    F_num = -derivada_centrada(V, 1.0, h)
    tabla_h.append((h, F_num, abs(F_num - 6.0)))
    print(f"      h = 1e-{e:02d}   F_num = {F_num:.12f}   |error| = {abs(F_num - 6.0):.2e}")
R["tabla_h"] = tabla_h
mejor = min(tabla_h, key=lambda t: t[2])
R["h_optimo_empirico"] = mejor[0]
# h óptimo teórico: h* = (3 eps |V| / |V'''|)^(1/3)  (se deduce en el informe)
# V'''(x) = -2184 x^-15 + 336 x^-9  ->  V'''(1) = -1848
# Error total ~ h^2/6 |V'''| (truncamiento) + eps*S/h (redondeo), con S ~ 1 la escala de
# los sumandos x^-12 y x^-6 en x = 1 (V(1) = 0 por cancelación, pero cada término vale 1).
V3 = -2184.0 + 336.0
R["V3_en_1"] = V3
h_teo = (3.0 * EPS_MAQ * 1.0 / abs(V3)) ** (1.0 / 3.0)
R["h_optimo_teorico_orden"] = h_teo
print(f"[A3] V'''(1) = {V3:.0f};  h* teórico = (3 eps S/|V'''|)^(1/3) = {h_teo:.2e};  "
      f"mejor h empírico = {mejor[0]:.0e}")

H_FUERZA = 1e-5
F_xmin = -derivada_centrada(V, xmin_num, H_FUERZA)
F_xmin_exacta_en_xnum = -dV(xmin_num)
print(f"[A3] F(xmin) con h={H_FUERZA:g}: {F_xmin:.3e}  (analítica evaluada en el mismo "
      f"xmin numérico: {F_xmin_exacta_en_xnum:.3e})")
R.update(F_xmin=F_xmin, F_xmin_analitica_mismo_x=F_xmin_exacta_en_xnum, h_fuerza=H_FUERZA)

# --- A4. Puntos de retorno: V(x) = E ------------------------------------------
# OJO: Vmin = -1/4, de modo que para E = -0.5 < Vmin NO existen puntos de retorno.
print("[A4] Puntos de retorno")
g_malla = np.linspace(0.8, 20.0, 400001)
gmin = (V(g_malla) - (-0.5)).min()
print(f"      E = -0.5:  min[V(x) - E] sobre malla = {gmin:.6f} > 0  -> V(x) = E no tiene solución")
R["E_-0.5_min_V_menos_E"] = float(gmin)
try:
    biseccion(lambda x: V(x) + 0.5, 0.8, 20.0)
except ValueError as err:
    print(f"      Bisección en [0.8, 20]: {err}")
    R["E_-0.5_biseccion"] = str(err)

retornos = {}
for E in (-0.2, -0.1):
    x1, ne1, _ = biseccion(lambda x, E=E: V(x) - E, 1.0, xmin_num, tol=1e-12)
    x2, ne2, _ = biseccion(lambda x, E=E: V(x) - E, xmin_num, 6.0, tol=1e-12)
    retornos[E] = (x1, x2)
    print(f"      E = {E:5.2f}:  x1 = {x1:.10f}   x2 = {x2:.10f}   "
          f"(x2 - xmin)/(xmin - x1) = {(x2 - XMIN_EXACTO) / (XMIN_EXACTO - x1):.4f}")
R["retornos"] = {str(E): list(v) for E, v in retornos.items()}

# Verificación independiente con forma cerrada. Con u = x^-6:  V = u^2 - u = (u - 1/2)^2 - 1/4
#   V = E  ->  u = 1/2 +- sqrt(1/4 + E)  ->  x = u^(-1/6)   (solo si E >= -1/4)
print("      Verificación con forma cerrada x = [1/2 ± sqrt(1/4 + E)]^(-1/6):")
cerrada = {}
for E in (-0.2, -0.1):
    raiz = np.sqrt(0.25 + E)
    x1c, x2c = (0.5 + raiz) ** (-1.0 / 6.0), (0.5 - raiz) ** (-1.0 / 6.0)
    cerrada[E] = (x1c, x2c)
    print(f"        E = {E:5.2f}:  x1 = {x1c:.10f}  x2 = {x2c:.10f}   "
          f"|dif. con bisección| = {abs(x1c - retornos[E][0]):.1e}, {abs(x2c - retornos[E][1]):.1e}")
R["retornos_forma_cerrada"] = {str(E): list(v) for E, v in cerrada.items()}
print(f"        E = -0.5: 1/4 + E = {0.25 - 0.5:.2f} < 0  ->  u complejo: no hay solución real")

# Cantidades analíticas usadas en el informe (derivadas de V)
d1_08 = dV(0.8)
V3_xmin = 2.0 ** (-2.5) * (-2184.0 + 336.0 * 2.0)       # V'''(xmin) en forma cerrada
print(f"      V'(0.8) = {d1_08:.2f};  V'''(xmin) = {V3_xmin:.2f};  k = V''(xmin) = {36 * 2 ** (-4 / 3):.4f}")
R["dV_en_0p8"] = d1_08
R["V3_en_xmin"] = V3_xmin

# --- Pregunta 1: Bisección vs Newton-Raphson ---------------------------------
print("[P1] Bisección vs Newton-Raphson para x0 (evaluaciones de función)")
tabla_conv = []
for tol in (1e-4, 1e-8, 1e-12):
    _, nb, _ = biseccion(V, A_BRACKET, B_BRACKET, tol=tol)
    _, nn, _ = newton_raphson(V, dV, 0.9, tol=tol)
    tabla_conv.append((tol, nb, nn))
    print(f"      tol = {tol:.0e}:  Bisección = {nb:3d} evals   Newton(x_ini=0.9) = {nn:3d} evals")
R["tabla_convergencia"] = tabla_conv
x0_new, nev_new, hist_new = newton_raphson(V, dV, 0.9, tol=1e-14)
x0_bis2, nev_bis2, hist_bis2 = biseccion(V, A_BRACKET, B_BRACKET, tol=1e-14)
R["newton_iterados"] = [(n, x) for n, x in hist_new]

# Robustez: Newton arrancando a la DERECHA de la raíz (x_ini = 1.5) diverge, porque
# allí V < 0 y V' > 0 (V crece hacia 0^-), y el paso -V/V' empuja hacia x grandes.
x_r, _, hist_r = newton_raphson(V, dV, 1.5, tol=1e-12, max_iter=12)
print("      Newton desde x_ini = 1.5 (12 iteraciones):",
      " -> ".join(f"{h[1]:.3g}" for h in hist_r[:6]), "...", f"{hist_r[-1][1]:.3g}")
R["newton_desde_1p5"] = [h[1] for h in hist_r]

# --- Figura 1: potencial con mínimo y puntos de retorno ----------------------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
xx = np.linspace(0.92, 3.5, 2000)
ax.plot(xx, V(xx), "k", lw=1.8, label=r"$V(x)=x^{-12}-x^{-6}$")
ax.axhline(0, color="gray", lw=0.8)
ax.plot(x0_bis, 0, "o", color="tab:red", label=fr"$x_0={x0_bis:.4f}$ (bisección)")
ax.plot(xmin_num, vmin_num, "s", color="tab:blue",
        label=fr"$x_{{min}}={xmin_num:.4f}$, $V_{{min}}={vmin_num:.3f}$")
cols = {-0.2: "tab:green", -0.1: "tab:orange"}
for E, (x1, x2) in retornos.items():
    ax.hlines(E, x1, x2, colors=cols[E], linestyles="--", lw=1.2)
    ax.plot([x1, x2], [E, E], "^", color=cols[E],
            label=fr"$E={E}$: $x_1={x1:.3f}$, $x_2={x2:.3f}$")
ax.axhline(-0.5, color="tab:purple", ls=":", lw=1.5)
ax.annotate(r"$E=-0.5<V_{min}$: sin puntos de retorno", (2.0, -0.5),
            xytext=(1.7, -0.42), color="tab:purple", fontsize=9)
ax.set_xlim(0.92, 3.5)
ax.set_ylim(-0.6, 1.0)
ax.set_xlabel(r"$x$ (unidades de $\sigma$)")
ax.set_ylabel(r"$V$ (unidades de $\varepsilon$)")
ax.set_title("Potencial de Lennard-Jones: raíz, mínimo y puntos de retorno")
ax.legend(fontsize=8, loc="upper right")
fig.savefig("figuras/fig1_potencial.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------
# PARTE B
# ----------------------------------------------------------------------------
print("=" * 70)
print("PARTE B: serie de Fourier periodizada en [0.8, 3.8]")
print("=" * 70)
XA, L = 0.8, 3.0
XB = XA + L
NMAX = 50                    # N = 50 armónicos exige calcular hasta n = 50
M_SIMP = 20000

# --- B2. Coeficientes con Simpson 1/3 ----------------------------------------
a_n, b_n = coef_fourier(V, XA, L, NMAX, M=M_SIMP)
print("   n        a_n              b_n")
for n in range(0, 21):
    print(f"  {n:2d}   {a_n[n]:+14.8f}   {b_n[n]:+14.8f}")
R["a_n"] = a_n.tolist()
R["b_n"] = b_n.tolist()
R["a0_medio"] = a_n[0] / 2.0

# Validación de Simpson: (i) refinamiento de malla; (ii) caso con solución analítica
print("[B2] Validación de Simpson")
val = {}
for M in (2000, 4000, 8000, 20000):
    a_t, b_t = coef_fourier(V, XA, L, 50, M=M)
    val[M] = (a_t[1], b_t[1], a_t[50], b_t[50])
for M in (2000, 4000, 8000):
    d = [abs(val[M][i] - val[20000][i]) for i in range(4)]
    print(f"      M = {M:5d}:  |Δa1|={d[0]:.1e} |Δb1|={d[1]:.1e} |Δa50|={d[2]:.1e} |Δb50|={d[3]:.1e}")
R["validacion_simpson"] = {str(M): v for M, v in val.items()}

K_ARM = 36.0 * 2.0 ** (-4.0 / 3.0)          # V''(xmin) del LJ = 36 * 2^(-4/3)
R["k_armonico"] = K_ARM
Vh = lambda xi: 0.5 * K_ARM * xi ** 2        # pozo armónico centrado en su mínimo
ah, bh = coef_fourier(Vh, -L / 2, L, NMAX, M=M_SIMP)
ah_exacto = np.array([K_ARM * L ** 2 * (-1) ** n / (2 * np.pi ** 2 * n ** 2) if n else 0 for n in range(NMAX + 1)])
err_val = np.max(np.abs(ah[1:] - ah_exacto[1:]))
print(f"      Caso armónico (analítico a_n = k L^2 (-1)^n / (2 pi^2 n^2)): "
      f"error máx |a_n num - exacto| = {err_val:.2e}, máx |b_n| = {np.max(np.abs(bh)):.2e}")
R["validacion_armonico_error_a"] = float(err_val)
R["armonico_max_bn"] = float(np.max(np.abs(bh)))

# --- B3. Reconstrucción ------------------------------------------------------
xf = np.linspace(XA, XB, 30001)
Vexa = V(xf)
J = V(XA) - V(XB)            # salto en el borde de la extensión periódica
print(f"[B3] V(0.8) = {V(XA):.6f}  V(3.8) = {V(XB):.6e}  salto J = {J:.6f}  "
      f"-> la serie converge en el borde a (V(0.8)+V(3.8))/2 = {(V(XA) + V(XB)) / 2:.6f}")
R.update(V_en_0p8=V(XA), V_en_3p8=V(XB), salto_J=J, punto_medio_salto=(V(XA) + V(XB)) / 2)

Ns = [5, 10, 20, 50]
recon = {}
tabla_err = []
GIBBS = 0.0894898722360838   # (1/pi) Si(pi) - 1/2   (constante de Wilbraham-Gibbs)
R["constante_gibbs"] = GIBBS
for N in Ns:
    vp = serie_fourier(xf, a_n, b_n, L, N)
    recon[N] = vp
    err = vp - Vexa
    err_abs = np.abs(err)
    max_total = err_abs.max()
    x_max_total = xf[err_abs.argmax()]
    lejos = (xf > XA + 0.1) & (xf < XB - 0.1)
    max_int = err_abs[lejos].max()
    rms = np.sqrt(np.mean(err ** 2))
    # Overshoot de Gibbs: pico positivo del error justo a la derecha de x=0.8
    zona_izq = (xf > XA) & (xf < XA + 2.0 * L / N)
    zona_der = (xf < XB) & (xf > XB - 2.0 * L / N)
    ov_izq = err[zona_izq].max()
    ov_der = err[zona_der].min()
    tabla_err.append(dict(N=N, max_total=max_total, x_max_total=x_max_total, max_interior=max_int,
                          rms=rms, overshoot_izq=ov_izq, overshoot_der=ov_der))
    print(f"      N = {N:2d}: max|err| total = {max_total:.4f} (en x={x_max_total:.3f}) | "
          f"max|err| interior(|x-borde|>0.1) = {max_int:.4f} | RMS = {rms:.4f} | "
          f"overshoot izq = {ov_izq:.4f}, der = {ov_der:.4f}  (0.0895 J = {GIBBS * J:.4f})")
R["tabla_errores"] = tabla_err

# Dependencia del error interior con N (para verificar el decaimiento ~ 1/N)
Ns_scan = [5, 10, 20, 30, 40, 50]
R["interior_vs_N"] = []
for N in Ns_scan:
    vp = serie_fourier(xf, a_n, b_n, L, N)
    lejos = (xf > XA + 0.1) & (xf < XB - 0.1)
    R["interior_vs_N"].append((N, float(np.abs(vp - Vexa)[lejos].max())))

# --- Figuras 2-4 ---------------------------------------------------------------
# Fig 2: espectro
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
n20 = np.arange(0, 21)
axs[0].stem(n20, a_n[:21], linefmt="C0-", markerfmt="C0o", basefmt="k-", label=r"$a_n$")
axs[0].set_title(r"Coeficientes coseno $a_n$")
axs[0].set_xlabel("$n$")
axs[0].set_ylabel("$a_n$")
axs[1].stem(n20[1:], b_n[1:21], linefmt="C3-", markerfmt="C3o", basefmt="k-", label=r"$b_n$")
axs[1].set_title(r"Coeficientes seno $b_n$")
axs[1].set_xlabel("$n$")
axs[1].set_ylabel("$b_n$")
fig.suptitle("Espectro de Fourier del LJ periodizado en [0.8, 3.8]", y=1.04)
fig.savefig("figuras/fig2_espectro.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
nn = np.arange(1, NMAX + 1)
ax.semilogy(nn, np.abs(a_n[1:]), "o-", ms=3.5, label=r"$|a_n|$")
ax.semilogy(nn, np.abs(b_n[1:]), "s-", ms=3.5, label=r"$|b_n|$")
ax.semilogy(nn, 2 * J / (2 * np.pi * nn), "k--", lw=1, label=r"$J/(\pi n)$ (salto)")
ax.set_xlabel("$n$")
ax.set_ylabel("Magnitud")
ax.set_title(r"Espectro $|a_n|$, $|b_n|$ vs $n$ (escala log)")
ax.legend()
fig.savefig("figuras/fig2b_espectro_log.pdf")
plt.close(fig)

# Fig 3: reconstrucción
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for ax, N in zip(axs.ravel(), Ns):
    ax.plot(xf, Vexa, "k", lw=1.6, label=r"$V(x)$ exacto")
    ax.plot(xf, recon[N], "tab:red", lw=1.0, label=fr"Fourier $N={N}$")
    ax.set_title(fr"$N={N}$ armónicos")
    ax.set_ylim(-3, 14)
    ax.legend(fontsize=8)
for ax in axs[1]:
    ax.set_xlabel("$x$")
for ax in axs[:, 0]:
    ax.set_ylabel("$V$")
fig.suptitle("Reconstrucción de Fourier del potencial de Lennard-Jones")
fig.savefig("figuras/fig3_reconstruccion.pdf")
plt.close(fig)

# Fig 4: Gibbs (error cerca de los dos bordes)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for N, c in zip((10, 20, 50), ("C0", "C1", "C3")):
    err = recon[N] - Vexa
    m1 = xf < XA + 0.45
    m2 = xf > XB - 0.45
    axs[0].plot(xf[m1], err[m1], c, lw=1.1, label=f"$N={N}$")
    axs[1].plot(xf[m2], err[m2], c, lw=1.1, label=f"$N={N}$")
for ax in axs:
    ax.axhline(GIBBS * J, color="k", ls="--", lw=0.8)
    ax.axhline(-GIBBS * J, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("$x$")
    ax.legend(fontsize=8)
axs[0].set_ylabel(r"error $V_p^{(N)}-V$")
axs[0].set_title(r"Borde izquierdo ($x=0.8$)")
axs[1].set_title(r"Borde derecho ($x=3.8$)")
fig.suptitle(r"Fenómeno de Gibbs: el pico no decrece con $N$ (líneas: $\pm0.0895\,J$)", y=1.04)
fig.savefig("figuras/fig4_gibbs.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------
# Pregunta 1 (figura) y Pregunta 2 (figura)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
nb_ev = [h[0] for h in hist_bis2]
eb = [max(abs(h[1] - 1.0), 1e-17) for h in hist_bis2]
nn_ev = [h[0] for h in hist_new]
en = [max(abs(h[1] - 1.0), 1e-17) for h in hist_new]
ax.semilogy(nb_ev, eb, "o-", ms=3, label="Bisección")
ax.semilogy(nn_ev, en, "s-", ms=5, label="Newton-Raphson")
ax.set_xlabel("evaluaciones de función (f y f' cuentan por separado)")
ax.set_ylabel(r"$|x_k - x_0|$")
ax.set_title(r"Convergencia hacia $x_0=1$")
ax.legend()
fig.savefig("figuras/fig5_convergencia_raiz.pdf")
plt.close(fig)

# Pregunta 2: decaimiento espectral LJ vs armónico (amplitud A_n = sqrt(a_n^2 + b_n^2))
A_lj = np.sqrt(a_n[1:] ** 2 + b_n[1:] ** 2)
A_h = np.sqrt(ah[1:] ** 2 + bh[1:] ** 2)
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.loglog(nn, A_lj / A_lj[0], "o-", ms=3.5, label="Lennard-Jones (asimétrico)")
ax.loglog(nn, A_h / A_h[0], "s-", ms=3.5, label="Pozo armónico (simétrico)")
ax.loglog(nn, 1.0 / nn, "k--", lw=0.9, label=r"$\propto 1/n$")
ax.loglog(nn, 1.0 / nn ** 2, "k:", lw=0.9, label=r"$\propto 1/n^2$")
ax.set_xlabel("$n$")
ax.set_ylabel(r"$A_n/A_1$")
ax.set_title("Decaimiento espectral")
ax.legend(fontsize=8)
fig.savefig("figuras/fig6_decaimiento.pdf")
plt.close(fig)

# pendientes log-log por ajuste lineal a mano (mínimos cuadrados), n = 5..50
def pendiente(n, y):
    lx, ly = np.log(n), np.log(y)
    return np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2)

sel = slice(4, None)
p_lj = pendiente(nn[sel], A_lj[sel])
p_h = pendiente(nn[sel], A_h[sel])
sel2 = slice(24, None)                                   # n = 25..50
p_lj2 = pendiente(nn[sel2], A_lj[sel2])
print(f"[P2] pendiente log-log de A_n:  LJ(n=5..50) = {p_lj:.3f}, LJ(n=25..50) = {p_lj2:.3f}, "
      f"armónico(n=5..50) = {p_h:.3f}")
razon = A_lj / (J / (np.pi * nn))                        # A_n frente a la asíntota del salto
print("      A_n / [J/(pi n)] para n = 5, 10, 20, 30, 40, 50:",
      ", ".join(f"{razon[n - 1]:.3f}" for n in (5, 10, 20, 30, 40, 50)))
R["razon_asintota_salto"] = {str(n): float(razon[n - 1]) for n in (5, 10, 20, 30, 40, 50)}
R["pendiente_LJ_n25_50"] = p_lj2
print(f"      armónico: max|b_n| = {np.max(np.abs(bh)):.1e} (simétrico respecto a su centro)")
potencia_b = np.sum(b_n[1:21] ** 2) / np.sum(a_n[1:21] ** 2 + b_n[1:21] ** 2)
print(f"      LJ: fracción de potencia en senos (n<=20) = {potencia_b:.3f}")
R.update(pendiente_LJ=p_lj, pendiente_armonico=p_h, fraccion_potencia_senos=potencia_b)

with open("resultados.json", "w") as fh:
    json.dump(R, fh, indent=2, default=float)
# ----------------------------------------------------------------------------
# Tablas en LaTeX (para \input en el informe): así ningún número se retipea a mano
# ----------------------------------------------------------------------------
os.makedirs("tablas", exist_ok=True)


def sci(v, d=2):
    """Notación científica en modo matemático de LaTeX."""
    m, e = f"{v:.{d}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


with open("tablas/tabla_coef.tex", "w") as fh:
    fh.write("\\begin{tabular}{r r r @{\\hspace{2.2em}} r r r}\n\\toprule\n")
    fh.write("$n$ & $a_n$ & $b_n$ & $n$ & $a_n$ & $b_n$ \\\\\n\\midrule\n")
    for n in range(0, 11):
        izq = f"{n} & ${a_n[n]:+.6f}$ & ${b_n[n]:+.6f}$"
        m = n + 11
        der = f"{m} & ${a_n[m]:+.6f}$ & ${b_n[m]:+.6f}$" if m <= 20 else " & & "
        fh.write(f"{izq} & {der} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/tabla_h.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c}\n\\toprule\n")
    fh.write("$h$ & $F_{\\text{num}}(1)$ & $|F_{\\text{num}}-6|$ \\\\\n\\midrule\n")
    for h, F_num, err in tabla_h:
        fh.write(f"{sci(h, 0)} & ${F_num:.10f}$ & {sci(err)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/tabla_conv.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c}\n\\toprule\n")
    fh.write("Tolerancia & Bisección & Newton-Raphson \\\\\n\\midrule\n")
    for tol, nb, nn_ in tabla_conv:
        fh.write(f"{sci(tol, 0)} & {nb} & {nn_} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/tabla_newton.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c}\n\\toprule\n")
    fh.write("$k$ & $x_k$ & $e_k=x_k-1$ \\\\\n\\midrule\n")
    for k, (ne, xk) in enumerate(hist_new[:6], start=1):
        fh.write(f"{k} & ${xk:.12f}$ & {sci(xk - 1.0)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/tabla_err.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c c c c}\n\\toprule\n")
    fh.write("$N$ & $\\max|e|$ (borde) & $\\max|e|$ (interior) & RMS & "
             "pico Gibbs izq. & pico Gibbs der. \\\\\n\\midrule\n")
    for t in tabla_err:
        fh.write(f"{t['N']} & ${t['max_total']:.4f}$ & ${t['max_interior']:.4f}$ & "
                 f"${t['rms']:.4f}$ & ${t['overshoot_izq']:+.4f}$ & ${t['overshoot_der']:+.4f}$ \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/tabla_razon.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c c c c c}\n\\toprule\n")
    fh.write("$n$ & 5 & 10 & 20 & 30 & 40 & 50 \\\\\n\\midrule\n")
    fh.write("$A_n\\,/\\,[J/(\\pi n)]$ & " +
             " & ".join(f"${razon[n - 1]:.3f}$" for n in (5, 10, 20, 30, 40, 50)) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("\nListo: figuras en ./figuras, tablas en ./tablas y números en resultados.json")
