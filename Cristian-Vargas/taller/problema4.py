# -*- coding: utf-8 -*-
"""Problema 4: gas de Van der Waals (Newton, secante) y sistema no lineal 2D."""
import math, cmath, os
from decimal import Decimal, getcontext
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metodos import *

os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
sal = open("resultados_p4.txt", "w")
def P(*a):
    s = " ".join(str(t) for t in a); print(s); sal.write(s + "\n")

# =====================================================================
#                        PARTE A : Van der Waals
# =====================================================================
getcontext().prec = 80                       # aritmética de 80 dígitos (solo para ver el orden)
D = Decimal
a_, b_, P_, T_, R_ = D("0.3643"), D("4.267e-5"), D("2e6"), D("300"), D("8.314462618")
RT = R_ * T_
fD = lambda v: (P_ + a_ / (v * v)) * (v - b_) - RT
dfD = lambda v: P_ - a_ / (v * v) + 2 * a_ * b_ / (v ** 3)
v0 = RT / P_                                  # semilla: gas ideal
P("RT =", RT, " v_ideal =", v0)

TOLD = D("1e-60")
xsN = newton_1d(fD, dfD, v0, tol=TOLD, max_iter=30)
xsS = secante(fD, v0, D("1.05") * v0, tol=TOLD, max_iter=40)
vstar = xsN[-1]
P("v* (Newton, 80 dígitos) =", vstar)
P("v* (secante)            =", xsS[-1])
P("f(v*) =", fD(vstar))

# comprobación en float
vf = float(vstar)
Z = float(P_ * vstar / RT)
P("v* = %.10e m3/mol ; Z = P v/RT = %.6f ; v/v_ideal = %.6f" % (vf, Z, vf / float(v0)))
# unicidad de la raíz: barrido de signos de f en float
af, bf, Pf, RTf = 0.3643, 4.267e-5, 2e6, float(RT)
ff = lambda v: (Pf + af / v ** 2) * (v - bf) - RTf
vs = np.logspace(math.log10(1.01 * bf), -1, 20000)
cambios = [(vs[i], vs[i + 1]) for i in range(len(vs) - 1) if ff(vs[i]) * ff(vs[i + 1]) < 0]
P("cambios de signo de f en (b, 0.1):", len(cambios), cambios)
# parámetros críticos
Pc = af / (27 * bf ** 2); Tc = 8 * af / (27 * bf * 8.314462618)
P("Pc = %.4e Pa ; Tc = %.2f K ; Tr = %.4f ; Pr = %.4f" % (Pc, Tc, 300 / Tc, Pf / Pc))
# desarrollo virial simple: B(T) = b - a/RT
B2 = bf - af / RTf
P("B2 = %.4e ; Z virial ~ 1 + B P/RT = %.6f" % (B2, 1 + B2 * Pf / RTf))

eN = [abs(x - vstar) for x in xsN]
eS = [abs(x - vstar) for x in xsS]
with open("tablas/p4_newton_secante.tex", "w") as fh:
    fh.write("\\begin{tabular}{r cc cc}\n\\toprule\n & \\multicolumn{2}{c}{Newton} & \\multicolumn{2}{c}{Secante}\\\\\n")
    fh.write("$k$ & $v_k$ [m$^3$/mol] & $e_k=|v_k-v^*|$ & $v_k$ [m$^3$/mol] & $e_k=|v_k-v^*|$\\\\\n\\midrule\n")
    nrow = max(len(xsN), len(xsS))
    for k in range(min(nrow, 9)):
        def cel(xs, er):
            if k < len(xs):
                return "%.12e & %s" % (float(xs[k]), ("%.2e" % float(er[k])) if er[k] > 0 else "$<10^{-60}$")
            return " & "
        fh.write("%d & %s & %s\\\\\n" % (k, cel(xsN, eN), cel(xsS, eS)))
    fh.write("\\bottomrule\n\\end{tabular}\n")


def ordenes(er):
    """p_k = ln(e_{k+1}/e_k)/ln(e_k/e_{k-1})"""
    o = []
    for k in range(1, len(er) - 1):
        if er[k + 1] > D("1e-70") and er[k] > 0 and er[k - 1] > 0:
            o.append(float(((er[k + 1]).ln() / er[k].ln()) if False else
                           ((er[k + 1] / er[k]).ln() / (er[k] / er[k - 1]).ln())))
    return o
oN, oS = ordenes(eN), ordenes(eS)
P("orden Newton :", [round(x, 3) for x in oN])
P("orden secante:", [round(x, 3) for x in oS])
# constante asintótica de Newton: e_{k+1}/e_k^2 -> f''/(2f')
cN = [float(eN[k + 1] / eN[k] ** 2) for k in range(len(eN) - 1) if eN[k + 1] > D("1e-60")]
d2 = lambda v: float(-0) + (-2 * af / v ** 3 * (v - bf) + 4 * af / v ** 2 * 0)  # placeholder (no se usa)
# f'' analítica: f'' = 2a/v^3 - 6ab/v^4
f2 = 2 * af / vf ** 3 - 6 * af * bf / vf ** 4
f1 = Pf - af / vf ** 2 + 2 * af * bf / vf ** 3
P("f'(v*) = %.6e ; f''(v*) = %.6e ; f''/(2f') = %.6e" % (f1, f2, f2 / (2 * f1)))
P("e_{k+1}/e_k^2 (Newton):", cN)
with open("tablas/p4_orden.tex", "w") as fh:
    fh.write("\\begin{tabular}{r cc cc}\n\\toprule\n & \\multicolumn{2}{c}{Newton} & \\multicolumn{2}{c}{Secante}\\\\\n")
    fh.write("$k$ & $e_{k+1}/e_k^{2}$ & $p_k$ & $e_{k+1}/e_k^{1.618}$ & $p_k$\\\\\n\\midrule\n")
    for k in range(1, 7):
        cN2 = "%.4f" % float(eN[k + 1] / eN[k] ** 2) if k + 1 < len(eN) and eN[k + 1] > D("1e-60") else "--"
        pN = "%.3f" % oN[k - 1] if k - 1 < len(oN) else "--"
        phi = (1 + math.sqrt(5)) / 2
        cS = "%.4f" % float(eS[k + 1] / (eS[k] ** D(str(phi)))) if k + 1 < len(eS) and eS[k + 1] > D("1e-60") else "--"
        pS = "%.3f" % oS[k - 1] if k - 1 < len(oS) else "--"
        fh.write("%d & %s & %s & %s & %s\\\\\n" % (k, cN2, pN, cS, pS))
    fh.write("\\bottomrule\n\\end{tabular}\n")

# --- figuras Parte A
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
vv = np.logspace(math.log10(1.2 * bf), -2, 800)
ax[0].loglog(vv, [Pf * 0 + (RTf / (v - bf) - af / v ** 2) for v in vv], label="isoterma VdW, 300 K")
ax[0].loglog(vv, [RTf / v for v in vv], "--", label="gas ideal")
ax[0].axhline(Pf, color="k", lw=.7); ax[0].plot(vf, Pf, "ro", label="$v^*$")
ax[0].set_xlabel("v [m$^3$/mol]"); ax[0].set_ylabel("P [Pa]"); ax[0].set_title("Isoterma de CO$_2$"); ax[0].legend(fontsize=8)
vv2 = np.linspace(0.5e-3, 2.2e-3, 300)
ax[1].plot(vv2, [ff(v) for v in vv2]); ax[1].axhline(0, color="k", lw=.7); ax[1].plot(vf, 0, "ro")
ax[1].set_xlabel("v [m$^3$/mol]"); ax[1].set_ylabel("f(v) [J/mol]"); ax[1].set_title("Función $f(v)$")
eNf = [max(float(x), 1e-62) for x in eN]; eSf = [max(float(x), 1e-62) for x in eS]
ax[2].semilogy(range(len(eNf)), eNf, "o-", label="Newton")
ax[2].semilogy(range(len(eSf)), eSf, "s-", label="Secante")
ax[2].set_xlabel("iteración $k$"); ax[2].set_ylabel(r"$|v_k-v^*|$"); ax[2].set_title("Error absoluto (aritmética de 80 dígitos)")
ax[2].legend(); ax[2].set_ylim(1e-62, 1)
plt.tight_layout(); plt.savefig("figuras/p4a_vdw.png", dpi=200); plt.close()

# =====================================================================
#                     PARTE B : sistema no lineal 2D
# =====================================================================
F = lambda X: [X[0] ** 3 - 3 * X[0] * X[1] ** 2 - 1, 3 * X[0] ** 2 * X[1] - X[1] ** 3]
J = lambda X: [[3 * X[0] ** 2 - 3 * X[1] ** 2, -6 * X[0] * X[1]],
               [6 * X[0] * X[1], 3 * X[0] ** 2 - 3 * X[1] ** 2]]
raices = [(1.0, 0.0), (-0.5, math.sqrt(3) / 2), (-0.5, -math.sqrt(3) / 2)]


def raiz_cercana(p):
    return min(range(3), key=lambda i: math.hypot(p[0] - raices[i][0], p[1] - raices[i][1]))


# ---- relajación: x' = x - w (c f1 - s f2) ; y' = y - w (s f1 + c f2) evaluadas con x' en la 2ª
def hacer_g(w, theta):
    c, s = math.cos(theta), math.sin(theta)
    g1 = lambda x, y: x - w * (c * F((x, y))[0] - s * F((x, y))[1])
    g2 = lambda xn, y: y - w * (s * F((xn, y))[0] + c * F((xn, y))[1])
    return g1, g2


def radio_espectral_GS(w, theta, raiz):
    """Radio espectral de la matriz de iteración (Gauss-Seidel) en la raíz, por regla de la cadena."""
    c, s = math.cos(theta), math.sin(theta)
    x, y = raiz
    Jm = J((x, y))
    g1x = 1 - w * (c * Jm[0][0] - s * Jm[1][0]); g1y = -w * (c * Jm[0][1] - s * Jm[1][1])
    # 2ª ecuación: y' = y - w (s f1(x',y) + c f2(x',y)),  x' depende de (x,y)
    fx = [Jm[0][0], Jm[1][0]]; fy = [Jm[0][1], Jm[1][1]]
    d1 = [w * 0, 0]
    dfdx = [fx[0] * g1x + fy[0] * 0, fx[1] * g1x + fy[1] * 0]
    dfdy = [fx[0] * g1y + fy[0], fx[1] * g1y + fy[1]]
    g2x = -w * (s * dfdx[0] + c * dfdx[1])
    g2y = 1 - w * (s * dfdy[0] + c * dfdy[1])
    tr, det = g1x + g2y, g1x * g2y - g1y * g2x
    disc = cmath.sqrt(tr * tr / 4 - det)
    return max(abs(tr / 2 + disc), abs(tr / 2 - disc))


W = 0.20
P("--- Relajación 2D ---")
resR = {}
for name, theta in (("theta=0", 0.0), ("theta=2pi/3", 2 * math.pi / 3), ("theta=4pi/3", 4 * math.pi / 3)):
    P(name, "  rho en raíces:", [round(radio_espectral_GS(W, theta, r), 4) for r in raices])
g1, g2 = hacer_g(W, 0.0)
inicio = (1.4, 0.4)
Pr = relajacion_2d(g1, g2, *inicio, tol=1e-12, max_iter=5000)
P("relajación theta=0 desde", inicio, "->", Pr[-1], " iter =", len(Pr) - 1)
resR["0"] = Pr
# segunda y tercera raíz con el ángulo apropiado
g1b, g2b = hacer_g(W, 2 * math.pi / 3)
Pr2 = relajacion_2d(g1b, g2b, -0.3, 0.6, tol=1e-12, max_iter=5000)
P("relajación theta=2pi/3 desde (-0.3,0.6) ->", Pr2[-1], " iter =", len(Pr2) - 1)
g1c, g2c = hacer_g(W, 4 * math.pi / 3)
Pr3 = relajacion_2d(g1c, g2c, -0.3, -0.6, tol=1e-12, max_iter=5000)
P("relajación theta=4pi/3 desde (-0.3,-0.6) ->", Pr3[-1], " iter =", len(Pr3) - 1)
# ¿qué pasa con theta=0 desde (-0.3,0.6)?
Pr4 = relajacion_2d(g1, g2, -0.3, 0.6, tol=1e-12, max_iter=300)
P("relajación theta=0 desde (-0.3,0.6) ->", Pr4[-1], " iter =", len(Pr4) - 1)

# barrido de omega para la raíz (1,0): rho = |1-3w|
ws = np.linspace(0.02, 0.7, 60)
rho_w = [radio_espectral_GS(w, 0.0, raices[0]) for w in ws]
its_w = []
for w in ws:
    g1w, g2w = hacer_g(w, 0.0)
    Pw = relajacion_2d(g1w, g2w, *inicio, tol=1e-10, max_iter=3000)
    ok = math.hypot(Pw[-1][0] - 1, Pw[-1][1]) < 1e-6
    its_w.append(len(Pw) - 1 if ok else np.nan)

with open("tablas/p4_relajacion.tex", "w") as fh:
    fh.write("\\begin{tabular}{r c c c}\n\\toprule\n$k$ & $x_k$ & $y_k$ & $\\|(x_k,y_k)-(1,0)\\|$\\\\\n\\midrule\n")
    for k, (x, y) in enumerate(Pr):
        if k < 8 or k in (10, 15, 20, 25, 30) or k == len(Pr) - 1:
            fh.write("%d & %.10f & %.10f & %.2e\\\\\n" % (k, x, y, math.hypot(x - 1, y)))
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---- Newton 2D
P("--- Newton 2D ---")
nw = {}
starts = {"A": (1.4, 0.4), "B": (-0.3, 0.6), "C": (-0.3, -0.6), "D": (0.5, 0.5)}
for nm, s in starts.items():
    Xs = newton_nd(F, J, s, tol=1e-14, max_iter=50)
    r = raices[raiz_cercana(Xs[-1])]
    nw[nm] = (Xs, r)
    P(nm, s, "->", Xs[-1], " raíz", r, " iteraciones", len(Xs) - 1)
with open("tablas/p4_newton2d.tex", "w") as fh:
    Xs, r = nw["A"]
    fh.write("\\begin{tabular}{r c c c c c}\n\\toprule\n$k$ & $x_k$ & $y_k$ & $\\|F(x_k)\\|$ & $e_k$ & $e_{k+1}/e_k^2$\\\\\n\\midrule\n")
    ek = [math.hypot(X[0] - r[0], X[1] - r[1]) for X in Xs]
    for k, X in enumerate(Xs):
        nf = norma2(F(X))
        q = "%.3f" % (ek[k + 1] / ek[k] ** 2) if k + 1 < len(Xs) and ek[k + 1] > 1e-15 else "--"
        fh.write("%d & %.12f & %.12f & %.2e & %.2e & %s\\\\\n" % (k, X[0], X[1], nf, ek[k], q))
    fh.write("\\bottomrule\n\\end{tabular}\n")
with open("tablas/p4_newton2d_otros.tex", "w") as fh:
    fh.write("\\begin{tabular}{c c c c c}\n\\toprule\nCaso & semilla $(x_0,y_0)$ & iteraciones & raíz alcanzada & $\\|F\\|$ final\\\\\n\\midrule\n")
    for nm, s in starts.items():
        Xs, r = nw[nm]
        fh.write("%s & $(%.1f,\\,%.1f)$ & %d & $(%.4f,\\,%.4f)$ & %.1e\\\\\n" %
                 (nm, s[0], s[1], len(Xs) - 1, Xs[-1][0], Xs[-1][1], norma2(F(Xs[-1]))))
    fh.write("\\bottomrule\n\\end{tabular}\n")

# Jacobiana singular en el origen
try:
    gauss_pivoteo(J((0.0, 0.0)), [1.0, 0.0])
except ZeroDivisionError as ex:
    P("Newton en (0,0): Jacobiana singular ->", ex)

# ---- Figuras Parte B
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
xx, yy = np.meshgrid(np.linspace(-1.6, 1.8, 500), np.linspace(-1.6, 1.6, 500))
ax[0].contour(xx, yy, xx ** 3 - 3 * xx * yy ** 2 - 1, [0], colors="C0")
ax[0].contour(xx, yy, 3 * xx ** 2 * yy - yy ** 3, [0], colors="C3")
ax[0].plot([r[0] for r in raices], [r[1] for r in raices], "ko", ms=8)
ax[0].plot([], [], "C0", label="$f_1=0$"); ax[0].plot([], [], "C3", label="$f_2=0$"); ax[0].legend()
ax[0].set_aspect("equal"); ax[0].set_title("Curvas de nivel $f_1=0$, $f_2=0$"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
# trayectorias
ax[1].plot([r[0] for r in raices], [r[1] for r in raices], "k*", ms=12)
Xs, r = nw["A"]
ax[1].plot([X[0] for X in Xs], [X[1] for X in Xs], "o-", label="Newton (A)")
ax[1].plot([p[0] for p in Pr[:60]], [p[1] for p in Pr[:60]], ".-", label="relajación $\\theta=0$")
for key, col in (("B", "C2"), ("C", "C4")):
    Xs2, _ = nw[key]
    ax[1].plot([X[0] for X in Xs2], [X[1] for X in Xs2], "o-", color=col, label="Newton (%s)" % key)
ax[1].plot([p[0] for p in Pr2[:80]], [p[1] for p in Pr2[:80]], ".:", color="C5", label=r"relajación $\theta=2\pi/3$")
ax[1].set_title("Trayectorias de las iteraciones"); ax[1].legend(fontsize=7); ax[1].set_xlabel("x"); ax[1].set_ylabel("y")
# convergencia
ek = [max(math.hypot(X[0] - 1, X[1]), 1e-17) for X in nw["A"][0]]
er = [max(math.hypot(p[0] - 1, p[1]), 1e-17) for p in Pr]
ax[2].semilogy(range(len(ek)), ek, "o-", label="Newton 2D")
ax[2].semilogy(range(len(er)), er, "s-", ms=3, label="relajación ($\\omega=0.2$)")
ax[2].semilogy(range(len(er)), [er[0] * (abs(1 - 3 * W)) ** k for k in range(len(er))], "k--", lw=.8, label=r"$\propto|1-3\omega|^k$")
ax[2].set_xlabel("iteración"); ax[2].set_ylabel("error euclídeo"); ax[2].set_title("Convergencia hacia $(1,0)$"); ax[2].legend()
plt.tight_layout(); plt.savefig("figuras/p4b_sistema.png", dpi=200); plt.close()

# cuencas de atracción (Newton 2D ≡ Newton complejo para z^3=1)
n = 500
xs_ = np.linspace(-1.6, 1.6, n); ys_ = np.linspace(-1.6, 1.6, n)
Z = xs_[None, :] + 1j * ys_[:, None]
its = np.zeros(Z.shape, dtype=int)
for _ in range(40):
    Z = Z - (Z ** 3 - 1) / (3 * Z ** 2 + 1e-300)
lab = np.zeros(Z.shape, dtype=int)
rc = [1, complex(-.5, math.sqrt(3) / 2), complex(-.5, -math.sqrt(3) / 2)]
d = np.stack([np.abs(Z - r) for r in rc])
lab = np.argmin(d, axis=0)
plt.figure(figsize=(6.4, 5.6))
plt.imshow(lab, extent=[-1.6, 1.6, -1.6, 1.6], origin="lower", cmap="Set2", vmin=0, vmax=7)
plt.plot([r.real for r in rc], [r.imag for r in rc], "k*", ms=12)
plt.title("Cuencas de atracción del método de Newton 2D"); plt.xlabel("$x_0$"); plt.ylabel("$y_0$")
plt.tight_layout(); plt.savefig("figuras/p4b_cuencas.png", dpi=200); plt.close()

# barrido en omega
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ws, rho_w, label=r"$\rho$ (raíz (1,0))"); ax[0].axhline(1, color="r", lw=.8)
ax[0].set_xlabel(r"$\omega$"); ax[0].set_ylabel("radio espectral"); ax[0].set_title("Estabilidad de la relajación"); ax[0].legend()
ax[1].plot(ws, its_w, "o-", ms=3); ax[1].set_xlabel(r"$\omega$"); ax[1].set_ylabel("iteraciones a $10^{-10}$")
ax[1].set_title(r"Iteraciones vs. $\omega$ (semilla (1.4, 0.4))")
plt.tight_layout(); plt.savefig("figuras/p4b_omega.png", dpi=200); plt.close()
P("rho minimo:", min(rho_w), " omega optimo ~", ws[int(np.argmin(rho_w))])
sal.close()
