# -*- coding: utf-8 -*-
"""Problema 3: ecuación de Kepler por relajación y bisección."""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metodos import *

os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
sal = open("resultados_p3.txt", "w")
def P(*a):
    s = " ".join(str(t) for t in a); print(s); sal.write(s + "\n")

M0, TOL = 1.5, 1e-6
f = lambda E, e, M: E - e * math.sin(E) - M
g = lambda E, e, M: M + e * math.sin(E)
dg = lambda E, e: e * math.cos(E)


def raiz_ref(e, M):
    """Raíz de referencia por bisección con tolerancia 1e-15."""
    c = biseccion(lambda E: f(E, e, M), 0.0, 2 * math.pi, tol=1e-15, max_iter=200)
    return c[-1]


# ---------- 1) y 2) relajación para e = 0.1 y e = 0.92 ----------
datos = {}
for e in (0.1, 0.92):
    Es = raiz_ref(e, M0)
    xs = relajacion_1d(lambda E: g(E, e, M0), M0, tol=TOL)
    datos[e] = (Es, xs)
    P("e=%.2f  E*=%.12f  |g'(E*)|=%.6f  iteraciones=%d  E_final=%.10f" %
      (e, Es, abs(dg(Es, e)), len(xs) - 1, xs[-1]))
    # razón de errores sucesivos
    er = [abs(x - Es) for x in xs]
    P("  razones e_{k+1}/e_k:", [round(er[i + 1] / er[i], 4) for i in range(min(8, len(er) - 1)) if er[i] > 0])
    with open("tablas/p3_relajacion_e%s.tex" % ("01" if e == 0.1 else "092"), "w") as fh:
        fh.write("\\begin{tabular}{r c c c c}\n\\toprule\n$k$ & $E_k$ [rad] & $|E_k-E_{k-1}|$ & $|E_k-E^*|$ & $|g'(E_k)|$\\\\\n\\midrule\n")
        for kk, x in enumerate(xs):
            d = "--" if kk == 0 else "%.2e" % abs(x - xs[kk - 1])
            fh.write("%d & %.8f & %s & %.2e & %.4f\\\\\n" % (kk, x, d, abs(x - Es), abs(dg(x, e))))
        fh.write("\\bottomrule\n\\end{tabular}\n")

# caso patológico ilustrativo: e=0.92 con M pequeño
Mp = 0.1
Esp = raiz_ref(0.92, Mp)
xsp = relajacion_1d(lambda E: g(E, 0.92, Mp), Mp, tol=TOL)
P("e=0.92, M=0.1: E*=%.10f |g'|=%.4f iteraciones=%d" % (Esp, abs(dg(Esp, 0.92)), len(xsp) - 1))

# ---------- bisección para los dos casos ----------
bis = {}
for e in (0.1, 0.92):
    cs = biseccion(lambda E: f(E, e, M0), 0.0, 2 * math.pi, tol=TOL)
    bis[e] = cs
    P("bisección e=%.2f: iteraciones=%d  E=%.10f" % (e, len(cs), cs[-1]))
    with open("tablas/p3_biseccion_e%s.tex" % ("01" if e == 0.1 else "092"), "w") as fh:
        Es = datos[e][0]
        fh.write("\\begin{tabular}{r c c}\n\\toprule\n$k$ & $E_k$ [rad] & $|E_k-E^*|$\\\\\n\\midrule\n")
        for kk, x in enumerate(cs, 1):
            if kk <= 6 or kk >= len(cs) - 2:
                fh.write("%d & %.8f & %.2e\\\\\n" % (kk, x, abs(x - Es)))
            elif kk == 7:
                fh.write("\\vdots & \\vdots & \\vdots\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------- 3) iteraciones vs excentricidad ----------
es = np.round(np.arange(0.10, 0.9501, 0.01), 2)
itR15, itR01, itB, rho15, rho01 = [], [], [], [], []
for e in es:
    e = float(e)
    itR15.append(len(relajacion_1d(lambda E: g(E, e, M0), M0, tol=TOL)) - 1)
    itR01.append(len(relajacion_1d(lambda E: g(E, e, 0.1), 0.1, tol=TOL)) - 1)
    itB.append(len(biseccion(lambda E: f(E, e, M0), 0.0, 2 * math.pi, tol=TOL)))
    rho15.append(abs(dg(raiz_ref(e, M0), e)))
    rho01.append(abs(dg(raiz_ref(e, 0.1), e)))
P("bisección: min/max iteraciones =", min(itB), max(itB))
P("relajación M=1.5: min/max =", min(itR15), max(itR15), " M=0.1: ", min(itR01), max(itR01))
sel = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.92, 0.95]
with open("tablas/p3_iteraciones.tex", "w") as fh:
    fh.write("\\begin{tabular}{c cc cc c}\n\\toprule\n & \\multicolumn{2}{c}{$M=1.5$ rad} & \\multicolumn{2}{c}{$M=0.1$ rad} & \\\\\n")
    fh.write("$e$ & relajación & $|g'(E^*)|$ & relajación & $|g'(E^*)|$ & bisección\\\\\n\\midrule\n")
    for s in sel:
        i = int(np.argmin(np.abs(es - s)))
        fh.write("%.2f & %d & %.3f & %d & %.3f & %d\\\\\n" % (es[i], itR15[i], rho15[i], itR01[i], rho01[i], itB[i]))
    fh.write("\\bottomrule\n\\end{tabular}\n")
P("tabla:", [(float(es[int(np.argmin(np.abs(es - s)))]), itR15[int(np.argmin(np.abs(es - s)))],
              itR01[int(np.argmin(np.abs(es - s)))], itB[int(np.argmin(np.abs(es - s)))]) for s in sel])
# cruce: primer e para el cual relajación (M=1.5) necesita más iteraciones que bisección
cruce = [float(es[i]) for i in range(len(es)) if itR15[i] > itB[i]]
P("relajación M=1.5 supera a bisección desde e =", cruce[0] if cruce else None)
cruce2 = [float(es[i]) for i in range(len(es)) if itR01[i] > itB[i]]
P("relajación M=0.1 supera a bisección desde e =", cruce2[0] if cruce2 else None)

# ================== FIGURAS ==================
# (a) f(E) y g(E) con cobweb
Ef = np.linspace(0, 2 * math.pi, 500)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, e in zip(axs, (0.1, 0.92)):
    Es, xs = datos[e]
    ax.plot(Ef, [g(E, e, M0) for E in Ef], label=r"$g(E)=M+e\sin E$")
    ax.plot(Ef, Ef, "k--", label=r"$y=E$")
    px, py = [xs[0]], [0.0]
    for xk in xs[:12]:
        gx = g(xk, e, M0)
        px += [xk, gx]; py += [gx, gx]
    # telaraña: (x0,x0)->(x0,g(x0))->(g(x0),g(x0)) ...
    cx, cy = [xs[0]], [xs[0]]
    for xk in xs[:12]:
        gx = g(xk, e, M0)
        cx += [xk, gx]; cy += [gx, gx]
    ax.plot(cx, cy, "r-", lw=1, label="iteración (telaraña)")
    ax.plot(Es, Es, "go", ms=8, label=r"$E^*=%.4f$" % Es)
    ax.set_xlim(0, 2 * math.pi); ax.set_ylim(0, 2 * math.pi)
    ax.set_title(r"$e=%.2f$,  $|g'(E^*)|=%.3f$,  %d iteraciones" % (e, abs(dg(Es, e)), len(xs) - 1))
    ax.set_xlabel("E [rad]"); ax.set_ylabel("g(E)"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig("figuras/p3_telarana.png", dpi=200); plt.close()

# (b) error vs iteración (semilog)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for e, c in ((0.1, "C0"), (0.92, "C3")):
    Es, xs = datos[e]
    er = [max(abs(x - Es), 1e-17) for x in xs]
    axs[0].semilogy(range(len(er)), er, "o-", color=c, label="relajación $e=%.2f$" % e)
    Esb = Es
    erb = [max(abs(x - Esb), 1e-17) for x in bis[e]]
    axs[0].semilogy(range(1, len(erb) + 1), erb, "s--", color=c, alpha=.5, ms=4, label="bisección $e=%.2f$" % e)
axs[0].axhline(TOL, color="k", lw=.7, ls=":")
axs[0].set_xlabel("iteración"); axs[0].set_ylabel(r"$|E_k-E^*|$"); axs[0].set_title("$M=1.5$ rad"); axs[0].legend(fontsize=8)
er = [max(abs(x - Esp), 1e-17) for x in xsp]
axs[1].semilogy(range(len(er)), er, "o-", color="C3", label="relajación $e=0.92$")
axs[1].semilogy(range(len(er)), [er[0] * (abs(dg(Esp, 0.92))) ** i for i in range(len(er))], "k--", lw=.8,
                label=r"$\propto|g'(E^*)|^k=%.3f^k$" % abs(dg(Esp, 0.92)))
axs[1].axhline(TOL, color="k", lw=.7, ls=":")
axs[1].set_xlabel("iteración"); axs[1].set_title("caso desfavorable: $M=0.1$ rad, $e=0.92$"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("figuras/p3_error.png", dpi=200); plt.close()

# (c) iteraciones vs e
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
axs[0].plot(es, itR15, "o-", ms=3, label="relajación ($M=1.5$)")
axs[0].plot(es, itR01, "^-", ms=3, label="relajación ($M=0.1$)")
axs[0].plot(es, itB, "s-", ms=3, label="bisección")
axs[0].set_xlabel("excentricidad $e$"); axs[0].set_ylabel(r"iteraciones para $\varepsilon=10^{-6}$")
axs[0].set_title("Coste de cada método"); axs[0].legend()
axs[1].plot(es, rho15, label=r"$|g'(E^*)|$, $M=1.5$"); axs[1].plot(es, rho01, label=r"$|g'(E^*)|$, $M=0.1$")
axs[1].plot(es, es, "k--", lw=.8, label=r"cota $e$")
axs[1].axhline(1, color="r", lw=.8)
axs[1].set_xlabel("excentricidad $e$"); axs[1].set_ylabel("factor de contracción"); axs[1].legend()
axs[1].set_title("Razón de convergencia asintótica")
plt.tight_layout(); plt.savefig("figuras/p3_iteraciones.png", dpi=200); plt.close()

# (d) órbita: posición para distintos M (aplicación)
fig, ax = plt.subplots(figsize=(6, 5))
for e, c in ((0.1, "C0"), (0.92, "C3")):
    th = np.linspace(0, 2 * math.pi, 400)
    a = 1.0; b = a * math.sqrt(1 - e * e)
    ax.plot(a * np.cos(th) - a * e, b * np.sin(th), color=c, alpha=.6, label="órbita $e=%.2f$" % e)
    Es = datos[e][0]
    ax.plot(a * math.cos(Es) - a * e, b * math.sin(Es), "o", color=c)
ax.plot(0, 0, "y*", ms=15, mec="k"); ax.set_aspect("equal")
ax.set_title(r"Posición para $M=1.5$ rad (foco en el origen)"); ax.legend(fontsize=8)
ax.set_xlabel("x / a"); ax.set_ylabel("y / a")
plt.tight_layout(); plt.savefig("figuras/p3_orbita.png", dpi=200); plt.close()
for e in (0.1, 0.92):
    Es = datos[e][0]
    P("e=%.2f  r/a=%.4f  anomalía verdadera nu=%.4f rad" % (
        e, 1 - e * math.cos(Es),
        2 * math.atan2(math.sqrt(1 + e) * math.sin(Es / 2), math.sqrt(1 - e) * math.cos(Es / 2))))
sal.close()
