# -*- coding: utf-8 -*-
"""Problema 1: perfil de temperatura estacionario en una barra 1D."""
import math, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metodos import *

os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})

# ---------------- datos del problema ----------------
L, k, TA, TB, N = 1.0, 45.0, 100.0, 20.0, 6
fuentes = {
    "q1": (lambda x: 5000.0 + 0 * x,            r"$q_1=5000$"),
    "q2": (lambda x: 10000.0 * x / L,           r"$q_2=10000\,x/L$"),
    "q3": (lambda x: 8000.0 * np.sin(np.pi * x / L), r"$q_3=8000\sin(\pi x/L)$"),
}


def malla(N):
    dx = L / (N + 1)
    return dx, np.array([(i + 1) * dx for i in range(N)])


def construir_A(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 2.0
        if i > 0: A[i, i - 1] = -1.0
        if i < N - 1: A[i, i + 1] = -1.0
    return A


def construir_b(q, N):
    dx, x = malla(N)
    b = np.array([dx ** 2 / k * q(xi) for xi in x])
    b[0] += TA; b[-1] += TB              # condiciones de frontera pasan al lado derecho
    return b


def T_exacta(nombre, x):
    lin = TA + (TB - TA) * x / L
    if nombre == "q1":
        return lin + 5000.0 / (2 * k) * x * (L - x)
    if nombre == "q2":
        c = 10000.0 / (k * L)
        return lin + c * (L ** 2 * x - x ** 3) / 6.0
    return lin + 8000.0 / k * (L / np.pi) ** 2 * np.sin(np.pi * x / L)


sal = open("resultados_p1.txt", "w")
def P(*a):
    s = " ".join(str(t) for t in a); print(s); sal.write(s + "\n")

dx, x = malla(N)
A = construir_A(N)
P("dx =", dx, " dx^2/k =", dx ** 2 / k)
P("x_i =", x)
b1 = construir_b(fuentes["q1"][0], N)
P("b (q1) =", b1)

# ---------- 2) Gauss con pivoteo parcial ----------
Tg, info = gauss_pivoteo(A, b1, guardar_etapas=True)
P("Intercambios de filas en Gauss:", info["intercambios"])
P("U diag =", np.diag(info["U"]))
P("c =", info["c"])
P("T Gauss (q1) =", Tg)
P("residuo ||A T - b|| =", norma2(matvec(A, Tg) - b1))

# Demostración de la necesidad del pivoteo: 1e-17 x1 + x2 = 1 ; x1 + x2 = 2
eps = 1e-17
Ad = [[eps, 1.0], [1.0, 1.0]]; bd = [1.0, 2.0]
x_sin, _ = gauss_pivoteo(Ad, bd, pivoteo=False)
x_con, inf2 = gauss_pivoteo(Ad, bd, pivoteo=True)
P("Demo pivoteo: sin pivoteo", x_sin, " con pivoteo", x_con, inf2["intercambios"])

# ---------- 3) LU / Thomas ----------
a_sub = -np.ones(N - 1); b_dia = 2 * np.ones(N); c_sup = -np.ones(N - 1)
l, u = thomas_factor(a_sub, b_dia, c_sup)
P("l_i =", l); P("u_i =", u)
Lm, Um = lu_doolittle(A)                  # LU general de Doolittle
P("Doolittle == Thomas ?  ", max(abs(Lm[i + 1, i] - l[i]) for i in range(N - 1)),
  max(abs(Um[i, i] - u[i]) for i in range(N)))
P("||L U - A||_max =", np.max(np.abs(matmul(Lm, Um) - A)))

resT, resG, errs = {}, {}, {}
for nom, (q, lab) in fuentes.items():
    b = construir_b(q, N)
    resT[nom] = thomas_resolver(l, u, c_sup, b)          # SIN refactorizar
    resG[nom], _ = gauss_pivoteo(A, b)
    ex = T_exacta(nom, x)
    errs[nom] = (np.max(np.abs(resT[nom] - resG[nom])), np.max(np.abs(resT[nom] - ex)))
    P(nom, "T =", resT[nom])
    P(nom, "max|Thomas-Gauss| = %.3e   max|T_num-T_exacta| = %.3e" % errs[nom])

# tabla de resultados
with open("tablas/p1_temperaturas.tex", "w") as f:
    f.write("\\begin{tabular}{c c cc cc cc}\n\\toprule\n")
    f.write(" & & \\multicolumn{2}{c}{$q_1$} & \\multicolumn{2}{c}{$q_2$} & \\multicolumn{2}{c}{$q_3$}\\\\\n")
    f.write("$i$ & $x_i$ [m] & Thomas & Exacta & Thomas & Exacta & Thomas & Exacta\\\\\n\\midrule\n")
    for i in range(N):
        row = "%d & %.4f" % (i + 1, x[i])
        for nom in ("q1", "q2", "q3"):
            row += " & %.4f & %.4f" % (resT[nom][i], T_exacta(nom, x[i]))
        f.write(row + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("tablas/p1_gauss_thomas.tex", "w") as f:
    f.write("\\begin{tabular}{c ccc ccc}\n\\toprule\n")
    f.write("$i$ & $l_i$ & $u_i$ & $u_i=(i+1)/i$ & $b^{(1)}_i$ & $c_i$ (Gauss) & $T_i^{Gauss}$\\\\\n\\midrule\n")
    for i in range(N):
        li = "%.6f" % l[i - 1] if i > 0 else "--"
        f.write("%d & %s & %.6f & %.6f & %.5f & %.5f & %.5f\\\\\n" %
                (i + 1, li, u[i], (i + 2) / (i + 1), b1[i], info["c"][i], Tg[i]))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- 4) Inversa ----------
Ainv = inversa_por_columnas(l, u, c_sup)
P("||A Ainv - I||_max =", np.max(np.abs(matmul(A, Ainv) - np.eye(N))))
exacta_inv = np.array([[min(i, j) * (N + 1 - max(i, j)) / (N + 1)
                        for j in range(1, N + 1)] for i in range(1, N + 1)])
P("max|Ainv - formula| =", np.max(np.abs(Ainv - exacta_inv)))
P("7*Ainv =\n", np.round(7 * Ainv, 10))
# Ainv con Gauss (comprobación cruzada)
Ainv_g = np.zeros((N, N))
for j in range(N):
    e = np.zeros(N); e[j] = 1
    Ainv_g[:, j], _ = gauss_pivoteo(A, e)
P("max|Ainv_Thomas - Ainv_Gauss| =", np.max(np.abs(Ainv - Ainv_g)))
# T = A^{-1} b
Tinv = matvec(Ainv, b1)
P("max|Ainv b - T| =", np.max(np.abs(Tinv - Tg)))
# Interpretación: respuesta a fuente puntual: T_i = sum_j G_ij (dx^2 q_j/k) + T homogénea
with open("tablas/p1_inversa.tex", "w") as f:
    f.write("\\[\n7\\,A^{-1}=\\begin{pmatrix}\n")
    for i in range(N):
        f.write(" & ".join("%d" % round(7 * Ainv[i, j]) for j in range(N)) + "\\\\\n")
    f.write("\\end{pmatrix}\n\\]\n")

# ---------- coste y convergencia de malla ----------
Ns = [25, 50, 100, 200, 300, 400]
tG, tT = [], []
for n in Ns:
    An = construir_A(n); bn = construir_b(fuentes["q3"][0], n)
    t0 = time.perf_counter(); gauss_pivoteo(An, bn); tG.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    ln, un = thomas_factor(-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1))
    thomas_resolver(ln, un, -np.ones(n - 1), bn); tT.append(time.perf_counter() - t0)
P("tiempos Gauss", tG); P("tiempos Thomas", tT)

Nh = [6, 12, 25, 50, 100, 200, 400]
eh = []
for n in Nh:
    dxn, xn = malla(n)
    ln, un = thomas_factor(-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1))
    Tn = thomas_resolver(ln, un, -np.ones(n - 1), construir_b(fuentes["q3"][0], n))
    eh.append(np.max(np.abs(Tn - T_exacta("q3", xn))))
P("error max q3 vs N:", list(zip(Nh, eh)))
ordenes = [math.log(eh[i] / eh[i + 1]) / math.log((Nh[i + 1] + 1) / (Nh[i] + 1)) for i in range(len(Nh) - 1)]
P("orden observado:", ordenes)

# ======================= FIGURAS =======================
xf = np.linspace(0, L, 400)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
cols = {"q1": "C0", "q2": "C1", "q3": "C2"}
for nom, (q, lab) in fuentes.items():
    ax[0].plot(xf, q(xf), color=cols[nom], label=lab)
    xx = np.concatenate(([0], x, [L])); TT = np.concatenate(([TA], resT[nom], [TB]))
    ax[1].plot(xf, T_exacta(nom, xf), "-", color=cols[nom], alpha=.6)
    ax[1].plot(xx, TT, "o", color=cols[nom], label=lab)
ax[0].set_xlabel("x [m]"); ax[0].set_ylabel(r"$q(x)$ [W/m$^3$]"); ax[0].set_title("Perfiles de fuente"); ax[0].legend()
ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("T [°C]"); ax[1].set_title("Temperatura: numérica (puntos) vs. exacta (línea)")
ax[1].legend()
plt.tight_layout(); plt.savefig("figuras/p1_perfiles.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
im = ax[0].imshow(Ainv, cmap="viridis")
for i in range(N):
    for j in range(N):
        ax[0].text(j, i, "%.2f" % Ainv[i, j], ha="center", va="center", color="w", fontsize=8)
ax[0].set_xticks(range(N)); ax[0].set_yticks(range(N))
ax[0].set_xticklabels(range(1, N + 1)); ax[0].set_yticklabels(range(1, N + 1))
ax[0].set_xlabel("nodo fuente $j$"); ax[0].set_ylabel("nodo respuesta $i$"); ax[0].grid(False)
ax[0].set_title(r"Matriz $A^{-1}$ (función de Green discreta)")
plt.colorbar(im, ax=ax[0], fraction=.046)
xx = np.concatenate(([0], x, [L]))
for j in range(N):
    ax[1].plot(xx, np.concatenate(([0], Ainv[:, j], [0])), "o-", label="$j=%d$" % (j + 1))
ax[1].set_xlabel("x [m]"); ax[1].set_ylabel(r"$G_{ij}$ (columna $j$)")
ax[1].set_title("Respuesta a una fuente unitaria en el nodo $j$"); ax[1].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig("figuras/p1_inversa.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].loglog(Ns, tG, "o-", label="Gauss (pivoteo)"); ax[0].loglog(Ns, tT, "s-", label="Thomas (LU)")
ax[0].loglog(Ns, [tG[-1] * (n / Ns[-1]) ** 3 for n in Ns], "k--", lw=.8, label=r"$\propto N^3$")
ax[0].loglog(Ns, [tT[-1] * (n / Ns[-1]) for n in Ns], "k:", lw=.8, label=r"$\propto N$")
ax[0].set_xlabel("N"); ax[0].set_ylabel("tiempo [s]"); ax[0].set_title("Coste computacional"); ax[0].legend()
ax[1].loglog(Nh, eh, "o-", label="error máx. $q_3$")
ax[1].loglog(Nh, [eh[0] * ((Nh[0] + 1) / (n + 1)) ** 2 for n in Nh], "k--", lw=.8, label=r"$\propto\Delta x^2$")
ax[1].set_xlabel("N"); ax[1].set_ylabel(r"$\max_i|T_i-T(x_i)|$"); ax[1].set_title("Convergencia de la discretización"); ax[1].legend()
nm = ["q1", "q2", "q3"]
w = .35; idx = np.arange(3)
ax[2].bar(idx - w / 2, [errs[n][0] for n in nm], w, label="|Thomas − Gauss|")
ax[2].bar(idx + w / 2, [errs[n][1] for n in nm], w, label="|Numérica − Exacta|")
ax[2].set_yscale("log"); ax[2].set_xticks(idx); ax[2].set_xticklabels(nm); ax[2].legend(fontsize=9)
ax[2].set_title("Errores para $N=6$")
plt.tight_layout(); plt.savefig("figuras/p1_coste_error.png", dpi=200); plt.close()

with open("tablas/p1_coste.tex", "w") as f:
    f.write("\\begin{tabular}{r cc c}\n\\toprule\n$N$ & $t_{Gauss}$ [ms] & $t_{Thomas}$ [ms] & $t_G/t_T$\\\\\n\\midrule\n")
    for n, a_, b_ in zip(Ns, tG, tT):
        f.write("%d & %.2f & %.3f & %.0f\\\\\n" % (n, 1e3 * a_, 1e3 * b_, a_ / b_))
    f.write("\\bottomrule\n\\end{tabular}\n")
with open("tablas/p1_malla.tex", "w") as f:
    f.write("\\begin{tabular}{r c c}\n\\toprule\n$N$ & $\\max_i|T_i-T(x_i)|$ & orden observado\\\\\n\\midrule\n")
    for i, n in enumerate(Nh):
        o = "%.2f" % ordenes[i - 1] if i > 0 else "--"
        f.write("%d & %.3e & %s\\\\\n" % (n, eh[i], o))
    f.write("\\bottomrule\n\\end{tabular}\n")
sal.close()
