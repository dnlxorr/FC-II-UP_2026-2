# -*- coding: utf-8 -*-
"""Problema 2: modos normales de vibración de N=4 masas acopladas."""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metodos import *

os.makedirs("figuras", exist_ok=True); os.makedirs("tablas", exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})
sal = open("resultados_p2.txt", "w")
def P(*a):
    s = " ".join(str(t) for t in a); print(s); sal.write(s + "\n")

N, m, k = 4, 0.5, 200.0
K = np.zeros((N, N)); Minv = np.zeros((N, N))
for i in range(N):
    K[i, i] = 2 * k
    Minv[i, i] = 1.0 / m
    if i > 0: K[i, i - 1] = -k
    if i < N - 1: K[i, i + 1] = -k
A = matmul(Minv, K)                       # A = M^{-1} K
P("K =\n", K); P("A = M^-1 K =\n", A)

# ---------- Jacobi ----------
lamJ, VJ, histJ = jacobi_eigen(A)
P("Jacobi: barridos =", len(histJ) - 1, " off-norm =", histJ)
P("lambda (Jacobi) =", lamJ)
for j in range(N):                        # normalización de signo: 1ª componente positiva
    if VJ[0, j] < 0: VJ[:, j] *= -1
omega = np.sqrt(lamJ)
P("omega =", omega, "\nf [Hz] =", omega / (2 * math.pi), "\nT [s] =", 2 * math.pi / omega)
P("V =\n", VJ)

# ---------- solución analítica ----------
lamA = np.array([4 * k / m * math.sin(j * math.pi / (2 * (N + 1))) ** 2 for j in range(1, N + 1)])
VA = np.array([[math.sqrt(2 / (N + 1)) * math.sin(i * j * math.pi / (N + 1)) for j in range(1, N + 1)]
               for i in range(1, N + 1)])
P("lambda analítico =", lamA, "\nmax|dif| =", np.max(np.abs(lamA - lamJ)))
P("|<v_J, v_A>| =", [abs(sum(VJ[i, j] * VA[i, j] for i in range(N))) for j in range(N)])
res = [norma2(matvec(A, VJ[:, j]) - lamJ[j] * VJ[:, j]) for j in range(N)]
P("residuos ||Av-lam v|| =", res)
ortho = matmul(VJ.T, VJ)
P("max|V^T V - I| =", np.max(np.abs(ortho - np.eye(N))))

# ---------- potencia + deflación ----------
lamP, VP, itP, histP = potencia_con_deflacion(A, [1.0, 2.0, 3.0, 5.0])
P("Potencia+deflación lambda =", lamP, "iteraciones =", itP)
P("max|lamP - lamJ| =", max(abs(lamP[i] - lamJ[N - 1 - i]) for i in range(N)))
for j in range(N):
    if VP[j][0] < 0: VP[j] = -VP[j]
P("|<vP,vJ>| =", [abs(sum(VP[j][i] * VJ[i, N - 1 - j] for i in range(N))) for j in range(N)])

# ---------- tablas ----------
with open("tablas/p2_modos.tex", "w") as f:
    f.write("\\begin{tabular}{c ccc cc c}\n\\toprule\n")
    f.write("Modo & $\\lambda_j$ [s$^{-2}$] (Jacobi) & $\\lambda_j$ analítico & error abs. & $\\omega_j$ [rad/s] & $f_j$ [Hz] & $T_j$ [ms]\\\\\n\\midrule\n")
    for j in range(N):
        f.write("%d & %.6f & %.6f & %.1e & %.5f & %.4f & %.3f\\\\\n" %
                (j + 1, lamJ[j], lamA[j], abs(lamJ[j] - lamA[j]), omega[j], omega[j] / (2 * math.pi),
                 1e3 * 2 * math.pi / omega[j]))
    f.write("\\bottomrule\n\\end{tabular}\n")
with open("tablas/p2_vectores.tex", "w") as f:
    f.write("\\begin{tabular}{c cccc}\n\\toprule\n Modo & $v_{j,1}$ & $v_{j,2}$ & $v_{j,3}$ & $v_{j,4}$\\\\\n\\midrule\n")
    for j in range(N):
        f.write("%d & " % (j + 1) + " & ".join("%+.6f" % VJ[i, j] for i in range(N)) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open("tablas/p2_potencia.tex", "w") as f:
    f.write("\\begin{tabular}{c c c c c}\n\\toprule\n Paso & $\\lambda$ (potencia) & $\\lambda$ (Jacobi) & iteraciones & $|\\Delta\\lambda|$\\\\\n\\midrule\n")
    for j in range(N):
        f.write("%d & %.8f & %.8f & %d & %.1e\\\\\n" % (j + 1, lamP[j], lamJ[N - 1 - j], itP[j],
                                                       abs(lamP[j] - lamJ[N - 1 - j])))
    f.write("\\bottomrule\n\\end{tabular}\n")
with open("tablas/p2_jacobi_hist.tex", "w") as f:
    f.write("\\begin{tabular}{c c}\n\\toprule\n Barrido & $\\lVert A_{\\text{fuera de la diagonal}}\\rVert_F$\\\\\n\\midrule\n")
    for s, h in enumerate(histJ):
        f.write("%d & %.3e\\\\\n" % (s, h))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ================= FIGURAS =================
xs = np.arange(0, N + 2)                    # paredes en 0 y N+1
xf = np.linspace(0, N + 1, 300)
fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for j, ax in enumerate(axs.ravel()):
    y = np.concatenate(([0], VJ[:, j], [0]))
    ax.axhline(0, color="k", lw=.6)
    ax.plot(xf, math.sqrt(2 / (N + 1)) * np.sin((j + 1) * math.pi * xf / (N + 1)) * np.sign(VA[0, j]) * np.sign(VJ[0, j]) ** 0,
            "--", color="gray", label="onda estacionaria analítica")
    ax.plot(xs, y, "o-", color="C%d" % j, ms=9, label="eigenvector calculado (Jacobi)")
    ax.plot([0, N + 1], [0, 0], "ks", ms=10)
    ax.set_title(r"Modo %d:  $\omega_%d=%.2f$ rad/s,  $f_%d=%.2f$ Hz" % (j + 1, j + 1, omega[j], j + 1, omega[j] / (2 * math.pi)))
    ax.set_ylabel("amplitud normalizada")
    if j >= 2: ax.set_xlabel("posición de la masa $i$ (0 y 5 = paredes)")
    if j == 0: ax.legend(fontsize=8, loc="lower center")
plt.tight_layout(); plt.savefig("figuras/p2_modos.png", dpi=200); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
ax[0].semilogy(range(len(histJ)), [max(h, 1e-17) for h in histJ], "o-")
ax[0].set_xlabel("barrido de Jacobi"); ax[0].set_ylabel(r"$\|A_{\rm fuera\ diag}\|_F$"); ax[0].set_title("Convergencia de Jacobi")
for j in range(N):
    h = histP[j]
    err = [max(abs(v - lamP[j]), 1e-17) for v in h]
    ax[1].semilogy(range(1, len(h) + 1), err, label="paso %d ($\\lambda=%.1f$)" % (j + 1, lamP[j]))
ax[1].set_xlabel("iteración"); ax[1].set_ylabel(r"$|\lambda^{(n)}-\lambda|$"); ax[1].set_title("Potencia con deflación"); ax[1].legend(fontsize=8)
ax[2].bar(np.arange(1, N + 1) - .2, omega, .4, label=r"$\omega_j$ [rad/s]")
ax[2].bar(np.arange(1, N + 1) + .2, omega / (2 * math.pi), .4, label=r"$f_j$ [Hz]")
ax[2].set_xlabel("modo"); ax[2].set_title("Espectro de frecuencias"); ax[2].legend(); ax[2].set_xticks(range(1, N + 1))
plt.tight_layout(); plt.savefig("figuras/p2_convergencia.png", dpi=200); plt.close()

# Movimiento: superposición de modos (condición inicial: masa 1 desplazada)
t = np.linspace(0, 0.5, 600)
x0 = np.array([1.0, 0, 0, 0])
coef = [sum(VJ[i, j] * x0[i] for i in range(N)) for j in range(N)]   # v_j^T x0 (M=mI)
fig, ax = plt.subplots(figsize=(9, 4))
for i in range(N):
    xi = sum(coef[j] * VJ[i, j] * np.cos(omega[j] * t) for j in range(N))
    ax.plot(t, xi, label="masa %d" % (i + 1))
ax.set_xlabel("t [s]"); ax.set_ylabel("desplazamiento [u.a.]"); ax.set_title("Superposición de modos: masa 1 desplazada en $t=0$")
ax.legend(ncol=4, fontsize=8)
plt.tight_layout(); plt.savefig("figuras/p2_dinamica.png", dpi=200); plt.close()
P("coeficientes modales c_j =", coef)
sal.close()
