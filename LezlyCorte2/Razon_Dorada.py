# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:54:26 2026

@author: lezly
"""

"""
Búsqueda de la razón dorada (Golden-Section Search) aplicada a la física
=======================================================================
Problema: encontrar la distancia de equilibrio r0 entre dos átomos de argón
que interactúan mediante el potencial de Lennard-Jones:

    U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]

El equilibrio es el MÍNIMO de U(r). Solución analítica: r0 = 2^(1/6)*sigma.

Uso:  python busqueda_razon_dorada_lj.py
Genera las figuras en la carpeta Plan_Dia_U/ (usadas por las diapositivas).
"""
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Parámetros físicos (argón) ----
EPS = 0.0103      # eV  (profundidad del pozo)
SIGMA = 3.40      # Å   (distancia donde U = 0)

def U(r):
    """Potencial de Lennard-Jones en eV, r en Å."""
    return 4.0 * EPS * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

# ---- Razón dorada ----
INVPHI = (math.sqrt(5) - 1) / 2      # 1/phi = 0.6180339...

def golden_section_search(f, a, b, tol=1e-5, max_iter=200):
    """Minimiza f (unimodal) en [a, b]. Devuelve (x_min, historial)."""
    c = b - INVPHI * (b - a)
    d = a + INVPHI * (b - a)
    fc, fd = f(c), f(d)
    hist = [(a, b, c, d)]
    it = 0
    while (b - a) > tol and it < max_iter:
        if fc < fd:                    # el mínimo está en [a, d]
            b, d, fd = d, c, fc        # c se reutiliza como nuevo d
            c = b - INVPHI * (b - a)
            fc = f(c)
        else:                          # el mínimo está en [c, b]
            a, c, fc = c, d, fd        # d se reutiliza como nuevo c
            d = a + INVPHI * (b - a)
            fd = f(d)
        it += 1
        hist.append((a, b, c, d))
    return (a + b) / 2, hist

def main():
    a0, b0 = 3.0, 6.0                     # intervalo inicial en Å
    r_num, hist = golden_section_search(U, a0, b0, tol=1e-5)
    r_exact = 2 ** (1 / 6) * SIGMA

    print(f"Intervalo inicial : [{a0}, {b0}] Å")
    print(f"Iteraciones       : {len(hist) - 1}")
    print(f"r0 numérico       : {r_num:.6f} Å")
    print(f"r0 analítico      : {r_exact:.6f} Å")
    print(f"Error absoluto    : {abs(r_num - r_exact):.2e} Å")
    print(f"U(r0)             : {U(r_num):.6f} eV  (teórico: {-EPS:.6f} eV)")
    print("\n it |    a       |    b       |  ancho")
    for k in range(0, min(len(hist), 11)):
        a, b, _, _ = hist[k]
        print(f"{k:3d} | {a:10.6f} | {b:10.6f} | {b - a:.6f}")

    # ---------- Figuras ----------
    out = "Plan_Dia_U"
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 12})
    red = "#9B0000"

    # 1) Potencial
    r = np.linspace(3.2, 7.5, 600)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(r, U(r) * 1000, color=red, lw=2.2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(r_exact, U(r_exact) * 1000, "o", color="black", ms=8)
    ax.annotate(r"$r_0=2^{1/6}\sigma\approx3.816$ Å",
                (r_exact, U(r_exact) * 1000), (4.6, -6), fontsize=11,
                arrowprops=dict(arrowstyle="->"))
    ax.set_ylim(-14, 25); ax.set_xlabel("r (Å)"); ax.set_ylabel("U(r) (meV)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{out}/fig_lj_potential.png", dpi=200); plt.close(fig)

    # 2) Iteraciones sobre la curva
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(r, U(r) * 1000, color="lightgray", lw=2)
    cols = ["#1f77b4", "#2ca02c", "#ff7f0e", red]
    for k, col in zip([0, 1, 2, 4], cols):
        a, b, c, d = hist[k]
        y = -12.5 - 2.2 * [0, 1, 2, 4].index(k)
        ax.plot([a, b], [y, y], color=col, lw=5, solid_capstyle="butt")
        ax.plot([c, d], [y, y], "o", color="white", mec=col, mew=2, ms=7)
        ax.text(b + 0.08, y - 0.4, f"k={k}", color=col, fontsize=10)
    ax.axvline(r_exact, color="black", ls="--", lw=1)
    ax.set_ylim(-22, 12); ax.set_xlim(2.9, 7.5)
    ax.set_xlabel("r (Å)"); ax.set_ylabel("U(r) (meV)")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{out}/fig_iterations.png", dpi=200); plt.close(fig)

    # 3) Convergencia
    widths = [b - a for a, b, _, _ in hist]
    ks = np.arange(len(widths))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(ks, widths, "o-", color=red, ms=4, label="Golden-section")
    ax.semilogy(ks, (b0 - a0) * INVPHI ** ks, "k--", lw=1,
                label=r"$(b_0-a_0)\,\varphi^{-k}$")
    ax.set_xlabel("k"); ax.set_ylabel("b - a (Å)")
    ax.grid(alpha=.3, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{out}/fig_convergence.png", dpi=200); plt.close(fig)

    # 4) Proporción áurea
    fig, ax = plt.subplots(figsize=(6, 1.8))
    L = 1.0; x = INVPHI
    ax.plot([0, L], [0, 0], color="black", lw=6, solid_capstyle="butt")
    ax.plot([0, x], [0.0, 0.0], color=red, lw=6, solid_capstyle="butt")
    ax.plot([x, x], [-0.15, 0.15], color="black", lw=1.5)
    ax.text(x / 2, 0.22, r"$a$", ha="center", fontsize=14)
    ax.text((x + 1) / 2, 0.22, r"$b$", ha="center", fontsize=14)
    ax.text(0.5, -0.45, r"$\dfrac{a+b}{a}=\dfrac{a}{b}=\varphi$",
            ha="center", fontsize=14)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.8, 0.5); ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_golden_ratio.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    main()