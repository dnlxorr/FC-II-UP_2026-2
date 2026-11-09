"""
Física Computacional - Lennard-Jones (12-6) y serie de Fourier periodizada
Parte A: bisección, búsqueda dorada, diferencias finitas, puntos de retorno
Parte B: coeficientes por Simpson 1/3, reconstrucción, Gibbs y error
Solo numpy (arreglos) y matplotlib (gráficas).
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parámetros
e = 1
s = 1

def potencial(x):
    return e * ((s / x)**12 - (s / x)**6)

def dpotencial(x):                     # derivada analítica (solo para Newton)
    return e * (-12 * s**12 / x**13 + 6 * s**6 / x**7)


# ---------------------------------------------------------------- algoritmos
def biseccion(f, a, b, tol=1e-8, itmax=10000):
    """Devuelve (raíz, iteraciones, evaluaciones de f)."""
    fa, fb = f(a), f(b)
    nev = 2
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a, b]")
    for i in range(1, itmax + 1):
        c = (a + b) / 2
        fc = f(c); nev += 1
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
        if (b - a) / 2 < tol:
            break
    return (a + b) / 2, i, nev


def newton(f, df, x, tol=1e-8, itmax=100):
    nev = 0
    for i in range(1, itmax + 1):
        xn = x - f(x) / df(x); nev += 2
        if abs(xn - x) < tol:
            return xn, i, nev
        x = xn
    return x, itmax, nev


def dorada(f, a, b, tol=1e-10):
    """Búsqueda dorada: mínimo de f en [a, b]."""
    r = (np.sqrt(5) - 1) / 2
    c, d = b - r * (b - a), a + r * (b - a)
    fc, fd = f(c), f(d)
    it = 0
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - r * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + r * (b - a); fd = f(d)
        it += 1
    return (a + b) / 2, it


def fuerza(x, h=1e-5):
    """F = -dV/dx por diferencias finitas centradas."""
    return -(potencial(x + h) - potencial(x - h)) / (2 * h)


def simpson(y, h):
    """Simpson 1/3 compuesto (n subintervalos, n par)."""
    assert (len(y) - 1) % 2 == 0
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


# =========================================================== PARTE A
print("=" * 55, "\nPARTE A\n", "=" * 55, sep="")

# A1. Punto de corte
x0, it, nev = biseccion(potencial, 0.8, 1.5)
print(f"x0   = {x0:.8f}  (exacto 1)  [{it} it, {nev} evals]")

# A2. Mínimo (dorada)
xmin, itd = dorada(potencial, 1.0, 2.0)
Vmin = potencial(xmin)
print(f"xmin = {xmin:.8f}  (exacto {2**(1/6):.8f})  [{itd} it]")
print(f"Vmin = {Vmin:.8f}  (exacto -0.25)")

# A3. Fuerza en xmin (y chequeo en x=1)
print(f"F(xmin) = {fuerza(xmin):.3e}   F(1.0) = {fuerza(1.0):.6f} (analítica 6)")

# A4. Puntos de retorno
def puntos_retorno(E):
    if E < Vmin:
        return None                      # E por debajo del fondo del pozo
    g = lambda x: potencial(x) - E
    x1 = biseccion(g, 0.8, xmin)[0]
    x2 = biseccion(g, xmin, 50.0)[0]
    return x1, x2

for E in (-0.5, -0.2):
    pr = puntos_retorno(E)
    if pr is None:
        print(f"E = {E}: sin puntos de retorno (Vmin = {Vmin:.2f} > E)")
    else:
        print(f"E = {E}: x1 = {pr[0]:.6f}, x2 = {pr[1]:.6f}")

# Discusión 1: bisección vs Newton
_, ib, nb = biseccion(potencial, 0.8, 1.5, tol=1e-10)
_, inw, nn = newton(potencial, dpotencial, 0.9, tol=1e-10)
print(f"\nBisección: {ib} it / {nb} evals | Newton: {inw} it / {nn} evals")

# Figura 1
xs = np.linspace(0.9, 3.5, 1000)
E = -0.2
x1, x2 = puntos_retorno(E)
plt.figure(figsize=(7, 4.5))
plt.plot(xs, potencial(xs), label="V(x)")
plt.axhline(0, color="gray", lw=0.6)
plt.plot(x0, 0, "go", label=f"$x_0$ = {x0:.3f}")
plt.plot(xmin, Vmin, "ro", label=f"$x_{{min}}$ = {xmin:.4f}")
plt.axhline(E, color="orange", ls="--", lw=0.8, label=f"E = {E}")
plt.plot([x1, x2], [E, E], "s", color="orange", label=f"$x_1$={x1:.3f}, $x_2$={x2:.3f}")
plt.ylim(-0.4, 1.0); plt.xlabel("x"); plt.ylabel("V(x)")
plt.title("Potencial de Lennard-Jones (reducido)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig1_potencial.pdf"); plt.savefig("fig1_potencial.png", dpi=200)

# =========================================================== PARTE B
print("\n" + "=" * 55, "\nPARTE B\n", "=" * 55, sep="")

xa, L = 0.8, 3.0
NMAX = 50
M = 6000                                  # subintervalos de Simpson (par)
xg = np.linspace(xa, xa + L, M + 1)
hg = L / M
Vg = potencial(xg)

an = np.zeros(NMAX + 1)
bn = np.zeros(NMAX + 1)
for n in range(NMAX + 1):
    k = 2 * np.pi * n / L
    an[n] = 2 / L * simpson(Vg * np.cos(k * xg), hg)
    bn[n] = 2 / L * simpson(Vg * np.sin(k * xg), hg)

print(f"{'n':>3} {'a_n':>12} {'b_n':>12}")
for n in range(21):
    print(f"{n:>3} {an[n]:>12.6f} {bn[n]:>12.6f}")

def Vp(x, N):
    out = np.full_like(x, an[0] / 2)
    for n in range(1, N + 1):
        k = 2 * np.pi * n / L
        out += an[n] * np.cos(k * x) + bn[n] * np.sin(k * x)
    return out

# Figura 2: espectro
n_ = np.arange(1, 21)
plt.figure(figsize=(7, 4))
plt.bar(n_ - 0.2, np.abs(an[1:21]), 0.4, label="$|a_n|$")
plt.bar(n_ + 0.2, np.abs(bn[1:21]), 0.4, label="$|b_n|$")
plt.yscale("log"); plt.xlabel("n"); plt.ylabel("amplitud")
plt.title("Espectro de Fourier"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_espectro.pdf"); plt.savefig("fig2_espectro.png", dpi=200)

# Figura 3: reconstrucciones
xf = np.linspace(xa, xa + L, 3000)
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, N in zip(axs.ravel(), (5, 10, 20, 50)):
    ax.plot(xf, potencial(xf), "k", lw=1, label="V(x)")
    ax.plot(xf, Vp(xf, N), "r", lw=0.9, label=f"Fourier N={N}")
    ax.set_ylim(-1, 14); ax.grid(alpha=0.3); ax.legend()
axs[1, 0].set_xlabel("x"); axs[1, 1].set_xlabel("x")
fig.suptitle("Reconstrucción de Fourier")
plt.tight_layout()
plt.savefig("fig3_reconstruccion.pdf"); plt.savefig("fig3_reconstruccion.png", dpi=200)

# Gibbs y error
salto = potencial(xa) - potencial(xa + L)
print(f"\nSalto en la periodización = {salto:.4f}")
print(f"Valor al que converge la serie en el borde = {(potencial(xa) + potencial(xa + L)) / 2:.4f}")
print(f"{'N':>4} {'err máx total':>14} {'err máx [1.0,3.5]':>18} {'sobreimpulso/salto':>19}")
for N in (5, 10, 20, 50):
    rec = Vp(xf, N)
    err = np.abs(rec - potencial(xf))
    mid = (xf >= 1.0) & (xf <= 3.5)
    # sobreimpulso: cuánto pasa la serie del valor real justo dentro del borde x=3.8 (V~0)
    cerca = xf > xa + L - 0.5
    sobre = abs(rec[cerca].min() - potencial(xf[cerca])[rec[cerca].argmin()])
    print(f"{N:>4} {err.max():>14.4f} {err[mid].max():>18.4f} {sobre / salto:>19.3f}")

err50 = np.abs(Vp(xf, 50) - potencial(xf))
plt.figure(figsize=(7, 4))
plt.semilogy(xf, err50)
plt.xlabel("x"); plt.ylabel("|Vp - V|"); plt.title("Error absoluto, N = 50")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig4_error_N50.pdf"); plt.savefig("fig4_error_N50.png", dpi=200)
print("\nFiguras guardadas.")