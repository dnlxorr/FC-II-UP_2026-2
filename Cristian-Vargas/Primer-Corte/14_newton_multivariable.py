r"""
=============================================================================
 TEMA: Metodo de Newton de dos o mas variables
 CURSO: Fisica Computacional II
=============================================================================

DEDUCCION Y JUSTIFICACION
-----------------------------------------------------------------
Se busca x* en R^n con F(x*) = 0, F = (F_1,...,F_n). Igual que en una
variable, se linealiza F alrededor de la estimacion x_k (Taylor de primer
orden en varias variables):

        F(x) ~ F(x_k) + J(x_k) (x - x_k) ,      J_ij = dF_i / dx_j .

Imponiendo F(x_{k+1}) = 0 en la aproximacion lineal:

        J(x_k) Delta_k = - F(x_k) ,      x_{k+1} = x_k + Delta_k .

Observacion CLAVE (justificacion de la implementacion): NO se calcula la
inversa J^{-1} (aunque la formula x_{k+1} = x_k - J^{-1} F lo sugiera):
se resuelve el SISTEMA LINEAL J Delta = -F con eliminacion/LU con pivoteo
(scripts 1-4). Es mas barato y mas preciso (ver script 5). Cada iteracion de
Newton en n variables es entonces la solucion de un sistema lineal: por eso
este tema esta en "ecuaciones lineales simultaneas" y "no lineales".

CONVERGENCIA CUADRATICA en n variables. Con e_k = x_k - x*, Taylor de F en
x_k evaluado en x* da  0 = F(x_k) - J(x_k) e_k + O(|e_k|^2), y restando la
definicion del paso J(x_k) Delta_k = -F(x_k):

        e_{k+1} = e_k + Delta_k = J(x_k)^{-1} O(|e_k|^2) = O(|e_k|^2) .

Requiere J(x*) no singular. Lejos de x*, el metodo puede divergir, ciclar o
avanzar lentamente: se muestra abajo con un caso exponencial.

NEWTON AMORTIGUADO (globalizacion). El paso de Newton Delta es una direccion
de DESCENSO para phi(x) = (1/2)||F(x)||^2, porque
        d phi(x + lam Delta)/d lam |_{lam=0} = F^T J Delta = - ||F||^2 < 0 .
Si el paso completo (lam = 1) no reduce ||F||, se reduce lam a la mitad hasta
que si lo haga. Esto garantiza avance cuando el paso completo sobrepasa.

EJEMPLO FISICO: circuito con diodos (analisis nodal no lineal)
-----------------------------------------------------------------
Fuente V0 --R1-- nodo 1 --R2-- nodo 2, con un diodo de cada nodo a tierra.
Corriente de un diodo (ecuacion de Shockley):  I_D(V) = Is (exp(V/Vt) - 1).

Ley de corrientes de Kirchhoff en cada nodo (incognitas V1, V2):

   F_1 = (V0 - V1)/R1 - I_D(V1) - (V1 - V2)/R2 = 0
   F_2 = (V1 - V2)/R2 - I_D(V2)                 = 0

La exponencial hace el sistema NO LINEAL: no hay forma de resolverlo con los
metodos lineales, y ahi entra Newton. Jacobiana (derivadas analiticas):

   dF_1/dV1 = -1/R1 - Is/Vt exp(V1/Vt) - 1/R2 ,   dF_1/dV2 = 1/R2
   dF_2/dV1 =  1/R2 ,                              dF_2/dV2 = -1/R2 - Is/Vt exp(V2/Vt)

Se generaliza a una escalera de N nodos (N variables), donde J es tridiagonal.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

np.set_printoptions(precision=8, suppress=True, linewidth=120)

# -----------------------------------------------------------------------
# Parametros fisicos
# -----------------------------------------------------------------------
IS = 1e-12          # corriente de saturacion (A), diodo de silicio
VT = 0.02585        # voltaje termico kT/q a ~300 K (V)
V0 = 5.0            # V
R1 = 1000.0         # Ohm
R2 = 1000.0         # Ohm


def i_diodo(v):
    return IS * (np.exp(v / VT) - 1.0)


def F_circuito(v, V0=V0):
    V1, V2 = v
    return np.array([(V0 - V1) / R1 - i_diodo(V1) - (V1 - V2) / R2,
                     (V1 - V2) / R2 - i_diodo(V2)])


def J_circuito(v):
    V1, V2 = v
    return np.array([[-1 / R1 - IS / VT * np.exp(V1 / VT) - 1 / R2, 1 / R2],
                     [1 / R2, -1 / R2 - IS / VT * np.exp(V2 / VT)]])


# -----------------------------------------------------------------------
# Solver lineal propio: LU con pivoteo parcial (scripts 3 y 4)
# -----------------------------------------------------------------------
def resolver_lu(A, b):
    n = len(b)
    U = A.astype(float).copy()
    c = b.astype(float).copy()
    for k in range(n - 1):
        fm = k + np.argmax(np.abs(U[k:, k]))
        if fm != k:
            U[[k, fm]] = U[[fm, k]]
            c[[k, fm]] = c[[fm, k]]
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            U[i, k:] -= m * U[k, k:]
            c[i] -= m * c[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


# -----------------------------------------------------------------------
# Newton en n variables (con opcion de amortiguamiento)
# -----------------------------------------------------------------------
def newton_nd(F, J, x0, tol=1e-12, max_iter=300, amortiguado=False, verbose=False):
    """Devuelve (x, historial). historial = lista de (x_k, ||F||_2, ||paso||_inf, lambda)."""
    x = np.array(x0, dtype=float)
    hist = []
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(max_iter):
            Fx = F(x)
            delta = resolver_lu(J(x), -Fx)               # J delta = -F
            lam = 1.0
            if amortiguado:
                n0 = np.linalg.norm(Fx)
                while lam > 1e-12 and not (
                        np.linalg.norm(F(x + lam * delta)) < (1 - 1e-4 * lam) * n0):
                    lam /= 2
            hist.append((x.copy(), np.linalg.norm(Fx), np.max(np.abs(lam * delta)), lam))
            if verbose:
                print(f"  k={k:2d} x={x}  ||F||={hist[-1][1]:.3e}  |dx|={hist[-1][2]:.3e}  lam={lam:.3g}")
            x = x + lam * delta
            if not np.all(np.isfinite(x)):
                return x, hist
            if np.max(np.abs(lam * delta)) < tol:
                hist.append((x.copy(), np.linalg.norm(F(x)), 0.0, lam))
                return x, hist
    return x, hist


# =========================================================================
# 1) Resolucion con un buen arranque
# =========================================================================
print("Newton en 2 variables, arranque (0.6, 0.5) V:")
x_sol, hist = newton_nd(F_circuito, J_circuito, [0.6, 0.5], verbose=True)
V1, V2 = x_sol
I_fuente = (V0 - V1) / R1
I_D1, I_D2 = i_diodo(V1), i_diodo(V2)
I_R2 = (V1 - V2) / R2

print(f"\nV1 = {V1:.10f} V,  V2 = {V2:.10f} V   ({len(hist)-1} iteraciones)")
print(f"Corriente de la fuente = {I_fuente*1e3:.6f} mA")
print(f"I_D1 = {I_D1*1e3:.6f} mA,  I_R2 = I_D2 = {I_D2*1e3:.6f} mA")

# comprobaciones fisicas independientes
kcl1 = I_fuente - I_D1 - I_R2
kcl2 = I_R2 - I_D2
P_fuente = V0 * I_fuente
P_disip = I_fuente ** 2 * R1 + I_R2 ** 2 * R2 + V1 * I_D1 + V2 * I_D2
print(f"\nResiduos KCL: nodo 1 = {kcl1:.2e} A, nodo 2 = {kcl2:.2e} A")
print(f"Balance de potencia: P_fuente = {P_fuente*1e3:.8f} mW, "
      f"P_disipada = {P_disip*1e3:.8f} mW, diferencia = {abs(P_fuente-P_disip):.2e} W")

# =========================================================================
# 2) Convergencia cuadratica verificada con 60 digitos (mpmath)
# =========================================================================
mp.mp.dps = 60
Is_, Vt_, V0_, R1_, R2_ = mp.mpf("1e-12"), mp.mpf("0.02585"), mp.mpf(5), mp.mpf(1000), mp.mpf(1000)
Fm = lambda a, b: [(V0_ - a) / R1_ - Is_ * (mp.e ** (a / Vt_) - 1) - (a - b) / R2_,
                   (a - b) / R2_ - Is_ * (mp.e ** (b / Vt_) - 1)]
raiz_mp = mp.findroot(Fm, (mp.mpf("0.57"), mp.mpf("0.47")))
xk = mp.matrix([mp.mpf("0.6"), mp.mpf("0.5")])
errs = []
for _ in range(8):
    a, b = xk
    Jm = mp.matrix([[-1 / R1_ - Is_ / Vt_ * mp.e ** (a / Vt_) - 1 / R2_, 1 / R2_],
                    [1 / R2_, -1 / R2_ - Is_ / Vt_ * mp.e ** (b / Vt_)]])
    errs.append(mp.norm(xk - raiz_mp, mp.inf))
    xk = xk + mp.lu_solve(Jm, -mp.matrix(Fm(a, b)))
print(f"\n{'k':>2} {'||e_k||_inf':>14} {'||e_(k+1)|| / ||e_k||^2':>26}")
for k in range(len(errs) - 1):
    print(f"{k:2d} {mp.nstr(errs[k], 5):>14} {mp.nstr(errs[k+1]/errs[k]**2, 8):>26}")
print("La razon se estabiliza en una constante: convergencia cuadratica en R^2.")

# =========================================================================
# 3) Sensibilidad al arranque: Newton simple vs amortiguado
# =========================================================================
print("\n" + "=" * 70)
print("SENSIBILIDAD AL ARRANQUE")
print("=" * 70)
print(f"{'arranque':>14} | {'Newton':>10} | {'amortiguado':>12}")
arranques = [(0.6, 0.5), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
for s in arranques:
    _, h_s = newton_nd(F_circuito, J_circuito, s)
    _, h_a = newton_nd(F_circuito, J_circuito, s, amortiguado=True)
    print(f"{str(s):>14} | {len(h_s)-1:10d} | {len(h_a)-1:12d}")

# fase de "arrastre": desde un arranque alto, V baja ~Vt por iteracion
_, h_alto = newton_nd(F_circuito, J_circuito, (2.0, 2.0))
caidas = [h_alto[i][0][0] - h_alto[i + 1][0][0] for i in range(10)]
print(f"\nDesde (2,2): caida de V1 por iteracion (primeras 10) = {np.round(caidas, 5)}")
print(f"   Vt = {VT} V: Newton en una exponencial reduce V ~Vt por paso -> fase LINEAL")
print("   (la convergencia cuadratica solo aparece cerca de la raiz).")
print("   Explicacion: para f(V)=exp(V/Vt)-c, el paso de Newton es")
print("   V_{k+1} = V_k - Vt (1 - c exp(-V_k/Vt)) ~ V_k - Vt  cuando exp(V_k/Vt) >> c.")

# =========================================================================
# 4) Continuacion: caracteristica I-V del circuito (V0 variable)
# =========================================================================
V0_barrido = np.linspace(0.2, 10.0, 80)
sol_cont = []
x_prev = np.array([0.1, 0.05])
for v0 in V0_barrido:
    Fv = lambda v, v0=v0: F_circuito(v, V0=v0)
    x_prev, _ = newton_nd(Fv, J_circuito, x_prev)            # arranque = solucion anterior
    sol_cont.append(x_prev.copy())
sol_cont = np.array(sol_cont)
I_cont = (V0_barrido - sol_cont[:, 0]) / R1
print(f"\nContinuacion en V0 (80 puntos, arranque = solucion anterior): "
      f"I(V0=10 V) = {I_cont[-1]*1e3:.4f} mA")

# =========================================================================
# 5) Escalera de N nodos (N variables, Jacobiana tridiagonal)
# =========================================================================
N = 6
R_serie = 1000.0


def F_escalera(v, V0=V0):
    vv = np.concatenate(([V0], v, [v[-1]]))       # el ultimo nodo no tiene resistencia saliente
    Fv = np.zeros(N)
    for i in range(N):
        corriente_entra = (vv[i] - vv[i + 1]) / R_serie
        corriente_sale = (vv[i + 1] - vv[i + 2]) / R_serie if i < N - 1 else 0.0
        Fv[i] = corriente_entra - corriente_sale - i_diodo(vv[i + 1])
    return Fv


def J_escalera(v):
    Jm = np.zeros((N, N))
    for i in range(N):
        Jm[i, i] = -IS / VT * np.exp(v[i] / VT) - (1 / R_serie) - (1 / R_serie if i < N - 1 else 0.0)
        if i > 0:
            Jm[i, i - 1] = 1 / R_serie
        if i < N - 1:
            Jm[i, i + 1] = 1 / R_serie
    return Jm


x0_esc = np.full(N, 0.5)
x_esc, h_esc = newton_nd(F_escalera, J_escalera, x0_esc)
print("\n" + "=" * 70)
print(f"ESCALERA DE {N} NODOS")
print("=" * 70)
print(f"Voltajes de nodo: {x_esc}")
print(f"Iteraciones: {len(h_esc)-1},   ||F||_2 final = {np.linalg.norm(F_escalera(x_esc)):.2e} A")
print("Normas ||F|| por iteracion:", np.array([h[1] for h in h_esc]))
print(f"Jacobiana de la escalera (tridiagonal):\n{J_escalera(x_esc)}")

# =========================================================================
# 6) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# (a) trayectoria en el plano (V1, V2) sobre curvas de nivel de log10 ||F||
g1 = np.linspace(0.3, 0.9, 120)
G1, G2 = np.meshgrid(g1, g1)
NF = np.zeros_like(G1)
for i in range(G1.shape[0]):
    for j in range(G1.shape[1]):
        NF[i, j] = np.linalg.norm(F_circuito([G1[i, j], G2[i, j]]))
cs = axes[0, 0].contourf(G1, G2, np.log10(NF + 1e-30), 25, cmap="viridis")
plt.colorbar(cs, ax=axes[0, 0], label=r"$\log_{10}\|F\|$")
for inicio, color in [((0.8, 0.8), "#C1121F"), ((0.4, 0.8), "#F18F01"), ((0.35, 0.35), "white")]:
    _, hh = newton_nd(F_circuito, J_circuito, inicio, amortiguado=True)
    pts = np.array([h[0] for h in hh])
    axes[0, 0].plot(pts[:, 0], pts[:, 1], "o-", ms=3, color=color, lw=1)
axes[0, 0].plot(*x_sol, "*", color="red", ms=15)
axes[0, 0].set_xlabel(r"$V_1$ (V)"); axes[0, 0].set_ylabel(r"$V_2$ (V)")
axes[0, 0].set_title("Trayectorias de Newton hacia la solucion")

# (b) ||F|| vs iteracion
for s, amort, etiqueta, ls in [((0.6, 0.5), False, "(0.6,0.5) Newton", "o-"),
                               ((0.0, 0.0), False, "(0,0) Newton", "s-"),
                               ((0.0, 0.0), True, "(0,0) amortiguado", "^-")]:
    _, hh = newton_nd(F_circuito, J_circuito, s, amortiguado=amort)
    axes[0, 1].semilogy([h[1] + 1e-20 for h in hh], ls, ms=3, label=etiqueta)
axes[0, 1].set_xlim(0, 60)
axes[0, 1].set_xlabel("Iteracion k"); axes[0, 1].set_ylabel(r"$\|F(x_k)\|_2$  (A)")
axes[0, 1].set_title("Convergencia segun el arranque (eje cortado en k=60)")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3, which="both")

# (c) convergencia cuadratica (mp)
e_f = np.array([float(x) for x in errs])
axes[0, 2].semilogy(np.arange(len(e_f)), e_f, "o-", color="#2E86AB")
axes[0, 2].set_xlabel("Iteracion k"); axes[0, 2].set_ylabel(r"$\|x_k-x^*\|_\infty$")
axes[0, 2].set_title("Convergencia cuadratica (60 digitos)")
axes[0, 2].grid(alpha=0.3, which="both")

# (d) mapa de iteraciones segun el arranque
vs = np.linspace(0.05, 1.5, 30)
mapa = np.full((len(vs), len(vs)), np.nan)
for i, a in enumerate(vs):
    for j, b in enumerate(vs):
        _, hh = newton_nd(F_circuito, J_circuito, (a, b), max_iter=250)
        ok = len(hh) < 250 and np.all(np.isfinite(hh[-1][0])) and \
            np.linalg.norm(hh[-1][0] - x_sol) < 1e-6
        mapa[j, i] = len(hh) - 1 if ok else np.nan
im = axes[1, 0].imshow(mapa, origin="lower", extent=[vs[0], vs[-1], vs[0], vs[-1]],
                       cmap="magma_r", aspect="auto")
plt.colorbar(im, ax=axes[1, 0], label="Iteraciones")
axes[1, 0].set_xlabel(r"Arranque $V_1^{(0)}$"); axes[1, 0].set_ylabel(r"Arranque $V_2^{(0)}$")
axes[1, 0].set_title("Iteraciones segun el arranque (Newton simple)")

# (e) caracteristica I-V por continuacion
axes[1, 1].plot(V0_barrido, sol_cont[:, 0], label=r"$V_1$")
axes[1, 1].plot(V0_barrido, sol_cont[:, 1], label=r"$V_2$")
axes[1, 1].set_xlabel(r"$V_0$ (V)"); axes[1, 1].set_ylabel("Voltaje de nodo (V)")
ax2 = axes[1, 1].twinx()
ax2.plot(V0_barrido, I_cont * 1e3, "k--", label="I fuente")
ax2.set_ylabel("Corriente de la fuente (mA)")
axes[1, 1].set_title("Barrido de $V_0$ por continuacion")
axes[1, 1].legend(loc="upper left", fontsize=8)
axes[1, 1].grid(alpha=0.3)

# (f) escalera
axes[1, 2].plot(np.arange(1, N + 1), x_esc, "o-", color="#2E86AB")
axes[1, 2].set_xlabel("Nodo i"); axes[1, 2].set_ylabel("Voltaje (V)")
axes[1, 2].set_title(f"Escalera de {N} diodos: perfil de voltaje")
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("14_newton_multivariable.png", dpi=150)
print("\nGrafica guardada en 14_newton_multivariable.png")