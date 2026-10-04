r"""
=============================================================================
 TEMA: Metodo de la secante
 CURSO: Fisica Computacional II
=============================================================================

MOTIVACION Y DEDUCCION
-----------------------------------------------------------------
Newton necesita f'(x_k). A veces f' no esta disponible (f es el resultado de
una simulacion, una integral numerica o una caja negra) o es cara de calcular.
La secante reemplaza la tangente por la RECTA SECANTE que pasa por los dos
ultimos puntos (x_{k-1}, f_{k-1}) y (x_k, f_k): es Newton con la derivada
aproximada por el cociente de diferencias

        f'(x_k) ~ ( f(x_k) - f(x_{k-1}) ) / ( x_k - x_{k-1} ) = f[x_{k-1}, x_k] .

Entonces

        x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / ( f(x_k) - f(x_{k-1}) ) .

Solo cuesta UNA evaluacion nueva de f por iteracion (la anterior se reutiliza).

ORDEN DE CONVERGENCIA (deduccion)
-----------------------------------------------------------------
Con diferencias divididas, f(x_k) = f[x_k, x*] e_k (porque f(x*) = 0). Asi

    e_{k+1} = e_k - f_k / f[x_{k-1},x_k]
            = e_k ( 1 - f[x_k,x*] / f[x_{k-1},x_k] )
            = e_k ( f[x_{k-1},x_k] - f[x_k,x*] ) / f[x_{k-1},x_k]
            = e_k e_{k-1} f[x_{k-1},x_k,x*] / f[x_{k-1},x_k]
            ~ ( f''(x*) / (2 f'(x*)) ) e_k e_{k-1} .

(se uso f[a,b] - f[b,c] = f[a,b,c] (a - c) y a - c = e_{k-1}). Suponiendo
e_{k+1} ~ A e_k^p, entonces e_{k-1} ~ (e_k/A)^{1/p} y la relacion anterior da
e_k^p ~ e_k^{1 + 1/p}, es decir

        p = 1 + 1/p   =>   p^2 - p - 1 = 0   =>   p = (1+sqrt 5)/2 ~ 1.618

(la razon aurea). Es superlineal pero menor que 2 de Newton. Sin embargo,
por costo por evaluacion de f, la secante es MEJOR: crecimiento por
evaluacion = 1.618 frente a sqrt(2) = 1.414 de Newton si cada derivada
cuesta una evaluacion extra (indice de eficiencia p^(1/costo)).

EJEMPLO FISICO: metodo de disparo para el oscilador armonico cuantico
-----------------------------------------------------------------
En unidades adimensionales (hbar = m = w = 1) la ecuacion de Schrodinger es

        -(1/2) psi'' + (1/2) x^2 psi = E psi   <=>   psi'' = (x^2 - 2E) psi .

Se exige psi -> 0 en x -> +-inf; numericamente, psi(-L) = 0 y psi(+L) = 0 con
L grande (justificacion: las soluciones no fisicas crecen como e^{x^2/2},
asi que el error de truncar en L = 6 es del orden de e^{-L^2} ~ 1e-16, muy
inferior al del integrador).

Metodo de disparo: dada una energia E de prueba, se integra desde x = -L con
psi(-L) = 0, psi'(-L) = 1 (la normalizacion no importa: la ecuacion es lineal)
hasta x = +L, y se define

        f(E) = psi_E(+L) .

Las energias permitidas son las RAICES de f(E). No existe una formula para
f'(E) (psi depende de E a traves de una integracion numerica), asi que
Newton no es practico: es el caso ideal para la secante.
Solucion exacta (para medir el error): E_n = n + 1/2.
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

np.set_printoptions(precision=10, suppress=True)
trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x / 1.x

L_BOX = 6.0
N_PASOS = 2400
H = 2 * L_BOX / N_PASOS
CONTADOR = {"evals": 0}


# -----------------------------------------------------------------------
# 1) Integrador RK4 de psi'' = (x^2 - 2E) psi
# -----------------------------------------------------------------------
def disparo(E, devolver_perfil=False):
    """Integra de -L a +L con RK4 y devuelve psi(+L) (o todo el perfil)."""
    CONTADOR["evals"] += 1
    x = -L_BOX
    psi, dpsi = 0.0, 1.0

    def deriv(x, psi, dpsi):
        return dpsi, (x * x - 2 * E) * psi

    xs = np.empty(N_PASOS + 1)
    ps = np.empty(N_PASOS + 1)
    xs[0], ps[0] = x, psi
    for i in range(N_PASOS):
        k1 = deriv(x, psi, dpsi)
        k2 = deriv(x + H / 2, psi + H / 2 * k1[0], dpsi + H / 2 * k1[1])
        k3 = deriv(x + H / 2, psi + H / 2 * k2[0], dpsi + H / 2 * k2[1])
        k4 = deriv(x + H, psi + H * k3[0], dpsi + H * k3[1])
        psi += H / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dpsi += H / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        x += H
        xs[i + 1], ps[i + 1] = x, psi
    return (xs, ps) if devolver_perfil else psi


# -----------------------------------------------------------------------
# 2) Metodo de la secante
# -----------------------------------------------------------------------
def secante(f, x0, x1, tol=1e-12, max_iter=50):
    """Devuelve la lista de iterados. Criterio: |x_{k+1}-x_k| < tol.
    Se protege contra f(x_k) = f(x_{k-1}) (denominador nulo)."""
    xs = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ZeroDivisionError("f(x_k) = f(x_{k-1}): la secante es horizontal")
        x2 = xs[-1] - f1 * (xs[-1] - xs[-2]) / (f1 - f0)
        xs.append(x2)
        if abs(x2 - xs[-2]) < tol:
            break
        f0, f1 = f1, f(x2)
    return np.array(xs)


# =========================================================================
# 3) Primeros cuatro niveles de energia
# =========================================================================
print(f"RK4 con h = {H:.4f}, L = {L_BOX}.  E exacta = n + 1/2\n")

# --- ADVERTENCIA: la secante NO esta atrapada en un intervalo (a diferencia
# de la biseccion). f(E) = psi(L) es extremadamente empinada (pasa de +1e13 a
# -1e13 entre E=0.3 y E=0.7), y con arranques lejanos la secante rebota:
CONTADOR["evals"] = 0
Es_lejos = secante(disparo, 0.3, 0.7, tol=1e-12)
print("Arranques LEJANOS [0.3, 0.7]: la secante necesita "
      f"{len(Es_lejos)-2} iteraciones y visita E_min = {Es_lejos.min():.3f} "
      "(fuera de [0.3, 0.7])")
print("En la practica se hace un barrido grueso de E para localizar el cambio de\n"
      "signo y se arranca la secante cerca de el:\n")
print(f"{'n':>2} {'arranques':>12} {'E calculada':>16} {'|E - E_exacta|':>16} "
      f"{'iter':>5} {'evals de f':>11}")

resultados = []
for n in range(4):
    CONTADOR["evals"] = 0
    Es = secante(disparo, n + 0.4, n + 0.6, tol=1e-12)
    resultados.append((n, Es, CONTADOR["evals"]))
    print(f"{n:2d} [{n+0.4:.1f},{n+0.6:.1f}] {Es[-1]:16.12f} "
          f"{abs(Es[-1] - (n + 0.5)):16.3e} {len(Es)-2:5d} {CONTADOR['evals']:11d}")

print("\nEl error residual (~1e-10) NO es de la secante (que converge hasta 1e-12):")
print("es el error del integrador RK4 (orden h^4) y del truncamiento en L. Se")
print("comprueba reduciendo h:")
for n_pasos in [600, 1200, 2400, 4800]:
    N_PASOS = n_pasos
    H = 2 * L_BOX / N_PASOS
    Es = secante(disparo, 0.4, 0.6, tol=1e-13)
    print(f"   N = {n_pasos:5d} pasos (h={H:.5f}): |E0 - 0.5| = {abs(Es[-1]-0.5):.3e}")
N_PASOS = 2400
H = 2 * L_BOX / N_PASOS

# =========================================================================
# 4) Costo: secante vs biseccion vs Newton con derivada numerica
# =========================================================================
CONTADOR["evals"] = 0
disparo(0.4); disparo(0.6)
CONTADOR["evals"] = 0
# biseccion en [0.3, 0.7] hasta 1e-10
a, b = 0.3, 0.7
fa = disparo(a)
while (b - a) > 1e-10:
    c = 0.5 * (a + b)
    fc = disparo(c)
    if fa * fc < 0:
        b = c
    else:
        a, fa = c, fc
evals_bis = CONTADOR["evals"]

CONTADOR["evals"] = 0
E = 0.4                                   # Newton con derivada por diferencia central
for _ in range(30):
    dE = 1e-5
    d = (disparo(E + dE) - disparo(E - dE)) / (2 * dE)       # 2 evaluaciones
    E_nuevo = E - disparo(E) / d                              # 1 evaluacion
    if abs(E_nuevo - E) < 1e-10:
        E = E_nuevo
        break
    E = E_nuevo
evals_newton = CONTADOR["evals"]

evals_sec = resultados[0][2]
print(f"\nEvaluaciones de f hasta ~1e-10 (estado base):")
print(f"   biseccion          : {evals_bis}")
print(f"   Newton (f' numerica): {evals_newton}  (3 evaluaciones por iteracion)")
print(f"   secante            : {evals_sec}")

# =========================================================================
# 5) Verificacion del orden de convergencia (p = 1.618) con 60 digitos
#    sobre el punto L1 del script 12 (f(u) con solucion exacta conocida)
# =========================================================================
mp.mp.dps = 300
G, M_T, M_L, R_TL, W = (mp.mpf("6.674e-11"), mp.mpf("5.974e24"),
                        mp.mpf("7.348e22"), mp.mpf("3.844e8"), mp.mpf("2.662e-6"))
al = G * M_T / (W ** 2 * R_TL ** 3)
be = G * M_L / (W ** 2 * R_TL ** 3)
g_mp = lambda u: al / u ** 2 - be / (1 - u) ** 2 - u
u_exacta = mp.findroot(g_mp, mp.mpf("0.85"))

xs_mp = [mp.mpf("0.80"), mp.mpf("0.90")]
for _ in range(11):
    f0, f1 = g_mp(xs_mp[-2]), g_mp(xs_mp[-1])
    xs_mp.append(xs_mp[-1] - f1 * (xs_mp[-1] - xs_mp[-2]) / (f1 - f0))
err_mp = [abs(x - u_exacta) for x in xs_mp]
p_est = [float(mp.log(err_mp[k + 1] / err_mp[k]) / mp.log(err_mp[k] / err_mp[k - 1]))
         for k in range(1, len(err_mp) - 1)]

print("\nOrden de convergencia estimado p_k = ln(e_{k+1}/e_k) / ln(e_k/e_{k-1})  (L1, 300 digitos):")
for k, pk in enumerate(p_est, start=1):
    print(f"   k={k}: p = {pk:.4f}")
print(f"   valor teorico (1+sqrt5)/2 = {(1+np.sqrt(5))/2:.4f}  (los primeros p_k aun no estan en el regimen asintotico)")

# =========================================================================
# 6) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# (a) f(E) con las raices
Eg = np.linspace(0.05, 4.4, 220)
fE = np.array([disparo(E) for E in Eg])
axes[0, 0].plot(Eg, np.sign(fE) * np.log10(1 + np.abs(fE)), color="#2E86AB")
axes[0, 0].axhline(0, color="k", lw=0.8)
for n in range(4):
    axes[0, 0].plot(n + 0.5, 0, "*", color="#C1121F", ms=14)
axes[0, 0].set_xlabel("E (unidades de $\\hbar\\omega$)")
axes[0, 0].set_ylabel(r"$\mathrm{sgn}(f)\,\log_{10}(1+|f|)$")
axes[0, 0].set_title(r"$f(E)=\psi_E(L)$: cada raiz es un nivel")
axes[0, 0].grid(alpha=0.3)

# (b) funciones de onda
from numpy.polynomial.hermite import hermval
xs_p, _ = disparo(0.5, True)
for n, Es, _ in resultados:
    xs_p, ps = disparo(Es[-1], True)
    # normalizar y comparar con la solucion exacta (Hermite)
    ps = ps / np.sqrt(trapz(ps ** 2, xs_p))
    coef = np.zeros(n + 1); coef[n] = 1
    exacta = hermval(xs_p, coef) * np.exp(-xs_p ** 2 / 2)
    exacta /= np.sqrt(trapz(exacta ** 2, xs_p))
    ps *= np.sign(ps @ exacta)
    axes[0, 1].plot(xs_p, ps + (n + 0.5), label=f"n={n}")
    axes[0, 1].plot(xs_p, exacta + (n + 0.5), "k--", lw=0.6)
    axes[0, 1].axhline(n + 0.5, color="gray", lw=0.4)
    print(f"max |psi_{n} disparo - psi_{n} exacta| = {np.max(np.abs(ps - exacta)):.2e}")
axes[0, 1].plot(xs_p, 0.5 * xs_p ** 2, "r:", lw=1, label=r"$V=x^2/2$")
axes[0, 1].set_xlim(-5, 5); axes[0, 1].set_ylim(0, 5)
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel(r"$E_n + \psi_n(x)$")
axes[0, 1].set_title("Funciones de onda (punteado: exacta)")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# (c) convergencia de la secante en cada nivel
for n, Es, _ in resultados:
    err = np.abs(Es - (n + 0.5))
    axes[0, 2].semilogy(np.arange(len(err)), err + 1e-17, "o-", ms=4, label=f"n={n}")
axes[0, 2].set_xlabel("Iteracion k")
axes[0, 2].set_ylabel(r"$|E_k-E_n|$")
axes[0, 2].set_title("Secante: error por nivel (piso: error del RK4)")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3, which="both")

# (d) orden de convergencia
axes[1, 0].plot(range(3, len(p_est) + 1), p_est[2:], "o-", color="#2E86AB", label=r"$p_k$ medido ($k\geq3$)")
axes[1, 0].axhline((1 + np.sqrt(5)) / 2, color="#C1121F", ls="--", label=r"$(1+\sqrt{5})/2$")
axes[1, 0].axhline(2, color="gray", ls=":", label="Newton (2)")
axes[1, 0].set_ylim(0.5, 2.4)
axes[1, 0].set_xlabel("Iteracion k"); axes[1, 0].set_ylabel("Orden estimado")
axes[1, 0].set_title("Orden de convergencia (L1, 300 digitos, arranques 0.80 y 0.90)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# (e) verificar e_{k+1} ~ C e_k e_{k-1}
e_f = np.array([float(x) for x in err_mp])
du = lambda u: -2 * al / u ** 3 - 2 * be / (1 - u) ** 3 - 1
d2u = lambda u: 6 * al / u ** 4 - 6 * be / (1 - u) ** 4
C = float(abs(d2u(u_exacta) / (2 * du(u_exacta))))
pred = C * e_f[1:-1] * e_f[:-2]
kk = np.arange(1, len(e_f) - 1)
axes[1, 1].semilogy(kk, e_f[2:], "o-", color="#2E86AB", label=r"$e_{k+1}$ medido")
axes[1, 1].semilogy(kk, pred, "x", color="#C1121F", ms=9, label=r"$C\,e_k e_{k-1}$ (teoria)")
axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("Error")
axes[1, 1].set_title(rf"Relacion $e_{{k+1}}\approx C e_k e_{{k-1}}$, $C={C:.3f}$")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

# (f) costo
etiquetas = ["biseccion", "Newton\n(f' numerica)", "secante"]
axes[1, 2].bar(etiquetas, [evals_bis, evals_newton, evals_sec],
               color=["#6A4C93", "#F18F01", "#2E86AB"])
for i, v in enumerate([evals_bis, evals_newton, evals_sec]):
    axes[1, 2].text(i, v + 0.5, str(v), ha="center")
axes[1, 2].set_ylabel("Evaluaciones de f (disparos RK4)")
axes[1, 2].set_title("Costo para el estado base (~1e-10)")
axes[1, 2].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("13_secante.png", dpi=150)
print("\nGrafica guardada en 13_secante.png")