r"""
=============================================================================
 TEMA: Convergencia del metodo de relajacion
 CURSO: Fisica Computacional II
=============================================================================

RESULTADO TEORICO QUE SE VERIFICA
-----------------------------------------------------------------
Sea x* un punto fijo (x* = f(x*)) y e_k = x_k - x* el error. Por Taylor de
primer orden de f alrededor de x*:

    x_{k+1} = f(x_k) = f(x*) + f'(x*) (x_k - x*) + O(e_k^2)
  =>  e_{k+1} = f'(x*) e_k + O(e_k^2) .

Consecuencias:
  (1) Convergencia si |f'(x*)| < 1 (el error se contrae); divergencia si > 1.
  (2) Es convergencia LINEAL: e_k ~ f'(x*)^k e_0. El numero de iteraciones
      para bajar el error de e_0 a tol es  N = ln(tol/e_0) / ln|f'(x*)| .
  (3) Si f' < 0 el error cambia de signo en cada paso (convergencia
      oscilatoria); si f' > 0 es monotona.
  (4) Estimacion del error SIN conocer x*: como x_{k+1} - x_k = e_{k+1} - e_k
      = (f' - 1) e_k, entonces

            e_k ~ (x_{k+1} - x_k) / (f' - 1) .

      Si f' esta cerca de 1, el error real es MUCHO mayor que el paso
      |x_{k+1} - x_k|: parar cuando el paso es pequeno puede engañar.
  (5) Sobre-relajacion: se itera  x_{k+1} = (1+w) f(x_k) - w x_k . Es la
      misma ecuacion (el punto fijo no cambia: si x = f(x) entonces
      (1+w) f(x) - w x = x), pero con derivada (1+w) f' - w, que se anula
      con  w_opt = f' / (1 - f')  : convergencia mucho mas rapida.

EJEMPLO FISICO: campo medio de Weiss (ferromagnetismo)
-----------------------------------------------------------------
En la teoria de campo medio, la magnetizacion reducida m de un ferromagneto
a temperatura T satisface la ecuacion de autoconsistencia

        m = tanh( m Tc / T ) = tanh( m / t ) ,      t = T / Tc ,

(Tc es la temperatura de Curie). Es un problema natural de punto fijo:
f(m) = tanh(m/t). Para t < 1 existe una solucion no nula m*(t) (fase
ferromagnetica); para t >= 1 solo m = 0.

    f'(m*) = (1 - m*^2) / t .

Cuando t -> 1^- se tiene m* -> 0 y f'(m*) -> 1: la convergencia se vuelve
cada vez mas lenta. Es la version numerica del fenomeno fisico de
"ralentizacion critica" (critical slowing down): cerca de la transicion de
fase el sistema tarda mucho en relajarse al equilibrio, y el algoritmo
tambien.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)


def relajacion(f, x0, tol=1e-10, max_iter=200000):
    xs = [x0]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            xs.append(f(xs[-1]))
            if not np.isfinite(xs[-1]) or abs(xs[-1]) > 1e8:
                break                      # divergencia: se aborta
            if abs(xs[-1] - xs[-2]) < tol:
                break
    return np.array(xs)


def m_estrella(t):
    """Raiz no nula de m = tanh(m/t) por Newton (referencia de alta
    precision; Newton se explica en el script 12)."""
    m = 1.0
    for _ in range(100):
        g = m - np.tanh(m / t)
        dg = 1 - (1 - np.tanh(m / t) ** 2) / t
        m -= g / dg
    return m


# =========================================================================
# 1) Tasa de convergencia medida vs teorica, en funcion de t
# =========================================================================
temps = np.array([0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
fprime_teo, razon_med, n_iter_med, n_iter_teo, m_vals = [], [], [], [], []
tol = 1e-10

print(f"{'t=T/Tc':>7} | {'m*':>9} | {'f(m*)':>8} | {'razon medida':>12} | "
      f"{'iter medidas':>12} | {'iter teoricas':>13}")
for t in temps:
    ms = m_estrella(t)
    fp = (1 - ms ** 2) / t
    f = lambda m, t=t: np.tanh(m / t)
    xs = relajacion(f, 1.0, tol=tol)
    err = np.abs(xs - ms)
    # razon e_{k+1}/e_k promediada en los ultimos pasos (regimen asintotico)
    r = np.mean(err[-6:-1][1:] / err[-6:-1][:-1])
    n_teo = np.log(tol / ((1 - fp) * abs(1.0 - ms))) / np.log(fp)
    fprime_teo.append(fp); razon_med.append(r)
    n_iter_med.append(len(xs) - 1); n_iter_teo.append(n_teo); m_vals.append(ms)
    print(f"{t:7.2f} | {ms:9.6f} | {fp:8.5f} | {r:12.5f} | "
          f"{len(xs)-1:12d} | {n_teo:13.1f}")

fprime_teo, razon_med = np.array(fprime_teo), np.array(razon_med)
print("\nLa razon medida e_{k+1}/e_k coincide con f'(m*): confirma e_{k+1} ~ f'(m*) e_k.")
print("Las iteraciones teoricas (asintoticas, con f' evaluada en m*) son una cota")
print("algo pesimista: lejos de m*, |f'(m)| es menor y las primeras iteraciones")
print("contraen mas rapido de lo que predice f'(m*). Al parar con paso < tol, el")
print("error real equivalente es tol/(1-f'), que es lo que usa la formula.")

# =========================================================================
# 2) El criterio de parada |x_{k+1}-x_k| < tol puede engañar
# =========================================================================
t_caso = 0.98
ms = m_estrella(t_caso)
fp = (1 - ms ** 2) / t_caso
xs = relajacion(lambda m: np.tanh(m / t_caso), 1.0, tol=1e-6)
err_real = np.abs(xs - ms)
paso = np.abs(np.diff(xs))
err_est = paso / abs(fp - 1)             # estimador (4)

print(f"\nt = {t_caso}: f'(m*) = {fp:.5f}")
print(f"Al parar con |dx| < 1e-6 (tras {len(xs)-1} iteraciones):")
print(f"   paso |x_(k+1)-x_k| = {paso[-1]:.3e}")
print(f"   error REAL         = {err_real[-1]:.3e}")
print(f"   error estimado     = {err_est[-1]:.3e}  (formula (4))")
print(f"   El error real es {err_real[-1]/paso[-1]:.0f} veces el paso: "
      f"1/(1-f') = {1/(1-fp):.0f}")

# =========================================================================
# 3) Sobre-relajacion
# =========================================================================
t_sr = 0.95
ms_sr = m_estrella(t_sr)
fp_sr = (1 - ms_sr ** 2) / t_sr
w_opt = fp_sr / (1 - fp_sr)
f_base = lambda m: np.tanh(m / t_sr)
g_w = lambda w: (lambda m: (1 + w) * f_base(m) - w * m)

print(f"\nSobre-relajacion a t = {t_sr}: f'(m*) = {fp_sr:.5f}, w_opt = f'/(1-f') = {w_opt:.3f}")

# w_opt anula la derivada de g SOLO en m*: es una propiedad LOCAL. Desde un
# arranque lejano, |g'(m)| puede superar 1 y la iteracion diverge:
xs_lejos = relajacion(g_w(w_opt), 1.0, tol=1e-12, max_iter=500)
dg_lejos = (1 + w_opt) * (1 - np.tanh(1.0 / t_sr) ** 2) / t_sr - w_opt
print(f"   Desde m0 = 1.00 con w_opt: g'(m0) = {dg_lejos:.2f} (|g'|>1) -> "
      f"{'DIVERGE' if not np.isfinite(xs_lejos[-1]) or abs(xs_lejos[-1])>1e3 else 'converge'}")

# Desde un arranque cercano (m0 = 0.45, a ~19% de m*) si funciona:
m0 = 0.45
xs_normal = relajacion(f_base, m0, tol=1e-12)
xs_w_mitad = relajacion(g_w(w_opt / 2), m0, tol=1e-12)
xs_w = relajacion(g_w(w_opt), m0, tol=1e-12)
print(f"   Desde m0 = {m0}:")
print(f"      sin sobre-relajar : {len(xs_normal)-1:4d} iteraciones")
print(f"      w = w_opt/2       : {len(xs_w_mitad)-1:4d} iteraciones")
print(f"      w = w_opt         : {len(xs_w)-1:4d} iteraciones")
print(f"      solucion: m* = {xs_w[-1]:.12f}  (referencia {ms_sr:.12f})")

# =========================================================================
# 4) Convergencia oscilatoria (f' < 0): x = cos(x), solo como verificacion
# =========================================================================
xs_cos = relajacion(np.cos, 1.0, tol=1e-12)
x_cos = xs_cos[-1]
print(f"\nCaso f'<0 (verificacion): x = cos(x) -> x* = {x_cos:.12f}, "
      f"f'(x*) = {-np.sin(x_cos):.4f} (error alterna de signo)")

# =========================================================================
# 5) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

tt = np.linspace(0.3, 0.995, 200)
mm = np.array([m_estrella(t) for t in tt])
axes[0, 0].plot(tt, mm, color="#2E86AB")
axes[0, 0].set_xlabel(r"$t=T/T_c$")
axes[0, 0].set_ylabel(r"$m^*$")
axes[0, 0].set_title("Magnetizacion espontanea (campo medio)")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(temps, fprime_teo, "o-", color="#2E86AB", label=r"$f'(m^*)$ teorica")
axes[0, 1].plot(temps, razon_med, "x", color="#C1121F", ms=9, label=r"razon medida $e_{k+1}/e_k$")
axes[0, 1].axhline(1, color="k", ls=":")
axes[0, 1].set_xlabel(r"$t=T/T_c$")
axes[0, 1].set_ylabel("Factor de contraccion por iteracion")
axes[0, 1].set_title(r"Tasa de convergencia $\to 1$ en $T_c$")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

axes[0, 2].semilogy(temps, n_iter_med, "o-", color="#2E86AB", label="medidas")
axes[0, 2].semilogy(temps, n_iter_teo, "k--", label=r"$\ln(tol/e_0)/\ln f'$")
axes[0, 2].set_xlabel(r"$t=T/T_c$")
axes[0, 2].set_ylabel("Iteraciones hasta tol = 1e-10")
axes[0, 2].set_title("Ralentizacion critica")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3, which="both")

axes[1, 0].semilogy(paso, label=r"paso $|x_{k+1}-x_k|$")
axes[1, 0].semilogy(err_real[:-1], label="error real")
axes[1, 0].semilogy(err_est, "k:", label=r"estimador $\Delta x/|f'-1|$")
axes[1, 0].set_xlabel("Iteracion k")
axes[1, 0].set_title(f"Criterio de parada engañoso (t={t_caso})")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, which="both")

axes[1, 1].semilogy(np.abs(xs_normal - ms_sr) + 1e-17, label="relajacion simple")
axes[1, 1].semilogy(np.abs(xs_w_mitad - ms_sr) + 1e-17, label=r"$\omega=\omega_{opt}/2$")
axes[1, 1].semilogy(np.abs(xs_w - ms_sr) + 1e-17, label=r"$\omega=\omega_{opt}$")
axes[1, 1].set_xlim(0, 250)
axes[1, 1].set_xlabel("Iteracion k")
axes[1, 1].set_ylabel("Error")
axes[1, 1].set_title(f"Sobre-relajacion (t={t_sr}, "
                     + r"$\omega_{opt}$" + f"={w_opt:.1f}, " + r"$m_0$=0.45)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

xx = np.linspace(0, 1.2, 300)
axes[1, 2].plot(xx, np.cos(xx), color="#2E86AB", label=r"$\cos x$")
axes[1, 2].plot(xx, xx, "k--")
cx, cy = [xs_cos[0]], [0.0]
for i in range(8):
    cx += [xs_cos[i], xs_cos[i + 1]]
    cy += [xs_cos[i + 1], xs_cos[i + 1]]
axes[1, 2].plot(cx, cy, color="#C1121F", lw=1)
axes[1, 2].set_title(r"$f'<0$: convergencia oscilatoria")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("9_convergencia_relajacion.png", dpi=150)
print("\nGrafica guardada en 9_convergencia_relajacion.png")