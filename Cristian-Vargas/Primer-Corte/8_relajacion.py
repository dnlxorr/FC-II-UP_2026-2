r"""
=============================================================================
 TEMA: Metodo de relajacion (iteracion de punto fijo) en una variable
 CURSO: Fisica Computacional II
=============================================================================

IDEA DEL METODO Y SU JUSTIFICACION
-----------------------------------------------------------------
Se quiere resolver una ecuacion no lineal. Si se puede reescribir en la forma

        x = f(x) ,

una solucion x* es un "punto fijo" de f: f(x*) = x*. El metodo de relajacion
propone partir de una estimacion x_0 y repetir

        x_{k+1} = f(x_k) .

Por que tiene sentido: si la sucesion converge a un limite x*, y f es
continua, entonces, pasando al limite en x_{k+1} = f(x_k),

        x* = lim x_{k+1} = lim f(x_k) = f(lim x_k) = f(x*) ,

es decir, el limite ES una solucion. La pregunta de fondo (cuando converge y
a que velocidad) se estudia en el script 9; aqui el resultado clave es:

        e_{k+1} = x_{k+1} - x* = f(x_k) - f(x*) ~ f'(x*) e_k ,

(Taylor de primer orden alrededor de x*), de modo que el error se multiplica
por f'(x*) en cada paso: hay convergencia si |f'(x*)| < 1.

EJEMPLO FISICO: ley de desplazamiento de Wien
-----------------------------------------------------------------
La radiancia espectral de un cuerpo negro por unidad de longitud de onda es

    I(lam) = 2 pi h c^2 / lam^5  *  1 / ( exp(h c /(lam k T)) - 1 )

Su maximo se halla con dI/dlam = 0. Con el cambio x = h c /(lam k T) la
condicion de maximo se reduce (tras derivar) a

        5 e^{-x} + x - 5 = 0      <=>      x = 5 - 5 e^{-x}  =  f(x) .

Esta ultima forma es una ecuacion de punto fijo lista para relajacion. Su
solucion x* ~ 4.9651 da la ley de Wien:

        lam_max T = b ,     b = h c / (k x*) .

Con b se obtiene la temperatura de la superficie del Sol a partir de la
longitud de onda del maximo de su espectro (lam_max ~ 502 nm).

POR QUE ESTA FORMA f(x) = 5 - 5 e^{-x} CONVERGE (y otras no)
-----------------------------------------------------------------
f'(x) = 5 e^{-x}; en x* ~ 4.965, f'(x*) = 5 e^{-4.965} ~ 0.035 << 1, asi que
cada iteracion reduce el error ~28 veces. Ojo: la ecuacion tambien se puede
despejar como x = -ln(1 - x/5), cuya derivada en x* es 1/(5-x*) ~ 28 > 1:
el punto fijo x* es REPULSOR (cualquier error se amplifica ~28 veces por
paso), asi que la iteracion se ALEJA de x*. Como la ecuacion original
5 e^{-x} + x - 5 = 0 tiene tambien la raiz trivial x = 0 (con f'(0) = 1/5 < 1,
atractora), la iteracion termina cayendo ahi. La forma de despejar importa,
y se compara abajo.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constantes CODATA (SI)
H_PLANCK = 6.62607015e-34       # J s
C_LUZ = 2.99792458e8            # m / s
K_B = 1.380649e-23              # J / K


def relajacion(f, x0, tol=1e-12, max_iter=200, verbose=True):
    """Itera x_{k+1} = f(x_k). Devuelve la lista de iterados.
    Criterio de parada: |x_{k+1} - x_k| < tol (ver script 9 sobre su
    relacion con el error real)."""
    xs = [x0]
    for k in range(max_iter):
        x_nuevo = f(xs[-1])
        xs.append(x_nuevo)
        if verbose:
            print(f"  k={k+1:3d}  x = {x_nuevo:.14f}   |dx| = {abs(x_nuevo - xs[-2]):.3e}")
        if not np.isfinite(x_nuevo):
            break
        if abs(x_nuevo - xs[-2]) < tol:
            break
    return np.array(xs)


# -----------------------------------------------------------------------
# 1) Resolucion con la forma convergente
# -----------------------------------------------------------------------
f_buena = lambda x: 5.0 - 5.0 * np.exp(-x)

print("Relajacion x = 5 - 5 exp(-x), x0 = 1:")
xs = relajacion(f_buena, 1.0)
x_star = xs[-1]

# Raiz de referencia con alta precision (Newton, ver script 12) para medir
# el error REAL, no solo el criterio de parada
g = lambda x: 5 * np.exp(-x) + x - 5
dg = lambda x: 1 - 5 * np.exp(-x)
x_ref = 5.0
for _ in range(50):
    x_ref -= g(x_ref) / dg(x_ref)

print(f"\nx* (relajacion) = {x_star:.14f}")
print(f"x* (referencia) = {x_ref:.14f}")
print(f"|error| = {abs(x_star - x_ref):.3e}   en {len(xs)-1} iteraciones")

# -----------------------------------------------------------------------
# 2) Consecuencia fisica: constante de Wien y temperatura del Sol
# -----------------------------------------------------------------------
b_wien = H_PLANCK * C_LUZ / (K_B * x_star)
b_ref = 2.897771955e-3            # valor CODATA (m K)
lam_sol = 502e-9                  # m (maximo del espectro solar)
T_sol = b_wien / lam_sol

print(f"\nConstante de Wien b = {b_wien:.9e} m K   (CODATA: {b_ref:.9e})")
print(f"Error relativo en b = {abs(b_wien - b_ref)/b_ref:.2e}")
print(f"Temperatura superficial del Sol: T = {T_sol:.1f} K")

# -----------------------------------------------------------------------
# 3) Comparacion: dos formas de despejar la misma ecuacion
# -----------------------------------------------------------------------
f_mala = lambda x: -np.log(1 - x / 5.0) if x < 5 else np.nan
print("\nOtra forma de despejar: x = -ln(1 - x/5), x0 = 4.0 (se aleja de x* y cae en x=0):")
xs_mala = relajacion(f_mala, 4.0, max_iter=12)
print(f"|f'(x*)| forma buena = {5*np.exp(-x_ref):.4f}   forma mala = {1/(5-x_ref):.4f}")

# =========================================================================
# 4) Graficas
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) diagrama de telarana (cobweb)
xx = np.linspace(0, 6, 400)
axes[0].plot(xx, f_buena(xx), color="#2E86AB", label=r"$f(x)=5-5e^{-x}$")
axes[0].plot(xx, xx, "k--", label=r"$y=x$")
cx, cy = [xs[0]], [0.0]
for i in range(min(len(xs) - 1, 6)):
    cx += [xs[i], xs[i + 1]]
    cy += [xs[i + 1], xs[i + 1]]
axes[0].plot(cx, cy, color="#C1121F", lw=1.2, label="iteracion")
axes[0].plot(x_ref, x_ref, "o", color="#F18F01", label=r"$x^*$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_title("Diagrama de telarana")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# (b) error real vs iteracion
err = np.abs(xs - x_ref)
axes[1].semilogy(range(len(err)), err + 1e-18, "o-", color="#2E86AB",
                 label="error real $|x_k-x^*|$")
k_ = np.arange(len(err))
axes[1].semilogy(k_, err[0] * (5 * np.exp(-x_ref)) ** k_, "k:",
                 label=r"$|e_0|\,|f'(x^*)|^k$")
axes[1].set_ylim(1e-17, 10)
axes[1].set_xlabel("Iteracion k")
axes[1].set_ylabel("Error")
axes[1].set_title("Convergencia lineal de la relajacion")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")

# (c) buena vs mala
err_mala = np.abs(xs_mala - x_ref)
axes[2].semilogy(range(len(err)), err + 1e-18, "o-", color="#2E86AB",
                 label=r"$x=5-5e^{-x}$  ($|f'|\approx0.035$)")
axes[2].semilogy(range(len(err_mala)), err_mala, "s-", color="#C1121F",
                 label=r"$x=-\ln(1-x/5)$  ($|f'|\approx28$, se aleja de $x^*$)")
axes[2].set_ylim(1e-17, 1e2)
axes[2].set_xlabel("Iteracion k")
axes[2].set_ylabel("Error")
axes[2].set_title("La forma de despejar decide la convergencia")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("8_relajacion.png", dpi=150)
print("\nGrafica guardada en 8_relajacion.png")