r"""
=============================================================================
 TEMA: Metodo de relajacion de dos o mas variables
 CURSO: Fisica Computacional II
=============================================================================

EXTENSION A VARIAS VARIABLES Y SU JUSTIFICACION
-----------------------------------------------------------------
Un sistema de n ecuaciones no lineales que se escribe como

        x_1 = f_1(x_1,...,x_n)
        ...                         <=>      x = F(x) ,   x in R^n
        x_n = f_n(x_1,...,x_n)

se resuelve igual que en una variable: x^{(k+1)} = F(x^{(k)}). Para el
analisis del error se repite el argumento de Taylor, ahora en varias
variables. Con e^{(k)} = x^{(k)} - x*:

    e^{(k+1)} = F(x* + e^{(k)}) - F(x*) = J e^{(k)} + O(|e|^2) ,
    J_ij = d f_i / d x_j  evaluada en x*     (matriz jacobiana).

La derivada f'(x*) de una variable es reemplazada por la matriz J. Iterando,
e^{(k)} ~ J^k e^{(0)}. Si J es diagonalizable, J^k = V diag(mu_i^k) V^{-1},
de modo que e^{(k)} -> 0 para todo e^{(0)} si y solo si TODOS los
eigenvalores cumplen |mu_i| < 1, es decir

        radio espectral  rho(J) = max_i |mu_i| < 1 .

Y la velocidad asintotica es e^{(k)} ~ rho(J)^k (la direccion con el mayor
|mu| es la que tarda mas en extinguirse). Esta es la generalizacion exacta
de |f'(x*)| < 1 (y por eso aqui se usan los eigenvalores del script 7).

DOS VARIANTES (Jacobi y Gauss-Seidel)
-----------------------------------------------------------------
 * Jacobi (simultanea): todas las componentes nuevas se calculan con el vector
   VIEJO:  x_i^{new} = f_i(x^{old}).
 * Gauss-Seidel (secuencial): cada componente usa de inmediato las ya
   actualizadas:  x_i^{new} = f_i(x_1^{new},...,x_{i-1}^{new}, x_i^{old},...).
   Si J = L + U (L = parte estrictamente triangular inferior, U = resto), el
   error obedece e^{new} = L e^{new} + U e^{old}, es decir
        e^{new} = (I - L)^{-1} U e^{old} ,
   y la tasa es el radio espectral de (I-L)^{-1} U. Es la misma idea que
   Gauss-Seidel en sistemas lineales (y SOR, que le anade un factor omega).

EJEMPLO FISICO 1: campo medio con dos subredes (antiferromagneto)
-----------------------------------------------------------------
Dos subredes A y B de espines con acoplamiento ferromagnetico intrasubred
(a = z1 J1/kT > 0), antiferromagnetico intersubred (b = z2 J2/kT < 0) y un
campo externo reducido H = mu B/kT. Las magnetizaciones reducidas cumplen

    m_A = tanh( a m_A + b m_B + H )
    m_B = tanh( a m_B + b m_A + H )

Dos incognitas acopladas: es un punto fijo en R^2.
Verificacion independiente: con H = 0 el estado antiferromagnetico tiene
m_B = -m_A = -m, y las dos ecuaciones se reducen a UNA sola,
m = tanh((a - b) m), que es exactamente el problema del script 9 con
1/t = a - b. El resultado 2D debe reproducirlo.

EJEMPLO FISICO 2: cadena de N espines en campo medio (n variables)
-----------------------------------------------------------------
Cadena abierta de N espines con acoplamiento K = J/kT entre vecinos y un
campo local h_1 solo en el primer sitio (una "sonda" en el extremo):

        m_i = tanh( K (m_{i-1} + m_{i+1}) + h_i ) ,   m_0 = m_{N+1} = 0 .

El perfil m_i muestra como la perturbacion local penetra en la cadena: decae
exponencialmente con la distancia (longitud de correlacion).
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=120)


# =========================================================================
# 1) Relajacion multivariable (Jacobi y Gauss-Seidel)
# =========================================================================
def relajacion_nd(F, x0, metodo="jacobi", tol=1e-12, max_iter=20000):
    """
    F : funcion R^n -> R^n.
    metodo : "jacobi" (actualizacion simultanea) o "gs" (Gauss-Seidel).
    Devuelve la lista de iterados y el historial de ||x_{k+1} - x_k||_inf.
    Para "gs" se evalua F(x) completo pero solo se toma la componente i
    (es didactico; no es la implementacion mas barata).
    """
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    pasos = []
    for _ in range(max_iter):
        x_old = x.copy()
        if metodo == "jacobi":
            x = F(x_old)
        else:
            x = x_old.copy()
            for i in range(len(x)):
                x[i] = F(x)[i]            # usa las componentes ya actualizadas
        xs.append(x.copy())
        pasos.append(np.max(np.abs(x - x_old)))
        if pasos[-1] < tol:
            break
    return np.array(xs), np.array(pasos)


def radio_espectral(M):
    return np.max(np.abs(np.linalg.eigvals(M)))


def tasa_gauss_seidel(J):
    """Radio espectral de (I - L)^{-1} U con J = L + U."""
    L = np.tril(J, -1)
    U = J - L
    return radio_espectral(np.linalg.solve(np.eye(len(J)) - L, U))


# =========================================================================
# 2) Ejemplo 1: dos subredes
# =========================================================================
a, b = 0.3, -1.2          # acoplamientos reducidos (adimensionales)
H = 0.2


def F_subredes(x, H=H):
    mA, mB = x
    return np.array([np.tanh(a * mA + b * mB + H),
                     np.tanh(a * mB + b * mA + H)])


def jacobiana_subredes(x, H=H):
    mA, mB = x
    sA = 1 - np.tanh(a * mA + b * mB + H) ** 2      # sech^2 del argumento
    sB = 1 - np.tanh(a * mB + b * mA + H) ** 2
    return np.array([[a * sA, b * sA],
                     [b * sB, a * sB]])


x0 = np.array([0.9, -0.1])
xs_j, pasos_j = relajacion_nd(F_subredes, x0, "jacobi")
xs_g, pasos_g = relajacion_nd(F_subredes, x0, "gs")
x_star = xs_j[-1]

J_star = jacobiana_subredes(x_star)
rho_j = radio_espectral(J_star)
rho_g = tasa_gauss_seidel(J_star)

print("=" * 70)
print("EJEMPLO 1: dos subredes, a=%.2f, b=%.2f, H=%.2f" % (a, b, H))
print("=" * 70)
print(f"Punto fijo: m_A = {x_star[0]:.10f}, m_B = {x_star[1]:.10f}")
print(f"Residuo ||F(x*) - x*||_inf = {np.max(np.abs(F_subredes(x_star) - x_star)):.3e}")
print(f"Jacobiana en x*:\n{J_star}")
print(f"Eigenvalores de J: {np.linalg.eigvals(J_star)}")
print(f"rho(J)  [Jacobi]       = {rho_j:.5f}  (<1: converge)")
print(f"rho(GS) [Gauss-Seidel] = {rho_g:.5f}")
print(f"Iteraciones: Jacobi = {len(xs_j)-1},  Gauss-Seidel = {len(xs_g)-1}")

# tasa medida
err_j = np.max(np.abs(xs_j - x_star), axis=1)
err_g = np.max(np.abs(xs_g - x_star), axis=1)
tasa_med_j = np.mean(err_j[-12:-2][1:] / err_j[-12:-2][:-1])
tasa_med_g = np.mean(err_g[-8:-2][1:] / err_g[-8:-2][:-1])
print(f"Tasa medida e_(k+1)/e_k: Jacobi = {tasa_med_j:.5f} (teorica {rho_j:.5f}), "
      f"GS = {tasa_med_g:.5f} (teorica {rho_g:.5f})")

# --- verificacion con H = 0 contra el problema de una variable (script 9) ---
F0 = lambda x: F_subredes(x, H=0.0)
xs0, _ = relajacion_nd(F0, [0.9, -0.1], "jacobi")
m_2d = xs0[-1]
m1 = 1.0
for _ in range(100):                                # Newton para m = tanh((a-b) m)
    g = m1 - np.tanh((a - b) * m1)
    dg = 1 - (a - b) * (1 - np.tanh((a - b) * m1) ** 2)
    m1 -= g / dg
print(f"\nVerificacion con H=0:  2D -> (m_A, m_B) = ({m_2d[0]:.10f}, {m_2d[1]:.10f})")
print(f"                       1D -> m = tanh((a-b)m) = {m1:.10f}")
print(f"   |m_A - m| = {abs(m_2d[0]-m1):.2e}   |m_B + m| = {abs(m_2d[1]+m1):.2e}")

# =========================================================================
# 3) Ejemplo 2: cadena de N espines (N variables)
# =========================================================================
N = 30
K = 0.4
h = np.zeros(N)
h[0] = 1.0                               # campo solo en el primer sitio


def F_cadena(m):
    vec = np.zeros(N)
    vec[:-1] += m[1:]                    # vecino derecho
    vec[1:] += m[:-1]                    # vecino izquierdo
    return np.tanh(K * vec + h)


def jacobiana_cadena(m):
    s = 1 - F_cadena(m) ** 2
    A_ady = np.eye(N, k=1) + np.eye(N, k=-1)
    return (s[:, None]) * K * A_ady


m0 = np.zeros(N)
ms_j, p_j = relajacion_nd(F_cadena, m0, "jacobi", tol=1e-13)
ms_g, p_g = relajacion_nd(F_cadena, m0, "gs", tol=1e-13)
m_cad = ms_g[-1]
Jc = jacobiana_cadena(m_cad)
print("\n" + "=" * 70)
print(f"EJEMPLO 2: cadena de N={N} espines, K={K}")
print("=" * 70)
print(f"rho(J) Jacobi = {radio_espectral(Jc):.5f},  rho(GS) = {tasa_gauss_seidel(Jc):.5f}")
print(f"Iteraciones: Jacobi = {len(ms_j)-1},  Gauss-Seidel = {len(ms_g)-1}")
print(f"Residuo ||F(m) - m||_inf = {np.max(np.abs(F_cadena(m_cad) - m_cad)):.3e}")
print(f"Diferencia Jacobi vs GS en la solucion: {np.max(np.abs(ms_j[-1]-ms_g[-1])):.2e}")

# longitud de correlacion: pendiente de ln m_i para i grandes
idx = np.arange(1, 13)
pend = np.polyfit(idx, np.log(np.abs(m_cad[idx - 1])), 1)[0]
print(f"Decaimiento ~ exp({pend:.3f} i)  ->  longitud de correlacion xi = {-1/pend:.3f} sitios")

# =========================================================================
# 4) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) espacio de fases (mA, mB)
mm = np.linspace(-1, 1, 25)
MA, MB = np.meshgrid(mm, mm)
Fx = np.tanh(a * MA + b * MB + H) - MA
Fy = np.tanh(a * MB + b * MA + H) - MB
axes[0, 0].quiver(MA, MB, Fx, Fy, color="lightgray")
axes[0, 0].plot(xs_j[:, 0], xs_j[:, 1], "o-", color="#C1121F", ms=3, label="Jacobi")
axes[0, 0].plot(xs_g[:, 0], xs_g[:, 1], "s--", color="#2E86AB", ms=3, label="Gauss-Seidel")
axes[0, 0].plot(*x_star, "*", color="#F18F01", ms=16, label=r"$x^*$")
axes[0, 0].set_xlabel(r"$m_A$")
axes[0, 0].set_ylabel(r"$m_B$")
axes[0, 0].set_title("Trayectoria de la relajacion en el plano $(m_A,m_B)$")
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# (b) error vs iteracion
k1 = np.arange(len(err_j)); k2 = np.arange(len(err_g))
axes[0, 1].semilogy(k1, err_j + 1e-17, label="Jacobi")
axes[0, 1].semilogy(k2, err_g + 1e-17, label="Gauss-Seidel")
axes[0, 1].semilogy(k1, err_j[0] * rho_j ** k1, "k:", label=r"$\rho(J)^k$")
axes[0, 1].semilogy(k2, err_g[0] * rho_g ** k2, "k--", label=r"$\rho(GS)^k$")
axes[0, 1].set_ylim(1e-15, 2)
axes[0, 1].set_xlabel("Iteracion k")
axes[0, 1].set_ylabel(r"$\|x_k-x^*\|_\infty$")
axes[0, 1].set_title("Convergencia: tasa = radio espectral")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3, which="both")

# (c) perfil de la cadena
sitios = np.arange(1, N + 1)
axes[1, 0].semilogy(sitios, np.abs(m_cad), "o-", color="#2E86AB", label="solucion de la cadena")
axes[1, 0].semilogy(sitios[:14], np.abs(m_cad[0]) * np.exp(pend * (sitios[:14] - 1)),
                    "k--", label=rf"$e^{{-i/\xi}}$, $\xi={-1/pend:.2f}$")
axes[1, 0].set_xlabel("Sitio i")
axes[1, 0].set_ylabel(r"$|m_i|$")
axes[1, 0].set_title("Penetracion de la perturbacion local")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, which="both")

# (d) convergencia de la cadena
axes[1, 1].semilogy(p_j + 1e-17, label=f"Jacobi ({len(ms_j)-1} it.)")
axes[1, 1].semilogy(p_g + 1e-17, label=f"Gauss-Seidel ({len(ms_g)-1} it.)")
axes[1, 1].set_xlabel("Iteracion k")
axes[1, 1].set_ylabel(r"$\|x_{k+1}-x_k\|_\infty$")
axes[1, 1].set_title(f"Cadena de {N} espines: Jacobi vs Gauss-Seidel")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("10_relajacion_multivariable.png", dpi=150)
print("\nGrafica guardada en 10_relajacion_multivariable.png")