"""
06 - Transformadas de funciones seno y coseno
==============================================
Parte A (continuas), con f(x) = exp(-a x) para x >= 0:
        F_c(k) = Integral_0^inf f(x) cos(kx) dx = a / (a^2 + k^2)
        F_s(k) = Integral_0^inf f(x) sin(kx) dx = k / (a^2 + k^2)
    Se verifican numéricamente con una suma de rectángulos.

Parte B (discretas, DCT y DST con media muestra de desfase):
        a_k = sum_n y_n cos(pi k (n+1/2)/N)      k = 0..N-1
        b_k = sum_n y_n sin(pi k (n+1/2)/N)      k = 1..N
    Se implementan a mano, se comprueba la inversa y se comparan con scipy.

Parte C (física): barra de longitud L con los extremos a temperatura 0
    (condición de Dirichlet => transformada SENO). Cada modo k decae como
        exp(-D (pi k / L)^2 t)
    así que evolucionamos la ecuación del calor en el espacio de modos.

Ejecutar:  python 06_seno_coseno.py
"""
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Parte A: transformadas continuas de exp(-a x)
# =====================================================================
print("=== Parte A: Fc y Fs de exp(-a x) ===")
a, k = 1.0, 2.0
x = np.linspace(0, 60, 600001)               # e^{-x} ya es ~0 en x = 60
dx = x[1] - x[0]
Fc_num = np.sum(np.exp(-a * x) * np.cos(k * x)) * dx
Fs_num = np.sum(np.exp(-a * x) * np.sin(k * x)) * dx
print(f"Fc: numérico = {Fc_num:.5f}   analítico a/(a²+k²) = {a / (a**2 + k**2):.5f}")
print(f"Fs: numérico = {Fs_num:.5f}   analítico k/(a²+k²) = {k / (a**2 + k**2):.5f}\n")


# =====================================================================
# Parte B: DCT y DST discretas
# =====================================================================
def dct(y):
    N = len(y)
    n = np.arange(N)
    return np.array([np.sum(y * np.cos(np.pi * kk * (n + 0.5) / N)) for kk in range(N)])


def idct(ak):
    N = len(ak)
    n = np.arange(N)
    y = np.zeros(N)
    for kk in range(1, N):
        y += 2 * ak[kk] * np.cos(np.pi * kk * (n + 0.5) / N)
    return (ak[0] + y) / N


def dst(y):
    N = len(y)
    n = np.arange(N)
    return np.array([np.sum(y * np.sin(np.pi * kk * (n + 0.5) / N)) for kk in range(1, N + 1)])


def idst(bk):
    N = len(bk)
    n = np.arange(N)
    y = np.zeros(N)
    for kk in range(1, N):
        y += 2 * bk[kk - 1] * np.sin(np.pi * kk * (n + 0.5) / N)
    return (y + (-1.0) ** n * bk[N - 1]) / N


print("=== Parte B: DCT y DST con N = 2, y = (3, 1) ===")
y = np.array([3.0, 1.0])
print("DCT a_k =", dct(y), "  (esperado: 4, sqrt(2) = 1.4142)")
print("DST b_k =", dst(y), "  (esperado: 2*sqrt(2) = 2.8284, 2)")
print("inversas:", idct(dct(y)), idst(dst(y)), "\n")

# Comparación con scipy (su definición lleva un factor 2)
try:
    from scipy.fft import dct as sp_dct, dst as sp_dst
    yy = np.random.default_rng(0).random(8)
    print("¿DCT propia = scipy/2?", np.allclose(dct(yy), sp_dct(yy, type=2) / 2))
    print("¿DST propia = scipy/2?", np.allclose(dst(yy), sp_dst(yy, type=2) / 2), "\n")
except ImportError:
    print("(scipy no está instalado: se omite la comparación)\n")

# =====================================================================
# Parte C: ecuación del calor en una barra con extremos a T = 0
# =====================================================================
print("=== Parte C: barra con extremos a 0 °C (transformada seno) ===")
L = 1.0                                      # longitud (m)
D = 1e-4                                     # difusividad térmica (m²/s)
N = 16                                       # puntos de la barra
x_n = (np.arange(N) + 0.5) * L / N           # puntos en el centro de cada celda
T0 = 100.0                                   # temperatura máxima inicial (°C)
T_ini = 4 * T0 * x_n * (L - x_n) / L**2      # perfil parabólico, 0 en los extremos

b = dst(T_ini)                               # 1) pasar al espacio de modos
coef = 2 * b / N                             # coeficientes de sin(pi k x / L)
coef[-1] = b[-1] / N                         # (el modo k = N lleva factor 1/N)
modos = np.arange(1, N + 1)


def temperatura(t):
    # 2) cada modo decae con su propia constante de tiempo
    decae = np.exp(-D * (np.pi * modos / L) ** 2 * t)
    return np.array([np.sum(coef * decae * np.sin(np.pi * modos * (n + 0.5) / N))
                     for n in range(N)])


t_final = 2000.0                             # s
T_final = temperatura(t_final)

# Solución exacta: x(L-x) = (8 L²/pi³) sum_{k impar} sin(k pi x/L)/k³
T_exacta = np.zeros(N)
for kk in range(1, 200, 2):
    T_exacta += (32 * T0 / (np.pi**3 * kk**3)) * np.exp(-D * (np.pi * kk / L) ** 2 * t_final) \
        * np.sin(kk * np.pi * x_n / L)

print(f"T en el centro: t=0 -> {T_ini.max():.2f} °C | t={t_final:.0f} s -> {T_final.max():.2f} °C")
print(f"máxima diferencia con la serie exacta = {np.max(np.abs(T_final - T_exacta)):.3f} °C")
print(f"¿t = 0 recupera el perfil inicial?  {np.allclose(temperatura(0.0), T_ini)}")

plt.plot(x_n, T_ini, "o-", label="t = 0")
plt.plot(x_n, T_final, "s-", label=f"t = {t_final:.0f} s (DST)")
plt.plot(x_n, T_exacta, "k--", label="serie exacta")
plt.xlabel("x (m)")
plt.ylabel("T (°C)")
plt.title("Difusión de calor con extremos a 0 °C")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
