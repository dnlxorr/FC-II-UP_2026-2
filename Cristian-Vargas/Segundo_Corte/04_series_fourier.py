"""
04 - Series de Fourier
=======================
Ejemplo físico: tensión cuadrada de un generador de funciones, amplitud 1 V,
periodo 2*pi (frecuencia angular 1 rad/s):

        f(t) = -1  para -pi < t < 0        f(t) = +1  para 0 < t < pi

Serie:  f(t) = a0/2 + sum_n [ a_n cos(n t) + b_n sin(n t) ]
        a_n = (1/pi) * Integral f(t) cos(n t) dt
        b_n = (1/pi) * Integral f(t) sin(n t) dt        (en [-pi, pi])

Resultado analítico (f impar => a_n = 0):
        b_n = 4/(n*pi)  si n es impar,   0  si n es par.

Aquí calculamos los coeficientes NUMÉRICAMENTE (suma de rectángulos, que sobre
un periodo completo es muy precisa) y los comparamos con la fórmula.

Ejecutar:  python 04_series_fourier.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Malla de integración sobre un periodo ----------------
M = 20000                                   # número de puntos
dt = 2 * np.pi / M
# Puntos en el CENTRO de cada celda (regla del punto medio). Así ningún punto cae
# exactamente en el salto t = 0, donde f no está definida, y a_n sale 0 como debe.
t = -np.pi + (np.arange(M) + 0.5) * dt


def f(t):
    """Onda cuadrada: -1 en (-pi,0), +1 en (0,pi)."""
    return np.where(t < 0, -1.0, 1.0)


# ---------------- Coeficientes numéricos ----------------
def coef_a(n):
    return np.sum(f(t) * np.cos(n * t)) * dt / np.pi


def coef_b(n):
    return np.sum(f(t) * np.sin(n * t)) * dt / np.pi


print("Coeficientes de Fourier (numérico vs analítico)")
print(f"{'n':>2} {'a_n':>10} {'b_n num':>10} {'b_n teórico':>12}")
for n in range(1, 8):
    b_teo = 4 / (n * np.pi) if n % 2 == 1 else 0.0
    print(f"{n:>2} {coef_a(n):10.6f} {coef_b(n):10.6f} {b_teo:12.6f}")


# ---------------- Suma parcial S_N(t) ----------------
def suma_parcial(tt, N):
    """Serie truncada con armónicos impares hasta N (usa b_n = 4/(n pi))."""
    S = np.zeros_like(tt)
    for n in range(1, N + 1, 2):
        S += (4 / (n * np.pi)) * np.sin(n * tt)
    return S


# ---------------- Comprobación de Leibniz en t = pi/2 ----------------
print("\nS_N(pi/2) -> 1   (equivale a la serie de Leibniz para pi/4)")
for N in [1, 3, 5, 7, 9, 11, 101]:
    print(f"  N = {N:>3}: S_N(pi/2) = {suma_parcial(np.array([np.pi / 2]), N)[0]:.6f}")

# ---------------- Fenómeno de Gibbs ----------------
tt = np.linspace(1e-4, 1.0, 100001)
print("\nMáximo de S_N cerca del salto (Gibbs, límite ≈ 1.1790):")
for N in [11, 51, 201]:
    print(f"  N = {N:>3}: max S_N = {suma_parcial(tt, N).max():.5f}")

# ---------------- Gráfica ----------------
tp = np.linspace(-np.pi, np.pi, 2000)
plt.plot(tp, f(tp), "k", label="f(t)")
for N, color in [(1, "gray"), (3, "blue"), (9, "red")]:
    plt.plot(tp, suma_parcial(tp, N), color=color, label=f"N = {N}")
plt.xlabel("t")
plt.ylabel("S_N(t)")
plt.title("Sumas parciales de la serie de Fourier de la onda cuadrada")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
