"""
05 - Transformada discreta de Fourier (DFT)
============================================
Definición (se implementa tal cual, con dos bucles):

        X_k = sum_{n=0}^{N-1} y_n exp(-2*pi*i*k*n/N)
        y_n = (1/N) sum_{k=0}^{N-1} X_k exp(+2*pi*i*k*n/N)

Eje de frecuencias:  f_k = k * fs / N   (fs = frecuencia de muestreo)
Para una señal real de amplitud A en f_k:  A = 2*|X_k|/N.

Ejemplo 1 (a mano): 4 muestras de una oscilación de 1 Hz durante 1 s.
Ejemplo 2: señal con dos oscilaciones (5 Hz y 12 Hz) muestreada a 64 Hz.

Ejecutar:  python 05_dft.py
"""
import numpy as np
import matplotlib.pyplot as plt


def dft(y):
    """DFT directa, O(N^2). Es la fórmula de la definición."""
    N = len(y)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += y[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def idft(X):
    """DFT inversa: signo + en el exponente y factor 1/N."""
    N = len(X)
    y = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            y[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return y / N


# =====================================================================
# Ejemplo 1: N = 4, cos(2*pi*t) muestreado cada 0.25 s
# =====================================================================
print("=== Ejemplo 1: N = 4 ===")
t1 = np.arange(4) * 0.25
y1 = np.cos(2 * np.pi * 1.0 * t1)            # (1, 0, -1, 0)
X1 = dft(y1)
print("y =", np.round(y1, 10))
print("X =", np.round(X1, 10), "  (esperado: 0, 2, 0, 2)")
print("amplitud 2|X1|/N =", 2 * abs(X1[1]) / 4, "  (esperado: 1)")
print("X3 = conj(X1)  (señal real):", np.isclose(X1[3], np.conj(X1[1])))
print("Parseval:  sum|y|^2 =", np.sum(abs(y1) ** 2),
      "  (1/N) sum|X|^2 =", np.sum(abs(X1) ** 2) / 4)
print("inversa:", np.round(idft(X1).real, 10), "\n")

# =====================================================================
# Ejemplo 2: dos oscilaciones, 5 Hz (amplitud 2) y 12 Hz (amplitud 0.5)
# =====================================================================
print("=== Ejemplo 2: dos frecuencias ===")
fs = 64.0                                    # frecuencia de muestreo (Hz)
N = 64                                       # muestras -> T = N/fs = 1 s
t = np.arange(N) / fs
y = 2.0 * np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 12 * t) + 1.0  # +1 = componente continua

X = dft(y)
freqs = np.arange(N) * fs / N                # f_k = k fs / N
amplitud = 2 * np.abs(X) / N                 # amplitud de cada senoidal
amplitud[0] = np.abs(X[0]) / N               # k = 0 es el valor medio (sin factor 2)

print("Picos del espectro (solo k <= N/2):")
for k in range(N // 2 + 1):
    if amplitud[k] > 1e-6:
        print(f"  k = {k:>2}  f = {freqs[k]:5.1f} Hz  amplitud = {amplitud[k]:.4f}")

print("¿coincide con np.fft.fft?  ", np.allclose(X, np.fft.fft(y)))
print("¿la inversa recupera y?    ", np.allclose(idft(X).real, y))

# ---------------- Gráficas ----------------
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(t, y, "o-")
ax[0].set_xlabel("t (s)")
ax[0].set_ylabel("y(t)")
ax[0].set_title("Señal muestreada")
ax[1].stem(freqs[: N // 2 + 1], amplitud[: N // 2 + 1])
ax[1].set_xlabel("f (Hz)")
ax[1].set_ylabel("amplitud")
ax[1].set_title("Espectro (DFT)")
plt.tight_layout()
plt.show()
