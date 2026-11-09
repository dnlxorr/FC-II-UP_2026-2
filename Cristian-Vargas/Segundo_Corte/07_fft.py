"""
07 - Transformada rápida de Fourier (FFT, Cooley-Tukey radix-2)
================================================================
Idea: separar los índices pares e impares,
        X_k       = E_k + W^k O_k
        X_{k+N/2} = E_k - W^k O_k          con W = exp(-2*pi*i/N)
donde E y O son DFT de longitud N/2 de las muestras pares e impares.
Se aplica recursivamente (N debe ser potencia de 2). Costo: O(N log2 N)
frente a O(N^2) de la DFT directa. El resultado es EXACTAMENTE el mismo.

Ejemplo físico: una nota de diapasón a 440 Hz con su segundo armónico (880 Hz,
amplitud menor) muestreada a fs = 8192 Hz con N = 1024 muestras.
Resolución en frecuencia: fs/N = 8 Hz (440 y 880 son múltiplos de 8).

Ejecutar:  python 07_fft.py
"""
import cmath
import time
import numpy as np
import matplotlib.pyplot as plt


def dft_lenta(y):
    """DFT directa O(N^2) en Python puro (para comparar tiempos)."""
    N = len(y)
    return [sum(y[n] * cmath.exp(-2j * cmath.pi * k * n / N) for n in range(N))
            for k in range(N)]


def fft(y):
    """FFT recursiva radix-2. len(y) debe ser potencia de 2."""
    N = len(y)
    if N == 1:                               # DFT de un solo punto: X_0 = y_0
        return [y[0]]
    E = fft(y[0::2])                         # DFT de las muestras pares
    O = fft(y[1::2])                         # DFT de las muestras impares
    X = [0] * N
    for k in range(N // 2):
        giro = cmath.exp(-2j * cmath.pi * k / N) * O[k]   # W^k * O_k
        X[k] = E[k] + giro                   # mariposa: dos salidas
        X[k + N // 2] = E[k] - giro          #           con un solo producto
    return X


# =====================================================================
# Paso 1: el ejemplo de 4 puntos de la guía
# =====================================================================
print("=== N = 4, y = (1, 0, -1, 0) ===")
print("pares   :", [1, -1], " impares:", [0, 0])
print("FFT     :", [complex(round(z.real, 10), round(z.imag, 10)) for z in fft([1, 0, -1, 0])])
print("esperado: 0, 2, 0, 2\n")

# =====================================================================
# Paso 2: comprobar que coincide con la DFT y con numpy
# =====================================================================
rng = np.random.default_rng(1)
y = rng.random(64)
print("=== Comprobación con N = 64 aleatorio ===")
print("¿FFT propia = DFT directa?   ", np.allclose(fft(list(y)), dft_lenta(list(y))))
print("¿FFT propia = np.fft.fft?    ", np.allclose(fft(list(y)), np.fft.fft(y)), "\n")

# =====================================================================
# Paso 3: tiempos (Python puro): DFT O(N^2) vs FFT O(N log N)
# =====================================================================
print("=== Tiempos ===")
print(f"{'N':>6} {'DFT (s)':>10} {'FFT (s)':>10} {'N^2':>10} {'N log2 N':>10} {'cociente t':>11}")
for m in range(6, 11):                       # N = 64 ... 1024
    N = 2**m
    senal = list(rng.random(N))
    t0 = time.perf_counter(); dft_lenta(senal); t_dft = time.perf_counter() - t0
    t0 = time.perf_counter(); fft(senal);       t_fft = time.perf_counter() - t0
    print(f"{N:>6} {t_dft:10.4f} {t_fft:10.4f} {N**2:>10} {N * m:>10} {t_dft / t_fft:11.1f}")

# =====================================================================
# Paso 4: ejemplo físico, diapasón a 440 Hz + armónico a 880 Hz
# =====================================================================
fs, N = 8192.0, 1024
t = np.arange(N) / fs
senal = 1.0 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)

X = np.array(fft(list(senal)))
f = np.arange(N) * fs / N
amp = 2 * np.abs(X) / N

print("\n=== Espectro del diapasón ===")
for k in range(N // 2):
    if amp[k] > 1e-3:
        print(f"  f = {f[k]:6.1f} Hz   amplitud = {amp[k]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(t[:200] * 1000, senal[:200])
ax[0].set_xlabel("t (ms)")
ax[0].set_ylabel("señal")
ax[0].set_title("Primeros 200 puntos")
ax[1].plot(f[: N // 2], amp[: N // 2])
ax[1].set_xlim(0, 1500)
ax[1].set_xlabel("f (Hz)")
ax[1].set_ylabel("amplitud")
ax[1].set_title("Espectro (FFT)")
plt.tight_layout()
plt.show()
