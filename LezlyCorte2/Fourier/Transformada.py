"""
Transformada discreta de Fourier (DFT) aplicada a la señal de un oscilador
amortiguado con dos modos (dos frecuencias). Se implementa la DFT directa
O(N^2) y se compara con la FFT de NumPy.
    X_k = sum_{n=0}^{N-1} x_n exp(-2 pi i k n / N)
"""
import time
import numpy as np
import matplotlib.pyplot as plt

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)   # matriz de Fourier
    return W @ x

def dft_bucles(x):
    """Versión con bucles explícitos (más didáctica, más lenta)."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

if __name__ == "__main__":
    # Señal: dos modos de vibración amortiguados + ruido
    fs = 200.0                      # frecuencia de muestreo (Hz)
    T = 5.0                         # duración (s)
    N = int(fs * T)
    t = np.arange(N) / fs
    f1, f2 = 12.0, 30.0             # frecuencias reales (Hz)
    rng = np.random.default_rng(1)
    x = (1.0 * np.exp(-0.4 * t) * np.sin(2*np.pi*f1*t)
         + 0.6 * np.exp(-0.8 * t) * np.sin(2*np.pi*f2*t)
         + rng.normal(0, 0.05, N))

    t0 = time.perf_counter(); X = dft(x); t_dft = time.perf_counter() - t0
    t0 = time.perf_counter(); Xf = np.fft.fft(x); t_fft = time.perf_counter() - t0

    print(f"Máx. diferencia DFT vs FFT: {np.max(np.abs(X - Xf)):.2e}")
    print(f"Tiempo DFT: {t_dft*1e3:.2f} ms | FFT: {t_fft*1e3:.3f} ms")

    freqs = np.fft.fftfreq(N, d=1/fs)
    mitad = N // 2
    amp = 2 * np.abs(X[:mitad]) / N
    top = np.argsort(amp)[-2:]
    print("Frecuencias dominantes (Hz):", np.sort(freqs[:mitad][top]))
    print(f"Resolución en frecuencia: {fs/N:.3f} Hz")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, x, lw=0.8)
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("x(t)")
    ax[0].set_title("Señal en el tiempo")
    ax[1].plot(freqs[:mitad], amp)
    ax[1].set_xlim(0, 60)
    ax[1].set_xlabel("Frecuencia (Hz)"); ax[1].set_ylabel("Amplitud")
    ax[1].set_title("Espectro (DFT)")
    plt.tight_layout()
    plt.savefig("dft_oscilador.png", dpi=150)
    plt.show()