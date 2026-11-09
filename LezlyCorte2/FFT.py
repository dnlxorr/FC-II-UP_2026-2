import numpy as np

# ---------- 1. FFT recursiva ----------
def fft(x):
    N = len(x)
    if N == 1:                       # caso base
        return x
    par   = fft(x[0::2])             # muestras pares
    impar = fft(x[1::2])             # muestras impares
    W = np.exp(-2j*np.pi*np.arange(N//2)/N)
    return np.concatenate([par + W*impar,    # k = 0 ... N/2-1
                           par - W*impar])   # k = N/2 ... N-1

# ---------- 2. Señal del ejemplo ----------
fs, N = 1024, 1024
t = np.arange(N) / fs
x = (np.sin(2*np.pi*50*t)
     + 0.5*np.sin(2*np.pi*120*t))

# ---------- 3. Espectro con nuestra FFT ----------
X = fft(x)
f = np.arange(N) * fs / N            # frecuencias (Hz)
A = 2*np.abs(X) / N                  # amplitud

half = N // 2
print(f[:half][A[:half] > 0.1])
print(A[:half][A[:half] > 0.1])

# ---------- 4. Verificación (opcional) ----------
print("¿Coincide con numpy?", np.allclose(X, np.fft.fft(x)))