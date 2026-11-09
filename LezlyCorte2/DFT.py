import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Generar la señal ----------
fs = 1000                 # frecuencia de muestreo (Hz)
T = 1.0                   # duración (s)
N = int(fs * T)           # número de muestras
t = np.arange(N) / fs     # vector de tiempo

# Señal: 50 Hz (amplitud 2) + 120 Hz (amplitud 1) + ruido
np.random.seed(0)
x = 2*np.sin(2*np.pi*50*t) + 1*np.sin(2*np.pi*120*t)
x = x + 0.8*np.random.randn(N)

# ---------- 2. DFT escrita "a mano" ----------
def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)   # matriz de la DFT
    return M @ x

X = dft(x)

# ---------- 3. Frecuencias y amplitudes ----------
frecuencias = np.arange(N) * fs / N
amplitud = 2 * np.abs(X) / N              # normalización

# Solo la mitad positiva (la otra mitad es simétrica)
mitad = N // 2

# ---------- 4. Verificar con la FFT de numpy ----------
X_fft = np.fft.fft(x)
print("¿Coincide con numpy?", np.allclose(X, X_fft))

# ---------- 5. Picos encontrados ----------
indices = np.argsort(amplitud[:mitad])[-2:]
for i in sorted(indices):
    print(f"Frecuencia: {frecuencias[i]:.1f} Hz  |  Amplitud: {amplitud[i]:.2f}")

# ---------- 6. Gráficas ----------
fig, ax = plt.subplots(2, 1, figsize=(9, 7))

ax[0].plot(t[:200], x[:200], color="steelblue")
ax[0].set_title("Señal medida (dominio del tiempo)")
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Voltaje (V)")
ax[0].grid(True)

ax[1].plot(frecuencias[:mitad], amplitud[:mitad], color="crimson")
ax[1].set_title("Espectro (dominio de la frecuencia)")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Amplitud (V)")
ax[1].set_xlim(0, 200)
ax[1].grid(True)

plt.tight_layout()
plt.show()