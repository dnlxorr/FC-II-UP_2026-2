#=====================================================
#5. Transformada discreta de Fourier (DFT)
#Largo de un péndulo a partir de su oscilación
#=====================================================
import numpy as np

g = 9.8   #gravedad (m/s²)

fs = 4.0                  #muestras por segundo
N = 32                    #número de muestras (8 s)
t = np.arange(N)/fs       #instantes de medición
theta = 0.1*np.cos(2*np.pi*0.5*t)   #ángulo medido (rad)

n = np.arange(N)
X = np.zeros(N, dtype=complex)
for k in range(N):
    X[k] = np.sum(theta*np.exp(-2j*np.pi*k*n/N))   #DFT

f = np.arange(N)*fs/N     #frecuencia de cada k (Hz)

print("5. DFT")
for k in range(N//2 + 1):
    print("   k =", k, "  f =", f[k], "Hz   |X| =", round(abs(X[k]), 4))

kp = np.argmax(np.abs(X[:N//2]))   #k del pico
L = g/(2*np.pi*f[kp])**2           #largo del péndulo
print("   frecuencia =", f[kp], "Hz  ->  L =", round(L, 4), "m")