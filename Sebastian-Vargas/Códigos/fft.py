#=====================================================
#7. Transformada rápida de Fourier (FFT)
#Frecuencias de una cuerda de guitarra
#=====================================================
import numpy as np

def fft(x):
    #FFT recursiva (N potencia de 2)
    N = len(x)
    if N == 1:
        return x.astype(complex)
    par = fft(x[0::2])     #FFT de las muestras pares
    impar = fft(x[1::2])   #FFT de las muestras impares
    w = np.exp(-2j*np.pi*np.arange(N//2)/N)
    return np.concatenate([par + w*impar, par - w*impar])

Lc = 0.65    #largo de la cuerda (m)
fs = 1024    #muestras por segundo
N = 1024     #número de muestras (1 s)
t = np.arange(N)/fs
y = np.sin(2*np.pi*110*t) + 0.5*np.sin(2*np.pi*220*t)   #fundamental + 2º armónico

Y = fft(y)
amp = 2*np.abs(Y[:N//2])/N     #amplitud de cada frecuencia
f = np.arange(N//2)*fs/N

print("7. FFT")
print("   frecuencias:", f[amp > 0.1], "Hz")
print("   amplitudes:", np.round(amp[amp > 0.1], 4))
print("   ¿igual a numpy?", np.allclose(Y, np.fft.fft(y)))
print("   operaciones DFT ~", N**2, "   FFT ~", int(N*np.log2(N)))
print("   velocidad de onda =", 2*Lc*f[amp > 0.1][0], "m/s")