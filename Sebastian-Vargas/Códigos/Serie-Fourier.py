#=====================================================
#4. Series de Fourier
#Voltaje de un rectificador de media onda
#=====================================================
import numpy as np

Vp = 170.0           #voltaje pico de la red (V)
T = 1/60             #periodo (s), red de 60 Hz
w0 = 2*np.pi/T       #frecuencia angular (rad/s)

t = np.linspace(0, T, 10001)
V = np.where(t < T/2, Vp*np.sin(w0*t), 0.0)   #media onda

#Coeficiente a0
a0 = 2/T * np.trapezoid(V, t)

#Coeficientes an y bn
N = 20
an = np.zeros(N + 1)
bn = np.zeros(N + 1)
for n in range(1, N + 1):
    an[n] = 2/T * np.trapezoid(V*np.cos(n*w0*t), t)
    bn[n] = 2/T * np.trapezoid(V*np.sin(n*w0*t), t)

print("4. Serie de Fourier")
print("   a0 =", round(a0, 4), "V   (valor medio a0/2 =", round(a0/2, 4), "V)")
for n in range(1, 7):
    print("   n =", n, "  an =", round(an[n], 4), "  bn =", round(bn[n], 4))

#Reconstrucción con los N términos
def serie(tp):
    s = a0/2
    for n in range(1, N + 1):
        s += an[n]*np.cos(n*w0*tp) + bn[n]*np.sin(n*w0*tp)
    return s

print("   V(T/4) serie =", round(serie(T/4), 3), "V   (real:", Vp, "V)")
print("   V(3T/4) serie =", round(serie(3*T/4), 3), "V   (real: 0 V)")