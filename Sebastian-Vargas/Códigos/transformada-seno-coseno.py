#=====================================================
#6. Transformadas seno y coseno de Fourier
#Respuesta de un filtro RC pasabajos
#=====================================================
import numpy as np

tau = 0.5                        #constante de tiempo RC (s)
t = np.linspace(0, 20, 20001)    #de 0 a "infinito"
h = np.exp(-t/tau)/tau           #respuesta al impulso (1/s)

print("6. Transformadas coseno y seno")
for w in [0.0, 1.0, 2.0, 4.0]:
    Fc = np.trapezoid(h*np.cos(w*t), t)   #transformada coseno
    Fs = np.trapezoid(h*np.sin(w*t), t)   #transformada seno
    Fc_teo = 1/(1 + (w*tau)**2)
    Fs_teo = w*tau/(1 + (w*tau)**2)
    print("   w =", w, "  Fc =", round(Fc, 4), "(", round(Fc_teo, 4), ")",
          "  Fs =", round(Fs, 4), "(", round(Fs_teo, 4), ")",
          "  ganancia =", round(np.sqrt(Fc**2 + Fs**2), 4))