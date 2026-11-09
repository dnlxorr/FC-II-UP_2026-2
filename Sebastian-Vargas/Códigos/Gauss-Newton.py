#=====================================================
#2. Método de Gauss-Newton
#Constante de tiempo de un capacitor que se descarga
#=====================================================
import numpy as np

V0 = 12.0                                  #voltaje inicial (V)
t = np.array([0, 1, 2, 3, 4])              #tiempo (s)
V = np.array([12.0, 7.3, 4.4, 2.7, 1.6])   #voltaje medido (V)

tau = 1.0   #suposición inicial (s)

print("2. Gauss-Newton")
for i in range(20):
    modelo = V0*np.exp(-t/tau)        #lo que predice el modelo
    r = V - modelo                    #residuos
    J = modelo*t/tau**2               #derivada del modelo respecto a tau
    dtau = np.sum(J*r) / np.sum(J*J)  #paso de Gauss-Newton
    tau = tau + dtau
    print("   iteración", i + 1, ": tau =", round(tau, 6), "s")
    if abs(dtau) < 1e-8:
        break