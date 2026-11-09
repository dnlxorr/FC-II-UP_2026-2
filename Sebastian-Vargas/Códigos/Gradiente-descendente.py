#=====================================================
#3. Gradiente descendente
#Equilibrio de una masa colgada de un resorte
#=====================================================
import numpy as np

g = 9.8   #gravedad (m/s²)

k = 50.0   #constante del resorte (N/m)
m = 1.0    #masa (kg)

def dU(y):
    #Derivada de U = ½ k y² - m g y
    return k*y - m*g

y = 0.0       #empieza con el resorte sin estirar
alfa = 0.01   #tamaño de paso

for i in range(1000):
    y = y - alfa*dU(y)   #bajar por la pendiente
    if abs(dU(y)) < 1e-8:
        break

print("3. Gradiente descendente")
print("   iteraciones:", i + 1)
print("   estiramiento =", round(y, 6), "m   (teoría mg/k =", m*g/k, "m)")