#=====================================================
#1. Búsqueda de la razón dorada
#Ángulo de máximo alcance de un proyectil
#=====================================================
import numpy as np

g = 9.8   #gravedad (m/s²)

v0 = 20.0   #rapidez inicial (m/s)

def alcance(th):
    #Alcance horizontal para un ángulo th en grados
    return v0**2 * np.sin(2*np.radians(th)) / g

phi = (np.sqrt(5) - 1) / 2   #razón dorada ≈ 0.618
a, b = 0.0, 90.0             #intervalo de búsqueda (grados)

while b - a > 1e-5:
    x1 = b - phi*(b - a)
    x2 = a + phi*(b - a)
    if alcance(x1) > alcance(x2):
        b = x2               #el máximo está en [a, x2]
    else:
        a = x1               #el máximo está en [x1, b]

th = (a + b) / 2
print("1. Razón dorada")
print("   ángulo óptimo =", round(th, 4), "grados")
print("   alcance máximo =", round(alcance(th), 4), "m")