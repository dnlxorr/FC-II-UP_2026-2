import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Datos del problema ----------
L = 1.0                    # longitud de la barra (m)
alpha = 0.01               # difusividad térmica (m²/s)
K = 50                     # número de modos que usamos
x = np.linspace(0, L, 401)
dx = x[1] - x[0]

# Temperatura inicial: pulso caliente en el centro
T0 = np.where((x >= 0.4) & (x <= 0.6), 100.0, 0.0)

# ---------- 2. Coeficientes de las transformadas ----------
k = np.arange(1, K + 1)

# Seno: b_k = (2/L) ∫ T0 sen(kπx/L) dx
b = np.array([2/L * np.sum(T0 * np.sin(kk*np.pi*x/L)) * dx for kk in k])

# Coseno: a_k = (2/L) ∫ T0 cos(kπx/L) dx   (incluye k = 0)
a0 = 2/L * np.sum(T0) * dx
a = np.array([2/L * np.sum(T0 * np.cos(kk*np.pi*x/L)) * dx for kk in k])

# ---------- 3. Evolución en el tiempo ----------
def T_seno(t):
    T = np.zeros_like(x)
    for kk, bk in zip(k, b):
        T += bk * np.sin(kk*np.pi*x/L) * np.exp(-alpha*(kk*np.pi/L)**2 * t)
    return T

def T_coseno(t):
    T = np.full_like(x, a0/2)
    for kk, ak in zip(k, a):
        T += ak * np.cos(kk*np.pi*x/L) * np.exp(-alpha*(kk*np.pi/L)**2 * t)
    return T

# ---------- 4. Resultados numéricos ----------
print(f"Temperatura promedio inicial: {np.mean(T0):.1f} °C")
print(f"Caso A (seno),   t = 60 s: T en el centro = {T_seno(60)[200]:.2f} °C")
print(f"Caso B (coseno), t = 60 s: T en el centro = {T_coseno(60)[200]:.2f} °C")

# ---------- 5. Gráficas ----------
tiempos = [0, 5, 20, 60]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for t in tiempos:
    ax[0].plot(x, T_seno(t), label=f"t = {t} s")
    ax[1].plot(x, T_coseno(t), label=f"t = {t} s")

ax[0].set_title("Caso A: extremos a 0 °C (transformada seno)")
ax[1].set_title("Caso B: extremos aislados (transformada coseno)")
for a_ in ax:
    a_.set_xlabel("Posición x (m)")
    a_.grid(True)
    a_.legend()
ax[0].set_ylabel("Temperatura (°C)")

plt.tight_layout()
plt.show()