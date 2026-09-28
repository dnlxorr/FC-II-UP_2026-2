"""
Metodo de relajacion (Gauss-Seidel / SOR) para un sistema de N variables:
cadena de N masas colgando en serie, conectadas por resortes.

Formula de relajacion para cada masa i (deducida minimizando la energia
potencial del sistema):

    x_i = (k_i * x_(i-1) + k_(i+1) * x_(i+1) + m_i * g) / (k_i + k_(i+1))

con x_0 = 0 (techo fijo) y k_(N+1) = 0 (extremo libre, ultima masa).
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.8

# ---------------------------------------------------------
# 1. Parametros fisicos del sistema (N=3 masas, pero el codigo
#    funciona para cualquier N con solo cambiar estas listas)
# ---------------------------------------------------------
m = np.array([1.0, 1.0, 1.0])          # masas [kg]
k = np.array([10.0, 10.0, 10.0])       # constantes de resorte [N/m]
N = len(m)

# ---------------------------------------------------------
# 2. Metodo de relajacion (SOR)
# ---------------------------------------------------------
def resolver_cadena(m, k, omega=1.0, tol=1e-8, max_iter=1000):
    N = len(m)
    x = np.zeros(N)
    errores = []

    for it in range(max_iter):
        x_anterior = x.copy()

        for i in range(N):
            x_prev = x[i-1] if i > 0 else 0.0          # vecino de arriba
            k_next = k[i+1] if i+1 < N else 0.0          # resorte de abajo
            x_next = x[i+1] if i+1 < N else 0.0          # vecino de abajo

            gs = (k[i]*x_prev + k_next*x_next + m[i]*g) / (k[i] + k_next)
            x[i] = (1 - omega)*x[i] + omega*gs

        error = np.max(np.abs(x - x_anterior))
        errores.append(error)
        if error < tol:
            break

    return x, errores, it + 1


# ---------------------------------------------------------
# 3. Resolver con distintos omega para comparar convergencia
# ---------------------------------------------------------
valores_omega = [1.0, 1.3, 1.5]
resultados = {}

for w in valores_omega:
    x_final, errores, n_iter = resolver_cadena(m, k, omega=w)
    resultados[w] = errores
    print(f"omega = {w:.1f}  ->  {n_iter:3d} iteraciones  "
          f"->  x = {np.round(x_final, 4)}")

# ---------------------------------------------------------
# 4. Graficas: convergencia y perfil final de desplazamientos
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for w, errores in resultados.items():
    ax1.plot(errores, label=f"$\\omega$ = {w}")
ax1.set_yscale('log')
ax1.set_xlabel("Iteración")
ax1.set_ylabel("Error máximo (escala log)")
ax1.set_title("Convergencia del método de relajación")
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

x_final, _, _ = resolver_cadena(m, k, omega=1.5)
posiciones = np.arange(1, N+1)
ax2.plot(x_final, posiciones, 'o-', color='tab:blue')
ax2.invert_yaxis()
ax2.set_xlabel("Desplazamiento x_i (m)")
ax2.set_ylabel("Masa i")
ax2.set_yticks(posiciones)
ax2.set_title("Perfil de desplazamientos en equilibrio")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cadena_masas_resultado.png", dpi=150)
plt.show()