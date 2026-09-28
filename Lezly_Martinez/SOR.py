"""
SOR (Successive Over-Relaxation) aplicado al mismo problema del potencial
electrostatico entre dos placas. Compara la convergencia para distintos
valores de omega, incluyendo omega=1 (que es el Gauss-Seidel puro que
usamos antes).

Formula que se ejecuta en cada punto (i,j) de la malla:

    V[i,j] = (1-w)*V[i,j] + (w/4)*(V[i-1,j] + V[i,j-1] + V[i+1,j] + V[i,j+1])

Nota: al momento de calcular V[i-1,j] y V[i,j-1] ya se usaron los valores
NUEVOS (k+1) porque el barrido va de arriba-abajo, izquierda-derecha;
V[i+1,j] y V[i,j+1] todavia son los valores VIEJOS (k). Eso es justamente
lo que dice la formula general con las sumas j<i y j>i.
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50
tolerancia = 1e-4
max_iter = 5000


def resolver_potencial(omega):
    V = np.zeros((N, N))
    fila_ini, fila_fin = N // 4, 3 * N // 4
    placa_pos, placa_neg = N // 4, 3 * N // 4

    V[fila_ini:fila_fin, placa_pos] = 100.0
    V[fila_ini:fila_fin, placa_neg] = -100.0

    fijo = np.zeros_like(V, dtype=bool)
    fijo[fila_ini:fila_fin, placa_pos] = True
    fijo[fila_ini:fila_fin, placa_neg] = True

    errores = []
    for it in range(max_iter):
        V_anterior = V.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                if not fijo[i, j]:
                    # ---- aqui se ejecuta la formula de SOR ----
                    gs = 0.25 * (V[i+1, j] + V[i-1, j] +
                                 V[i, j+1] + V[i, j-1])
                    V[i, j] = (1 - omega) * V[i, j] + omega * gs
                    # --------------------------------------------
        error = np.max(np.abs(V - V_anterior))
        errores.append(error)
        if error < tolerancia:
            break
    return errores, it + 1


# Probamos varios valores de omega, incluyendo omega=1 (Gauss-Seidel)
valores_omega = [0.8, 1.0, 1.5, 1.8, 1.95]
resultados = {}

for w in valores_omega:
    errores, n_iter = resolver_potencial(w)
    resultados[w] = errores
    print(f"omega = {w:4.2f}  ->  convergio en {n_iter} iteraciones")

# ---------------------------------------------------------
# Grafica comparativa de convergencia
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))
for w, errores in resultados.items():
    etiqueta = f"$\\omega$ = {w}" + ("  (Gauss-Seidel)" if w == 1.0 else "")
    plt.plot(errores, label=etiqueta)

plt.yscale('log')
plt.xlabel("Iteración")
plt.ylabel("Error máximo (escala log)")
plt.title("Convergencia de SOR para distintos valores de $\\omega$")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("sor_convergencia_comparada.png", dpi=150)
plt.show()