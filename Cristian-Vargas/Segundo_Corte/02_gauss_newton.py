"""
02 - Método de Gauss-Newton (y Newton multivariable)
=====================================================
Parte A (Newton): partícula en un potencial armónico 2D
        U(x, y) = x^2 + 2 y^2      (k_x = 2 N/m, k_y = 4 N/m)
    Paso de Newton:  H * Delta = -grad U   ->  x_nuevo = x + Delta
    Como U es cuadrática, llega al mínimo en UN solo paso.

Parte B (Gauss-Newton): ajuste de la descarga de un condensador
        V(t)/V0 = exp(-lambda * t)      (lambda = 1/RC, desconocido)
    Residuos:   r_i = modelo(t_i) - dato_i
    Jacobiana:  J_i = d r_i / d lambda
    Paso:       (J^T J) * Delta = -J^T r        ->  lambda_nuevo = lambda + Delta
    (se desprecia el término con segundas derivadas de la hessiana).

Ejecutar:  python 02_gauss_newton.py
"""
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Parte A: Newton multivariable sobre el potencial 2D
# =====================================================================
print("=== Parte A: Newton en U = x^2 + 2 y^2 ===")


def grad_U(p):
    x, y = p
    return np.array([2 * x, 4 * y])          # (dU/dx, dU/dy)


H = np.array([[2.0, 0.0],                    # hessiana (constante, U es cuadrática)
              [0.0, 4.0]])

p = np.array([2.0, 1.0])                     # punto inicial (m)
print("punto inicial:", p)
Delta = np.linalg.solve(H, -grad_U(p))       # se resuelve H*Delta = -grad (no se invierte H)
p = p + Delta
print("paso Delta   :", Delta)
print("punto nuevo  :", p, " <- mínimo exacto (0, 0) en un paso\n")

# =====================================================================
# Parte B: Gauss-Newton para ajustar lambda
# =====================================================================
print("=== Parte B: Gauss-Newton, descarga de un condensador ===")

t = np.array([1.0, 2.0, 3.0])                # tiempos (s)
y = np.array([0.62, 0.38, 0.22])             # V/V0 medido


def residuos(lam):
    return np.exp(-lam * t) - y              # r = modelo - dato


def jacobiana(lam):
    # d/d(lambda) de exp(-lambda t) = -t exp(-lambda t); una columna (1 parámetro)
    return (-t * np.exp(-lam * t)).reshape(-1, 1)


lam = 0.4                                    # valor inicial (1/s)
print(f"{'it':>2} {'lambda':>10} {'S=0.5*sum(r^2)':>16} {'Delta':>11}")
for it in range(6):
    r = residuos(lam)
    J = jacobiana(lam)
    S = 0.5 * np.sum(r**2)

    # Ecuación normal de Gauss-Newton:  (J^T J) Delta = -J^T r
    Delta = np.linalg.solve(J.T @ J, -J.T @ r)[0]
    print(f"{it:>2} {lam:10.6f} {S:16.6f} {Delta:+11.6f}")

    lam += Delta
    if abs(Delta) < 1e-8:                    # criterio de parada
        break

print(f"\nlambda ajustado = {lam:.5f} 1/s  ->  RC = 1/lambda = {1 / lam:.4f} s")

# ---------------- Gráfica ----------------
tt = np.linspace(0, 4, 200)
plt.plot(tt, np.exp(-0.4 * tt), "--", color="gray", label="inicial (λ = 0.4)")
plt.plot(tt, np.exp(-lam * tt), "r", label=f"ajustado (λ = {lam:.4f})")
plt.plot(t, y, "ko", label="datos")
plt.xlabel("t (s)")
plt.ylabel("V / V0")
plt.title("Ajuste por Gauss-Newton")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
