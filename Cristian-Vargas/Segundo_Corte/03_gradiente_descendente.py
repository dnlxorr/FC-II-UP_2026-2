"""
03 - Gradiente descendente
===========================
Ejemplo físico: partícula en el potencial armónico anisótropo
        U(x, y) = x^2 + 2 y^2        (k_x = 2 N/m, k_y = 4 N/m)

Regla:  p_nuevo = p - alpha * grad U(p)

Para este potencial A = diag(2, 4), así que cada paso multiplica
    x por (1 - 2*alpha)   e   y por (1 - 4*alpha).
Converge si alpha < 2/lambda_max = 0.5.  El alpha óptimo es
    alpha* = 2/(lambda_min + lambda_max) = 1/3.

Interpretación física: es el movimiento sobreamortiguado  gamma * dp/dt = -grad U
discretizado con Euler explícito, con alpha = dt/gamma.

Ejecutar:  python 03_gradiente_descendente.py
"""
import numpy as np
import matplotlib.pyplot as plt


def U(p):
    return p[0] ** 2 + 2 * p[1] ** 2


def grad_U(p):
    return np.array([2 * p[0], 4 * p[1]])


def gradiente_descendente(p0, alpha, n_iter=40, tol=1e-6):
    """Devuelve la trayectoria (lista de puntos) del método."""
    p = np.array(p0, dtype=float)
    trayectoria = [p.copy()]
    for _ in range(n_iter):
        g = grad_U(p)
        if np.linalg.norm(g) < tol:          # criterio de parada: |grad U| pequeño
            break
        p = p - alpha * g                    # paso en la dirección de máximo descenso
        trayectoria.append(p.copy())
    return np.array(trayectoria)


if __name__ == "__main__":
    p0 = (2.0, 1.0)

    # ---------- Caso 1: alpha = 0.1 (el de la guía) ----------
    tray = gradiente_descendente(p0, alpha=0.1, n_iter=500)
    print("alpha = 0.1")
    print(f"{'n':>3} {'x':>9} {'y':>9} {'U':>10}   (forma cerrada: x=2*0.8^n, y=0.6^n)")
    for n in range(6):
        x, y = tray[n]
        print(f"{n:>3} {x:9.4f} {y:9.4f} {U(tray[n]):10.4f}   "
              f"({2 * 0.8**n:.4f}, {0.6**n:.4f})")
    print(f"iteraciones hasta |grad U| < 1e-6: {len(tray) - 1}")
    print(f"punto final: ({tray[-1][0]:.2e}, {tray[-1][1]:.2e})\n")

    # ---------- Caso 2: alpha óptimo ----------
    alpha_opt = 2 / (2 + 4)
    tray_opt = gradiente_descendente(p0, alpha=alpha_opt, n_iter=500)
    print(f"alpha óptimo = {alpha_opt:.4f}: iteraciones = {len(tray_opt) - 1}")

    # ---------- Caso 3: alpha demasiado grande (diverge) ----------
    tray_div = gradiente_descendente(p0, alpha=0.6, n_iter=8, tol=0)
    print("alpha = 0.6 (> 0.5) diverge:")
    for n, p in enumerate(tray_div[:6]):
        print(f"  n={n}: ({p[0]:+9.4f}, {p[1]:+9.4f})   U = {U(p):10.4f}")

    # ---------- Gráfica: curvas de nivel + trayectoria ----------
    xs = np.linspace(-0.3, 2.8, 200)
    ys = np.linspace(-0.3, 1.9, 200)
    X, Y = np.meshgrid(xs, ys)
    plt.contour(X, Y, X**2 + 2 * Y**2, levels=[1, 2, 3, 4, 6], colors="gray")
    plt.plot(tray[:12, 0], tray[:12, 1], "r.-", label="α = 0.1")
    plt.plot(tray_opt[:12, 0], tray_opt[:12, 1], "b.-", label="α = 1/3 (óptimo)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Gradiente descendente sobre U = x² + 2y²")
    plt.gca().set_aspect("equal")
    plt.legend()
    plt.show()
