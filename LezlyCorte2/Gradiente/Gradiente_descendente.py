"""
Ajuste de un oscilador armónico amortiguado con datos ruidosos:
    x(t) = A * exp(-g t) * cos(w t + phi)
Parámetros a estimar: p = [A, g, w, phi]
Métodos: Gauss-Newton y gradiente descendente sobre S(p) = 1/2 * sum(r_i^2)
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Modelo y jacobiano ----------
def modelo(p, t):
    A, g, w, phi = p
    return A * np.exp(-g * t) * np.cos(w * t + phi)

def jacobiano(p, t):
    A, g, w, phi = p
    e = np.exp(-g * t)
    c = np.cos(w * t + phi)
    s = np.sin(w * t + phi)
    J = np.empty((t.size, 4))
    J[:, 0] = e * c               # dx/dA
    J[:, 1] = -A * t * e * c      # dx/dg
    J[:, 2] = -A * t * e * s      # dx/dw
    J[:, 3] = -A * e * s          # dx/dphi
    return J

def residuo(p, t, x):
    return modelo(p, t) - x

def costo(p, t, x):
    r = residuo(p, t, x)
    return 0.5 * np.sum(r**2)

# ---------- Gauss-Newton ----------
def gauss_newton(p0, t, x, tol=1e-10, max_iter=100):
    p = np.array(p0, float)
    hist = [costo(p, t, x)]
    for k in range(max_iter):
        r = residuo(p, t, x)
        J = jacobiano(p, t)
        # Resolver (J^T J) dp = -J^T r
        dp = np.linalg.solve(J.T @ J, -J.T @ r)
        p = p + dp
        hist.append(costo(p, t, x))
        if np.linalg.norm(dp) < tol:
            break
    return p, np.array(hist)

# ---------- Gradiente descendente ----------
def gradiente_descendente(p0, t, x, alpha=1e-4, tol=1e-8, max_iter=20000):
    p = np.array(p0, float)
    hist = [costo(p, t, x)]
    for k in range(max_iter):
        grad = jacobiano(p, t).T @ residuo(p, t, x)
        p_new = p - alpha * grad
        hist.append(costo(p_new, t, x))
        if np.linalg.norm(p_new - p) < tol:
            p = p_new
            break
        p = p_new
    return p, np.array(hist)

# ---------- Datos sintéticos ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    p_real = np.array([2.0, 0.3, 4.0, 0.5])      # A, gamma, omega, phi
    t = np.linspace(0, 10, 200)
    x = modelo(p_real, t) + rng.normal(0, 0.05, t.size)

    p0 = [1.5, 0.2, 3.8, 0.0]                    # estimación inicial

    p_gn, h_gn = gauss_newton(p0, t, x)
    p_gd, h_gd = gradiente_descendente(p0, t, x)

    nombres = ["A", "gamma", "omega", "phi"]
    print(f"{'param':>6} {'real':>8} {'G-N':>10} {'GD':>10}")
    for n, a, b, c in zip(nombres, p_real, p_gn, p_gd):
        print(f"{n:>6} {a:8.4f} {b:10.4f} {c:10.4f}")
    print(f"\nIteraciones G-N: {len(h_gn)-1} | GD: {len(h_gd)-1}")
    print(f"Costo final G-N: {h_gn[-1]:.5f} | GD: {h_gd[-1]:.5f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, x, ".", ms=4, label="Datos ruidosos")
    ax[0].plot(t, modelo(p_gn, t), label="Gauss-Newton")
    ax[0].plot(t, modelo(p_gd, t), "--", label="Grad. descendente")
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("x (m)")
    ax[0].set_title("Ajuste del oscilador amortiguado"); ax[0].legend()
    ax[1].semilogy(h_gn, "o-", label="Gauss-Newton")
    ax[1].semilogy(h_gd, label="Grad. descendente")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Iteración"); ax[1].set_ylabel("S(p)")
    ax[1].set_title("Convergencia"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("ajuste_oscilador.png", dpi=150)
    plt.show()