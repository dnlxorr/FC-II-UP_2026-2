import numpy as np


def tiempo_de_caida(H, v_t, g=9.8, tol=1e-8, max_iter=100):
    """Tiempo para caer una altura H con resistencia lineal (Newton)."""
    tau = v_t / g
    t = H / v_t  # aproximacion inicial ingenua (sin resistencia del aire)
    for i in range(max_iter):
        f = v_t*t - v_t*tau*(1 - np.exp(-t/tau)) - H
        v = v_t*(1 - np.exp(-t/tau))   # f'(t) = velocidad instantanea
        t_nuevo = t - f / v
        if abs(t_nuevo - t) < tol:
            return t_nuevo, i + 1
        t = t_nuevo
    raise RuntimeError("El metodo de Newton no convergio")


# Paracaidista: velocidad terminal 55 m/s, cae H = 1000 m
v_t, H = 55.0, 1000.0
t, n_iter = tiempo_de_caida(H, v_t)

print(f"Tiempo de caida t = {t:.6f} s")
print(f"Convergio en {n_iter} iteraciones")
print(f"Velocidad de impacto = {v_t*(1 - np.exp(-t*9.8/v_t)):.4f} m/s")