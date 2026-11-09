"""
Evolución temporal de un paquete de ondas gaussiano libre (V(x) = 0)
Método: FFT (split en espacio de momentos)

Universidad de Pamplona - Facultad de Ciencias Básicas
Física Computacional

Idea central del método
-----------------------
    psi(x, t+dt) = IFFT{ exp(-i*hbar*k^2*dt/(2m)) * FFT{ psi(x, t) } }

En el espacio de momentos el operador de energía cinética es diagonal
(solo multiplica), por eso la FFT convierte una derivada segunda en un
producto. Como V = 0, esta evolución es EXACTA (no hay error de Trotter).

Unidades naturales: hbar = 1, m = 1.
Todas las rutas son relativas (las figuras se guardan en ./figuras).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # permite guardar figuras sin abrir ventana
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# A. PARÁMETROS
# ---------------------------------------------------------------------
hbar = 1.0
m = 1.0
sigma0 = 1.0          # ancho inicial del paquete
k0 = 5.0              # momento inicial (velocidad de grupo v = hbar*k0/m)

L = 200.0             # tamaño del dominio: x en [-L/2, L/2)
N = 2**11             # número de puntos (potencia de 2 -> FFT eficiente)

dt = 0.05             # paso de tiempo
T = 10.0              # tiempo total de simulación
n_pasos = int(round(T / dt))

# ---------------------------------------------------------------------
# 1. DISCRETIZACIÓN DE x Y DE k
# ---------------------------------------------------------------------
dx = L / N
x = -L / 2 + dx * np.arange(N)            # [-L/2, L/2): sin repetir el extremo
# Frecuencias de la DFT: 0, dk, 2dk, ..., -2dk, -dk  (orden de numpy)
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)   # k_n = 2*pi*n/L
dk = 2 * np.pi / L
k_max = np.pi / dx                        # frecuencia de Nyquist

print(f"dx = {dx:.4f},  dk = {dk:.4f},  k_Nyquist = {k_max:.2f}")
print(f"k0 + 5/sigma0 = {k0 + 5/sigma0:.1f}  (debe ser << k_Nyquist)")

# ---------------------------------------------------------------------
# B. FUNCIÓN DE ONDA INICIAL (normalizada)
# ---------------------------------------------------------------------
psi0 = (1.0 / (sigma0**2 * np.pi) ** 0.25) \
       * np.exp(-x**2 / (2 * sigma0**2)) * np.exp(1j * k0 * x)

# ---------------------------------------------------------------------
# C. OPERADOR DE EVOLUCIÓN CINÉTICA EN ESPACIO k  (se calcula UNA vez)
# ---------------------------------------------------------------------
U_k = np.exp(-1j * hbar * k**2 / (2 * m) * dt)


def paso(psi):
    """Un paso dt:  x -> k  (FFT),  multiplicar por U_k,  k -> x  (IFFT)."""
    phi = np.fft.fft(psi)          # 1. cambio de base a momentos
    phi = phi * U_k                # 2. evolución cinética (diagonal en k)
    return np.fft.ifft(phi)        # 3. regreso al espacio real


# ---------------------------------------------------------------------
# Funciones auxiliares: promedios con la densidad de probabilidad
# ---------------------------------------------------------------------
def momentos_x(psi):
    """Devuelve norma, <x> y Delta x de |psi|^2 (suma de Riemann, dx)."""
    rho = np.abs(psi) ** 2
    norma = np.sum(rho) * dx
    x_med = np.sum(x * rho) * dx / norma
    var = np.sum((x - x_med) ** 2 * rho) * dx / norma
    return norma, x_med, np.sqrt(var)


def momentos_k(psi):
    """<k> y Delta k calculados con |phi(k)|^2 (phi = FFT de psi)."""
    rho_k = np.abs(np.fft.fft(psi)) ** 2
    rho_k = rho_k / np.sum(rho_k)              # normalización discreta
    k_med = np.sum(k * rho_k)
    var = np.sum((k - k_med) ** 2 * rho_k)
    return k_med, np.sqrt(var)


def psi_analitica(t):
    """Solución exacta del paquete gaussiano libre (ver el .tex)."""
    tau = hbar * t / (m * sigma0**2)
    b = x - hbar * k0 * t / m
    return (1.0 / (sigma0**2 * np.pi) ** 0.25) / np.sqrt(1 + 1j * tau) \
        * np.exp(-b**2 / (2 * sigma0**2 * (1 + 1j * tau))) \
        * np.exp(1j * (k0 * x - hbar * k0**2 * t / (2 * m)))


def sigma_analitico(t):
    """sigma(t) = sigma0 * sqrt(1 + (hbar t / (m sigma0^2))^2)."""
    return sigma0 * np.sqrt(1 + (hbar * t / (m * sigma0**2)) ** 2)


# ---------------------------------------------------------------------
# EVOLUCIÓN TEMPORAL
# ---------------------------------------------------------------------
tiempos = np.arange(n_pasos + 1) * dt
psi = psi0.copy()

historia_rho = np.zeros((n_pasos + 1, N))      # |psi|^2 en cada paso
sigma_num = np.zeros(n_pasos + 1)
x_med_num = np.zeros(n_pasos + 1)
norma_t = np.zeros(n_pasos + 1)
k_med_t = np.zeros(n_pasos + 1)
dk_t = np.zeros(n_pasos + 1)

for n in range(n_pasos + 1):
    norma, x_med, dx_std = momentos_x(psi)
    km, dkk = momentos_k(psi)
    historia_rho[n] = np.abs(psi) ** 2
    # IMPORTANTE: sigma(t) de la guía es el parámetro de exp(-x^2/(2 sigma^2)),
    # pero la desviación estándar de |psi|^2 es sigma/sqrt(2).
    sigma_num[n] = np.sqrt(2) * dx_std
    x_med_num[n] = x_med
    norma_t[n] = norma
    k_med_t[n] = km
    dk_t[n] = dkk
    if n < n_pasos:
        psi = paso(psi)

psi_final = psi.copy()

# ---------------------------------------------------------------------
# VERIFICACIONES NUMÉRICAS
# ---------------------------------------------------------------------
sigma_teo = sigma_analitico(tiempos)
x_med_teo = hbar * k0 * tiempos / m
err_sigma = np.abs(sigma_num - sigma_teo)
err_psi = np.max(np.abs(psi_final - psi_analitica(T)))

# La evolución es exacta: un solo paso grande T debe dar lo mismo que muchos pasos dt
U_T = np.exp(-1j * hbar * k**2 / (2 * m) * T)
psi_un_paso = np.fft.ifft(np.fft.fft(psi0) * U_T)
err_un_paso = np.max(np.abs(psi_un_paso - psi_final))

print("\n--- Verificaciones (t = T = %.1f) ---" % T)
print(f"Norma final                    : {norma_t[-1]:.12f}  (ideal 1)")
print(f"<x>(T) numérico / teórico      : {x_med_num[-1]:.6f} / {x_med_teo[-1]:.6f}")
print(f"<k>(T) numérico (ideal k0={k0})  : {k_med_t[-1]:.6f}")
print(f"Delta k inicial / final        : {dk_t[0]:.6f} / {dk_t[-1]:.6f} "
      f"(teórico 1/(sqrt2 sigma0) = {1/(np.sqrt(2)*sigma0):.6f})")
print(f"sigma(T) numérico / analítico  : {sigma_num[-1]:.6f} / {sigma_teo[-1]:.6f}")
print(f"max |sigma_num - sigma_teo|    : {err_sigma.max():.2e}")
print(f"max |psi_num - psi_analitica|  : {err_psi:.2e}")
print(f"Un paso T vs {n_pasos} pasos dt        : {err_un_paso:.2e}")

# ---------------------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------------------
os.makedirs("figuras", exist_ok=True)

# --- Figura 1: multipanel de la densidad de probabilidad en varios tiempos
t_mostrar = [0.0, 2.5, 5.0, 10.0]
fig, ejes = plt.subplots(len(t_mostrar), 1, figsize=(8, 9), sharex=True)
for ax, tm in zip(ejes, t_mostrar):
    n = int(round(tm / dt))
    ax.fill_between(x, historia_rho[n], color="#2e8b57", alpha=0.35)
    ax.plot(x, historia_rho[n], color="#006400", lw=1.5, label="FFT (numérico)")
    ax.plot(x, np.abs(psi_analitica(tiempos[n])) ** 2, "k--", lw=0.9,
            label="analítico")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.set_title(rf"$t = {tm:.1f}$,  $\sigma(t) = {sigma_num[n]:.3f}$",
                 fontsize=10, loc="left")
    ax.set_xlim(-20, 70)
    ax.grid(alpha=0.3)
ejes[0].legend(loc="upper right")
ejes[-1].set_xlabel(r"$x$")
fig.suptitle("Dispersión del paquete gaussiano libre", fontsize=13)
fig.tight_layout()
fig.savefig("figuras/fig1_multipanel.png", dpi=200)
plt.close(fig)

# --- Figura 2: mapa x-t de la densidad
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(historia_rho, aspect="auto", origin="lower", cmap="viridis",
               extent=[x[0], x[-1] + dx, 0, T])
ax.plot(x_med_teo, tiempos, "w--", lw=1, label=r"$x = \hbar k_0 t/m$")
ax.set_xlim(-20, 70)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.set_title(r"Mapa espacio-tiempo de $|\psi(x,t)|^2$")
ax.legend(loc="upper left")
fig.colorbar(im, label=r"$|\psi|^2$")
fig.tight_layout()
fig.savefig("figuras/fig2_mapa_xt.png", dpi=200)
plt.close(fig)

# --- Figura 3: sigma(t) numérico vs analítico, y su diferencia
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.2]})
ax1.plot(tiempos, sigma_teo, "k-", lw=2, label=r"analítico $\sigma_0\sqrt{1+(\hbar t/m\sigma_0^2)^2}$")
ax1.plot(tiempos[::10], sigma_num[::10], "o", color="#006400", ms=5,
         label="numérico (FFT)")
ax1.axhline(sigma0, color="gray", ls=":", lw=1)
ax1.set_ylabel(r"$\sigma(t)$")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.semilogy(tiempos, err_sigma + 1e-18, color="#8b0000")
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$|\sigma_{num}-\sigma_{teo}|$")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figuras/fig3_sigma_t.png", dpi=200)
plt.close(fig)

print("\nFiguras guardadas en ./figuras")
