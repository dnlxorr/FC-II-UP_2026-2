import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#1. Parametros (unidades con hbar = 1)
hbar = 1.0
m = 1.0
sigma0 = 1.0
k0 = 2.0
L = 200.0
N = 1024          #potencia de 2
dt = 0.05
pasos = 160       #tiempo final = pasos*dt = 8

#2. Malla en x
x = np.linspace(-L/2, L/2, N, endpoint=False)
dx = x[1] - x[0]

#3. Malla en k (orden de numpy: positivos y luego negativos)
k = 2*np.pi*np.fft.fftfreq(N, d=dx)

#4. Funcion de onda inicial
psi = (1/(sigma0**2*np.pi))**0.25 * np.exp(-x**2/(2*sigma0**2)) * np.exp(1j*k0*x)

#5. Centro y desviacion estandar de |psi|^2
def momentos(psi):
    P = np.abs(psi)**2
    xm = np.sum(x*P)*dx
    x2 = np.sum(x**2*P)*dx
    return xm, np.sqrt(x2 - xm**2)

#Comprobaciones iniciales
print("dx =", dx, "  k_max =", np.pi/dx)
print("Norma inicial:", np.sum(np.abs(psi)**2)*dx)
print("Desv. est. inicial:", momentos(psi)[1], "  esperada:", sigma0/np.sqrt(2))

#6. Operador de evolucion en k (se calcula una vez)
U = np.exp(-1j*hbar*k**2/(2*m)*dt)

#7. Evolucion temporal
t = [0.0]
c = [momentos(psi)[0]]
s = [momentos(psi)[1]]
cuadros = [np.abs(psi)**2]          #|psi|^2 en cada paso (para la animacion)

for n in range(1, pasos+1):
    phi = np.fft.fft(psi)      #a espacio k
    phi = phi*U                #evolucion cinetica
    psi = np.fft.ifft(phi)     #de vuelta a x
    xm, sx = momentos(psi)
    t.append(n*dt)
    c.append(xm)
    s.append(sx)
    cuadros.append(np.abs(psi)**2)

t = np.array(t)
c = np.array(c)
s = np.array(s)

#8. Solucion analitica
s_teo = sigma0*np.sqrt(1 + (hbar*t/(m*sigma0**2))**2)
c_teo = hbar*k0*t/m

#9. Comprobaciones finales
P = np.abs(psi)**2
print("Norma final:", np.sum(P)*dx)
print("Prob. cerca del borde (|x|>80):", np.sum(P[np.abs(x) > 80])*dx)
print("Error max en sigma:", np.max(np.abs(np.sqrt(2)*s - s_teo)))
print("Error max en centro:", np.max(np.abs(c - c_teo)))

#10. Graficas (multipanel)
fig, ax = plt.subplots(2, 1, figsize=(7, 8))
for n in range(0, pasos+1, 40):
    ax[0].plot(x, cuadros[n], label=f"t = {t[n]:.0f}")
ax[0].set_xlim(-15, 40)
ax[0].set_xlabel("x")
ax[0].set_ylabel("|psi(x,t)|^2")
ax[0].legend()

ax[1].plot(t, np.sqrt(2)*s, "o", ms=3, label="numerico (sqrt2 * desv. est.)")
ax[1].plot(t, s_teo, "-", label="analitico")
ax[1].set_xlabel("t")
ax[1].set_ylabel("sigma(t)")
ax[1].legend()

plt.tight_layout()
plt.savefig("paquete_gaussiano.png", dpi=130)

#11. Animacion
fig2, ax2 = plt.subplots(figsize=(7, 4))
linea_num, = ax2.plot(x, cuadros[0], label="numerico")
linea_teo, = ax2.plot(x, cuadros[0], "--", label="analitico")
ax2.set_xlim(-15, 40)
ax2.set_ylim(0, 0.6)
ax2.set_xlabel("x")
ax2.set_ylabel("|psi(x,t)|^2")
ax2.legend(loc="upper right")
titulo = ax2.set_title("t = 0.00")
fig2.tight_layout()

def actualizar(n):
    #|psi|^2 analitico: gaussiana centrada en c_teo con ancho s_teo
    P_teo = np.exp(-(x - c_teo[n])**2/s_teo[n]**2)/(np.sqrt(np.pi)*s_teo[n])
    linea_num.set_ydata(cuadros[n])
    linea_teo.set_ydata(P_teo)
    titulo.set_text(f"t = {t[n]:.2f}")
    return linea_num, linea_teo, titulo

anim = FuncAnimation(fig2, actualizar, frames=len(cuadros), interval=30)
anim.save("paquete_gaussiano.gif", writer="pillow", fps=30)

plt.show()