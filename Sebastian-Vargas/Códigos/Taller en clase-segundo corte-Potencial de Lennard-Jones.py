import os
import numpy as np
import matplotlib.pyplot as plt

#Taller de Física Computacional: Lennard-Jones y series de Fourier
#Solo se usa numpy (arreglos) y matplotlib (gráficas); los algoritmos están hechos a mano
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

#Potencial de Lennard-Jones (unidades reducidas)
def potencial(x):
    return x**(-12) - x**(-6)

#Derivada del potencial (la usan Newton y el gradiente descendente)
def derivada(x):
    return -12*x**(-13) + 6*x**(-7)

#Segunda derivada (solo para comparar con la teoría)
def derivada2(x):
    return 156*x**(-14) - 42*x**(-8)

#Metodo de bisección (devuelve la raíz y la lista de puntos medios)
def biseccion(f, a, b, e=1e-12):
    fa = f(a)
    historial = []
    for i in range(200):
        c = (a+b)/2
        fc = f(c)
        historial.append(c)
        if fc == 0:
            return c, historial
        if fa*fc < 0:
            b = c
        else:
            a = c
            fa = fc
        if (b-a) < e:
            break
    return (a+b)/2, historial

#Metodo de Newton-Raphson (devuelve la raíz y la lista de iterados)
def newton(f, df, x, e=1e-12):
    historial = [x]
    for i in range(100):
        x = x - f(x)/df(x)
        historial.append(x)
        if abs(historial[-1] - historial[-2]) < e:
            break
    return x, historial

#Metodo de búsqueda dorada (devuelve el mínimo y la lista de puntos medios)
def dorada(f, a, b, e=1e-8):
    r = (np.sqrt(5) - 1)/2
    c = b - r*(b-a)
    d = a + r*(b-a)
    fc = f(c)
    fd = f(d)
    historial = []
    for i in range(200):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - r*(b-a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + r*(b-a)
            fd = f(d)
        historial.append((a+b)/2)
        if (b-a) < e:
            break
    return (a+b)/2, historial

#Metodo del gradiente descendente: x = x - eta*V'(x)
def gradiente(df, x, eta, e=1e-12):
    historial = [x]
    for i in range(5000):
        g = df(x)
        if abs(g) < e:
            break
        x = x - eta*g
        historial.append(x)
    return x, historial

#Fuerza F = -dV/dx con diferencias finitas centradas
def fuerza(x, h):
    return -(potencial(x+h) - potencial(x-h))/(2*h)

#Puntos de retorno: raíces de f(x) = E a cada lado del mínimo xm
def retorno(f, E, xm):
    g = lambda x: f(x) - E
    if g(xm) > 0:
        return None, None          #E está por debajo del fondo: no hay raíces
    x1, h1 = biseccion(g, 0.8, xm)
    x2, h2 = biseccion(g, xm, 10.0)
    return x1, x2

#Regla de Simpson 1/3 para puntos equiespaciados y (número de intervalos par)
def simpson(y, h):
    return h/3*(y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))

#Coeficientes de Fourier a_n y b_n (n = 0,...,nmax) de f en [x0, x0+L]
def coeficientes(f, x0, L, nmax, M):
    x = np.linspace(x0, x0 + L, M + 1)
    h = L/M
    fx = f(x)
    a = []
    b = []
    for n in range(nmax + 1):
        a.append(2/L*simpson(fx*np.cos(2*np.pi*n*x/L), h))
        b.append(2/L*simpson(fx*np.sin(2*np.pi*n*x/L), h))
    return np.array(a), np.array(b)

#Pendiente de la recta log-log (para ver cómo decae el espectro)
def pendiente(n, y):
    return np.polyfit(np.log(n), np.log(y), 1)[0]


#Parte A:
print("PARTE A")

#Punto de corte x0 (V(x0)=0): bisección y Newton
x0_bis, h_bis = biseccion(potencial, 0.8, 1.5)
x0_new, h_new = newton(potencial, derivada, 0.9)
it_bis = len(h_bis)               #iteraciones de la bisección
ev_bis = len(h_bis) + 1           #evaluaciones: V(a) una vez + una por iteración
it_new = len(h_new) - 1           #iteraciones de Newton
ev_new = 2*it_new                 #evaluaciones: V y V' en cada iteración
print("Bisección: x0 =", x0_bis, " iteraciones =", it_bis, " evaluaciones =", ev_bis, " error =", abs(x0_bis - 1))
print("Newton:    x0 =", x0_new, " iteraciones =", it_new, " evaluaciones =", ev_new, " error =", abs(x0_new - 1))
err_new = [abs(x - 1) for x in h_new]
print("Errores de Newton:", err_new)
razones = [err_new[k+1]/err_new[k]**2 for k in range(len(err_new) - 1) if err_new[k] > 1e-5]
razon_teorica = abs(derivada2(1.0)/(2*derivada(1.0)))
print("e(k+1)/e(k)^2 =", razones, " teórico =", razon_teorica)

#Newton partiendo a la derecha del mínimo (x = 1.3): no converge
x_mal, h_mal = newton(potencial, derivada, 1.3)
print("Newton desde 1.3: x final =", x_mal, " tras", len(h_mal) - 1, "iteraciones")

#Mínimo de energía: búsqueda dorada y gradiente descendente
xmin_exacto = 2**(1/6)
xm_dor, h_dor = dorada(potencial, 1.0, 2.0)
xm_gd, h_gd = gradiente(derivada, 1.3, 0.05)
ev_dor = len(h_dor) + 2           #dos evaluaciones al inicio y una por iteración
k = derivada2(xmin_exacto)        #curvatura en el mínimo
print("Dorada:    xmin =", xm_dor, " iteraciones =", len(h_dor), " evaluaciones =", ev_dor, " error =", abs(xm_dor - xmin_exacto), " V =", potencial(xm_dor))
print("Gradiente: xmin =", xm_gd, " iteraciones =", len(h_gd) - 1, " error =", abs(xm_gd - xmin_exacto), " V =", potencial(xm_gd))
print("Exacto: xmin =", xmin_exacto, " V(xmin) = -0.25   k = V''(xmin) =", k, "  2/k =", 2/k)

#Estabilidad del gradiente: converge si 0 < eta < 2/k
tabla_gd = []
for eta in [0.01, 0.05, 1/k, 0.12, 0.14, 0.15]:
    x, h = gradiente(derivada, 1.3, eta)
    converge = abs(x - xmin_exacto) < 1e-6
    ciclo = (not converge) and abs(h[-1] - h[-3]) < 1e-6
    tabla_gd.append([eta, abs(1 - eta*k), len(h) - 1, converge, ciclo, h[-2], h[-1]])
    print("eta =", eta, " rho =", abs(1 - eta*k), " iteraciones =", len(h) - 1, " converge =", converge, " ciclo de periodo 2 =", ciclo)

#Fuerza con diferencias finitas centradas (h = 1e-5) en el xmin de cada método
H = 1e-5
F_dor = fuerza(xm_dor, H)
F_dor_ex = -derivada(xm_dor)
F_gd = fuerza(xm_gd, H)
F_gd_ex = -derivada(xm_gd)
F_en_1 = fuerza(1.0, H)
k_fd = (potencial(xm_gd + 1e-3) - 2*potencial(xm_gd) + potencial(xm_gd - 1e-3))/1e-3**2
print("Fuerza (h=1e-5):  xmin dorada ->", F_dor, " exacta ahí:", F_dor_ex)
print("                  xmin gradiente ->", F_gd, " exacta ahí:", F_gd_ex)
print("                  en x=1 ->", F_en_1, " (exacta 6)")
print("V'' por diferencias (h=1e-3) =", k_fd, " exacta =", k)

#Error de la fuerza al variar h
hs = [10.0**(-j) for j in range(1, 13)]
err_x1 = [abs(fuerza(1.0, h) - (-derivada(1.0))) for h in hs]
err_xm = [abs(fuerza(xm_gd, h) - (-derivada(xm_gd))) for h in hs]
h_mejor = hs[int(np.argmin(err_x1))]
for h, e1, e2 in zip(hs, err_x1, err_xm):
    print("h =", h, " error en x=1:", e1, " error en xmin:", e2)

#Puntos de retorno
x1_m05, x2_m05 = retorno(potencial, -0.5, xmin_exacto)
print("E = -0.5:", x1_m05, x2_m05, " (None = no existen; V(xmin) - E =", potencial(xmin_exacto) + 0.5, ")")
x1, x2 = retorno(potencial, -0.2, xmin_exacto)
s = np.sqrt(1 + 4*(-0.2))
x1_ana = ((1 + s)/2)**(-1/6)
x2_ana = ((1 - s)/2)**(-1/6)
print("E = -0.2:", x1, x2, " analíticos:", x1_ana, x2_ana)
potencial4 = lambda x: 4*potencial(x)         #variante con el factor 4*epsilon
x1_4, x2_4 = retorno(potencial4, -0.5, xmin_exacto)
s4 = np.sqrt(1 + (-0.5))
print("4V, E = -0.5:", x1_4, x2_4, " analíticos:", ((1 + s4)/2)**(-1/6), ((1 - s4)/2)**(-1/6))


#Gráficas de la parte A
#Figura 1: potencial, mínimo y puntos de retorno
plt.figure(figsize=(6.6, 2.9))
plt.subplot(1, 2, 1)
x = np.linspace(0.85, 3.2, 3000)
plt.plot(x, potencial(x), color="C0")
plt.axhline(0, color="gray", lw=0.7)
plt.plot(1.0, 0.0, "o", color="C1", label=r"$x_0=1$ (corte)")
plt.plot(xmin_exacto, -0.25, "s", color="C3", label=r"$x_{\min}=2^{1/6}$")
plt.ylim(-0.5, 3.0)
plt.xlabel("$x$")
plt.ylabel(r"$V(x)=x^{-12}-x^{-6}$")
plt.title("(a) Forma global del potencial")
plt.legend(loc="upper right")
plt.subplot(1, 2, 2)
x = np.linspace(0.95, 2.6, 3000)
plt.plot(x, potencial(x), color="C0")
plt.axhline(-0.2, color="C2")
plt.fill_between(x, -0.2, potencial(x), where=(x >= x1) & (x <= x2), color="C2", alpha=0.25)
plt.plot([x1, x2], [-0.2, -0.2], "o", color="C2", label=rf"$E=-0.2$: $x_1={x1:.4f}$, $x_2={x2:.4f}$")
plt.axhline(-0.5, color="C3", ls="--", label=r"$E=-0.5$: sin intersección")
plt.axhline(-0.25, color="gray", ls=":")
plt.ylim(-0.85, 0.12)
plt.xlabel("$x$")
plt.title("(b) Zoom al pozo y puntos de retorno")
plt.legend(loc="lower right", framealpha=0.95)
plt.tight_layout()
plt.savefig("figs/fig1_potencial.pdf")

#Figura 2: convergencia hacia x0 (error contra evaluaciones de V)
plt.figure(figsize=(4.6, 3.0))
e_bis = np.maximum(np.abs(np.array(h_bis) - 1), 1e-17)
e_new = np.maximum(np.abs(np.array(h_new) - 1), 1e-17)
plt.semilogy(np.arange(1, len(h_bis) + 1) + 1, e_bis, "o-", ms=3, lw=1, label="Bisección")
plt.semilogy(2*np.arange(len(h_new)), e_new, "s-", ms=4, lw=1.2, color="C3", label=r"Newton ($V$ y $V'$ cuentan 2)")
plt.xlabel("evaluaciones de función")
plt.ylabel(r"$|x_k-x_0|$")
plt.title(r"Convergencia hacia $x_0=1$")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figs/fig2_raices.pdf")

#Figura 3: convergencia hacia xmin (la dorada se pide con tolerancia 1e-16 para ver dónde se estanca)
plt.figure(figsize=(4.6, 3.0))
xm_dor2, h_dor2 = dorada(potencial, 1.0, 2.0, 1e-16)
e_dor = np.maximum(np.abs(np.array(h_dor2[:80]) - xmin_exacto), 1e-17)    #con 80 iteraciones ya se ve el estancamiento
e_gd = np.maximum(np.abs(np.array(h_gd) - xmin_exacto), 1e-17)
kk = np.arange(len(h_gd))
plt.semilogy(np.arange(len(e_dor)), e_dor, "o-", ms=2.5, lw=1, label="Búsqueda dorada")
plt.semilogy(kk, e_gd, "s-", ms=3, lw=1, color="C3", label=r"Gradiente ($\eta=0.05$)")
plt.semilogy(kk, e_gd[0]*abs(1 - 0.05*k)**kk, "--", color="k", lw=0.8, label=r"$e_0|1-\eta V''|^k$")
plt.axhline(np.sqrt(2.2e-16), color="gray", ls=":")
plt.ylim(1e-17, 1)
plt.xlabel("iteración $k$")
plt.ylabel(r"$|x_k-2^{1/6}|$")
plt.title(r"Convergencia hacia $x_{\min}$")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figs/fig3_optimizacion.pdf")

#Figura 4: error de la fuerza contra h
plt.figure(figsize=(4.6, 3.0))
plt.loglog(hs, np.maximum(err_x1, 1e-17), "o-", ms=3.5, label=r"en $x=1$ ($F=6$)")
plt.loglog(hs, np.maximum(err_xm, 1e-17), "s-", ms=3.5, color="C3", label=r"en $x_{\min}$ ($F\approx0$)")
plt.loglog([1e-4, 1e-1], [err_x1[0]*(1e-4/0.1)**2, err_x1[0]], "--", color="gray", lw=0.8)
plt.text(2e-2, 3e-3, r"$\propto h^2$", color="gray")
plt.loglog([1e-12, 1e-8], [3e-16/1e-12, 3e-16/1e-8], "--", color="gray", lw=0.8)
plt.text(2e-12, 1e-3, r"$\propto \varepsilon/h$", color="gray")
plt.xlabel("paso $h$")
plt.ylabel("error absoluto de $F$")
plt.title("Diferencias centradas: truncamiento vs. redondeo")
plt.legend(loc="upper center")
plt.tight_layout()
plt.savefig("figs/fig4_fd.pdf")


#Parte B:
print()
print("PARTE B")

#Caja [0.8, 3.8], coeficientes hasta n = 400 con Simpson (M = 40000 intervalos)
x0 = 0.8
L = 3.0
M = 40000
an, bn = coeficientes(potencial, x0, L, 400, M)
An = np.sqrt(an**2 + bn**2)
print("n, a_n, b_n, A_n:")
for n in range(21):
    print(n, an[n], bn[n], An[n])

#Comprobación de Simpson: a0 exacto con la primitiva G(x) = -x^-11/11 + x^-5/5
G = lambda x: -x**(-11)/11 + x**(-5)/5
a0_exacto = 2/L*(G(x0 + L) - G(x0))
print("a0 Simpson =", an[0], " a0 exacto =", a0_exacto)

#Simpson contra el número de intervalos (referencia: M = 32000)
a_ref, b_ref = coeficientes(potencial, x0, L, 60, 32000)
tabla_simpson = []
for Mi in [1000, 2000, 4000, 8000, 16000]:
    ai, bi = coeficientes(potencial, x0, L, 60, Mi)
    d = np.maximum(np.abs(ai - a_ref), np.abs(bi - b_ref))
    tabla_simpson.append([Mi, L/Mi, np.max(d[:21]), np.max(d)])
    print("M =", Mi, " h =", L/Mi, " error n<=20:", np.max(d[:21]), " error n<=60:", np.max(d))

#Salto de la extensión periódica
V_izq = potencial(x0)
V_der = potencial(x0 + L)
salto = V_izq - V_der
medio = (V_izq + V_der)/2
print("V(0.8) =", V_izq, " V(3.8) =", V_der, " salto J =", salto, " promedio =", medio)

#Si(pi) con Simpson (sen t / t = sinc(t/pi) en numpy) para el sobreimpulso de Gibbs
t = np.linspace(0, np.pi, 2001)
Si_pi = simpson(np.sinc(t/np.pi), np.pi/2000)
gibbs_frac = Si_pi/np.pi - 0.5
gibbs_pred = gibbs_frac*salto
print("Si(pi) =", Si_pi, " 0.0895 =", gibbs_frac, " sobreimpulso previsto =", gibbs_pred)

#Serie parcial con N armónicos
def serie(x, N):
    s = an[0]/2 + 0*x
    for n in range(1, N + 1):
        s = s + an[n]*np.cos(2*np.pi*n*x/L) + bn[n]*np.sin(2*np.pi*n*x/L)
    return s

#Espectro: pendiente del decaimiento de A_n
n20_50 = np.arange(20, 51)
pend_An = pendiente(n20_50, An[20:51])
print("pendiente log-log de A_n (n=20..50) =", pend_An, "  n*A_n(50) =", 50*An[50], "  J/pi =", salto/np.pi)

#Reconstrucción y errores en una malla fina de 30001 puntos de [0.8, 3.8]
xg = np.linspace(x0, x0 + L, 30001)
Vg = potencial(xg)
P_total = simpson(Vg**2, L/30000)/L                     #potencia total (1/L)*integral de V^2
banda = (xg > x0 + 0.2) & (xg < x0 + L - 0.2)           #zona lejos de los bordes
izq = (xg > x0) & (xg < x0 + 0.5)
der = (xg > x0 + L - 0.5) & (xg < x0 + L)
s = an[0]/2 + 0*xg
curvas = {}
err_sup = []
err_banda = []
err_rms = []
err_sobre = []
stats = {}
for N in range(1, 61):
    s = s + an[N]*np.cos(2*np.pi*N*xg/L) + bn[N]*np.sin(2*np.pi*N*xg/L)
    e = s - Vg
    err_sup.append(np.max(np.abs(e)))
    err_banda.append(np.max(np.abs(e[banda])))
    err_rms.append(np.sqrt(np.mean(e**2)))
    err_sobre.append(np.max(e[izq]))
    if N in [5, 10, 20, 50]:
        curvas[N] = s.copy()
        P_N = an[0]**2/4 + 0.5*np.sum(an[1:N+1]**2 + bn[1:N+1]**2)
        j = np.argmax(np.abs(e))
        stats[N] = dict(sup=np.max(np.abs(e)), x_sup=xg[j], banda=err_banda[-1], rms=err_rms[-1],
                        sobre=err_sobre[-1], x_sobre=xg[izq][np.argmax(e[izq])],
                        sub=np.min(e[der]), x_sub=xg[der][np.argmin(e[der])],
                        S_borde=serie(x0, N), potencia=100*P_N/P_total)
print("Potencia total (1/L) int V^2 =", P_total)
for N in [5, 10, 20, 50]:
    st = stats[N]
    print("N =", N, " error máximo =", float(st["sup"]), " lejos del borde =", float(st["banda"]), " RMS =", float(st["rms"]),
          " sobreimpulso =", float(st["sobre"]), " subimpulso =", float(st["sub"]), " V_p(0.8) =", float(st["S_borde"]), " potencia (%) =", float(st["potencia"]))
print("N = 50: error máximo absoluto =", float(stats[50]["sup"]), " en x =", float(stats[50]["x_sup"]))
print("Predicción J/2 + |V'(x0)| L/(2 pi^2 N) =", salto/2 + abs(derivada(x0))*L/(2*np.pi**2*50))

#Gibbs con N grande: sobreimpulso, posición del pico y déficit de la serie en el borde
tabla_gibbs = []
for N in [10, 20, 50, 100, 200, 400]:
    tt = np.linspace(0, 3*L/(2*N + 1), 6001)
    e = serie(x0 + tt, N) - potencial(x0 + tt)
    j = np.argmax(e)
    deficit = medio - serie(x0, N)
    tabla_gibbs.append([N, e[j], tt[j], L/(2*N + 1), deficit, N*deficit])
    print("N =", N, " sobreimpulso =", e[j], " en t =", tt[j], " (L/(2N+1) =", L/(2*N + 1), ") déficit =", deficit, " N*déficit =", N*deficit)
deficit_teorico = abs(derivada(x0))*L/(2*np.pi**2)
print("N*déficit teórico = |V'(x0)| L/(2 pi^2) =", deficit_teorico)

#Pozo armónico en caja simétrica (y = x - xmin, y entre -L/2 y L/2)
ah, bh = coeficientes(lambda y: 0.5*k*y**2, -L/2, L, 60, 20000)
n5_50 = np.arange(5, 51)
pend_arm = pendiente(n5_50, np.abs(ah[5:51]))
const_arm = k*L**2/(2*np.pi**2)
print("Armónico: max|b_n| =", np.max(np.abs(bh)), " pendiente =", pend_arm, " n^2|a_n| =", [float(m**2*abs(ah[m])) for m in (5, 10, 20, 50)], " teórico =", const_arm)


#Gráficas de la parte B
#Figura 5: espectro de Fourier
plt.figure(figsize=(6.6, 3.0))
plt.subplot(1, 2, 1)
n = np.arange(51)
plt.semilogy(n, np.maximum(np.abs(an[:51]), 1e-8), "o", ms=3, label=r"$|a_n|$")
plt.semilogy(n, np.maximum(np.abs(bn[:51]), 1e-8), "s", ms=3, mfc="none", color="C3", label=r"$|b_n|$")
plt.xlabel("$n$")
plt.ylabel("magnitud del coeficiente")
plt.title(r"(a) Espectro de $V_p$ (Lennard-Jones, $L=3$)")
plt.legend(loc="upper right")
plt.subplot(1, 2, 2)
m = np.arange(1, 51)
plt.loglog(m, An[1:51], "o", ms=3, color="C4", label=r"LJ: $A_n=\sqrt{a_n^2+b_n^2}$")
plt.loglog(m, salto/(np.pi*m), "--", color="k", lw=0.9, label=r"$J/(\pi n)$")
plt.loglog(m, np.abs(ah[1:51]), "^", ms=3, color="C2", label=r"armónico: $|a_n|$")
plt.loglog(m, const_arm/m**2, ":", color="k", lw=0.9, label=r"$kL^2/(2\pi^2 n^2)$")
plt.xlabel("$n$")
plt.ylabel("amplitud")
plt.title("(b) Decaimiento espectral")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("figs/fig5_espectro.pdf")

#Figuras 6 y 6b: reconstrucción para N = 5, 10, 20, 50 (escala completa y ampliada)
for ylim, nombre in [((-3, 14), "fig6_reconstruccion.pdf"), ((-0.6, 1.6), "fig6b_reconstruccion_zoom.pdf")]:
    fig, ejes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True, sharey=True)
    for ax, N in zip(ejes.ravel(), [5, 10, 20, 50]):
        ax.plot(xg, Vg, color="k", lw=1.3, label=r"$V(x)$")
        ax.plot(xg, curvas[N], color="C3", lw=0.9, label=rf"$V_p$, $N={N}$")
        ax.set_ylim(ylim)
        ax.set_xlim(0.8, 3.8)
        ax.legend(loc="upper right", framealpha=0.95)
    for ax in ejes[1]:
        ax.set_xlabel("$x$")
    for ax in ejes[:, 0]:
        ax.set_ylabel("$V$")
    plt.tight_layout()
    plt.savefig("figs/" + nombre)

#Figura 7: error de la serie junto a los dos bordes (Gibbs)
plt.figure(figsize=(6.6, 3.0))
colores = {10: "C9", 20: "C0", 50: "C3"}
for N in [10, 20, 50]:
    e = curvas[N] - Vg
    plt.subplot(1, 2, 1)
    plt.plot(xg - x0, e, lw=1.0, color=colores[N], label=rf"$N={N}$")
    plt.subplot(1, 2, 2)
    plt.plot(xg - (x0 + L), e, lw=1.0, color=colores[N], label=rf"$N={N}$")
plt.subplot(1, 2, 1)
plt.axhline(gibbs_pred, color="k", ls="--", lw=0.8)
plt.xlim(0, 0.4)
plt.ylim(-1.5, 1.5)
plt.xlabel(r"$x-0.8$")
plt.ylabel(r"$V_p(x)-V(x)$")
plt.title("(a) Borde izquierdo (sobreimpulso)")
plt.text(0.395, gibbs_pred + 0.06, rf"$+0.0895\,J={gibbs_pred:.3f}$", ha="right", fontsize=7.5)
plt.legend(loc="lower right", framealpha=0.95, ncol=3)
plt.subplot(1, 2, 2)
plt.axhline(-gibbs_pred, color="k", ls="--", lw=0.8)
plt.xlim(-0.4, 0)
plt.ylim(-1.5, 1.5)
plt.xlabel(r"$x-3.8$")
plt.ylabel(r"$V_p(x)-V(x)$")
plt.title("(b) Borde derecho (subimpulso)")
plt.text(-0.395, -gibbs_pred - 0.22, rf"$-0.0895\,J=-{gibbs_pred:.3f}$", ha="left", fontsize=7.5)
plt.legend(loc="upper left", framealpha=0.95, ncol=3)
plt.tight_layout()
plt.savefig("figs/fig7_gibbs.pdf")

#Figura 8: errores contra N (convergencia no uniforme)
plt.figure(figsize=(4.8, 3.1))
N_ = np.arange(1, 61)
plt.loglog(N_, err_sup, "o-", ms=2.5, lw=1, color="C3", label=r"$\max|V_p-V|$ en $[0.8,3.8]$")
plt.loglog(N_, err_sobre, "d-", ms=2.5, lw=1, color="C4", label="sobreimpulso de Gibbs")
plt.loglog(N_, err_banda, "s-", ms=2.5, lw=1, color="C0", label=r"$\max|e|$ con $|x-{\rm borde}|>0.2$")
plt.loglog(N_, err_rms, "^-", ms=2.5, lw=1, color="C2", label="error RMS")
plt.xlabel("$N$ (armónicos)")
plt.ylabel("error absoluto")
plt.title("Convergencia no uniforme")
plt.legend(loc="lower left", fontsize=7)
plt.tight_layout()
plt.savefig("figs/fig8_error_N.pdf")

plt.show()
