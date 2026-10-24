"""
============================================================================
 ACTIVIDAD FC2 — Equilibrio y Puntos Prometidos en un Sistema de Cargas
 Electricas No Lineal
============================================================================

Contexto fisico
----------------
Dos cargas FIJAS:
    Q1 = +4 uC en (0, 0)
    Q2 = +1 uC en (2, 0)          [distancias en metros]

Una carga MOVIL q = +1 uC, sometida ademas a un potencial externo
NO LINEAL:
    Vext(x, y) = A (x^4 + y^4),      A = 0.5 V/m^4

La condicion de equilibrio es que la fuerza neta sobre q sea cero:

    F_net(x,y) = F_cargas(x,y) - q * grad(Vext)(x,y) = 0

Esto se traduce en dos ecuaciones no lineales acopladas f1(x,y)=0,
f2(x,y)=0

Este script resuelve:
    PARTE A -> metodo de relajacion (punto fijo amortiguado) en 2D
    PARTE B -> biseccion 1D sobre el eje x para U(x) = U0

y genera todas las graficas pedidas en el enunciado.
============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. CARPETA DE SALIDA (funciona en cualquier computador: crea una
#    carpeta "outputs" junto a este script, si no existe)
# ----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. CONSTANTES FISICAS DEL PROBLEMA
# ----------------------------------------------------------------------
k  = 8.99e9      # Constante de Coulomb [N m^2 / C^2]
Q1 = 4e-6        # Carga fija 1 [C]  (en el origen)
Q2 = 1e-6        # Carga fija 2 [C]  (en x = 2 m)
q  = 1e-6        # Carga movil  [C]
A  = 0.5         # Coeficiente del potencial externo [V/m^4]

X2 = 2.0         # posicion en x de Q2


# ----------------------------------------------------------------------
# 2. SISTEMA FISICO: f1(x,y)=0 , f2(x,y)=0
# ----------------------------------------------------------------------
#
# Fuerza de Coulomb sobre q debida a Q1 (en el origen):
#     F1 = k*Q1*q * (x, y) / (x^2+y^2)^(3/2)
#
# Fuerza de Coulomb sobre q debida a Q2 (en (2,0)):
#     F2 = k*Q2*q * (x-2, y) / ((x-2)^2+y^2)^(3/2)
#
# Fuerza del potencial externo:
#     F_ext = -q*grad(Vext) = -q*(4A x^3, 4A y^3)
#
# La carga q se puede factorizar y CANCELAR en las dos ecuaciones
# (no afecta la posicion de equilibrio, solo su signo importaria).
#
# Igualando F_cargas = q*grad(Vext) componente a componente:
#
#   f1(x,y) = k*Q1*x/r1^3 + k*Q2*(x-2)/r2^3 - 4*A*x^3 = 0
#   f2(x,y) = k*Q1*y/r1^3 + k*Q2*y/r2^3     - 4*A*y^3 = 0
#
# con r1 = sqrt(x^2+y^2), r2 = sqrt((x-2)^2+y^2)

def _r1r2(x, y):
    r1 = (x**2 + y**2)**1.5
    r2 = ((x - X2)**2 + y**2)**1.5
    return r1, r2


def f1(x, y):
    r1, r2 = _r1r2(x, y)
    return k*Q1*x/r1 + k*Q2*(x - X2)/r2 - 4*A*x**3


def f2(x, y):
    r1, r2 = _r1r2(x, y)
    return k*Q1*y/r1 + k*Q2*y/r2 - 4*A*y**3


def residual_norm(x, y):
    """||F(x,y)|| , usado para decidir si una iteracion 'mejora'."""
    return np.hypot(f1(x, y), f2(x, y))


# ----------------------------------------------------------------------
# 3. FORMA DE PUNTO FIJO  x = g1(x,y),  y = g2(x,y)
# ----------------------------------------------------------------------
#
# Se despeja el termino cubico (el mas "aislado" algebraicamente) y se
# extrae raiz cubica REAL (por eso se usa signo * |.|^(1/3), no ** (1/3)
# directo, que en Python falla para negativos):
#
#   x^3 = [ k*Q1*x/r1^3 + k*Q2*(x-2)/r2^3 ] / (4A)   ->  x = g1(x,y)
#   y^3 = [ k*Q1*y/r1^3 + k*Q2*y/r2^3     ] / (4A)   ->  y = g2(x,y)

def _cbrt(v):
    return np.sign(v) * np.abs(v)**(1.0/3.0)


def g1(x, y):
    r1, r2 = _r1r2(x, y)
    rhs = (k*Q1*x/r1 + k*Q2*(x - X2)/r2) / (4*A)
    return _cbrt(rhs)


def g2(x, y):
    r1, r2 = _r1r2(x, y)
    rhs = (k*Q1*y/r1 + k*Q2*y/r2) / (4*A)
    return _cbrt(rhs)


def jacobiano_g(x, y, h=1e-6):
    """Jacobiano numerico de (g1,g2) en (x,y) -> analisis de estabilidad."""
    dg1dx = (g1(x+h, y) - g1(x-h, y)) / (2*h)
    dg1dy = (g1(x, y+h) - g1(x, y-h)) / (2*h)
    dg2dx = (g2(x+h, y) - g2(x-h, y)) / (2*h)
    dg2dy = (g2(x, y+h) - g2(x, y-h)) / (2*h)
    return np.array([[dg1dx, dg1dy], [dg2dx, dg2dy]])


# ----------------------------------------------------------------------
# 4. PARTE A: METODO DE RELAJACION (PUNTO FIJO AMORTIGUADO) EN 2D
# ----------------------------------------------------------------------
#
# La relajacion NO aplica g(x,y) directamente (eso es punto fijo puro,
# omega=1) sino que MEZCLA el valor viejo con el nuevo mediante un
# factor de relajacion omega in (0,1]:
#
#   x_(n+1) = (1-omega)*x_n + omega*g1(x_n,y_n)
#   y_(n+1) = (1-omega)*y_n + omega*g2(x_n,y_n)
#
# omega=1        -> punto fijo puro (puede divergir si |g'|>1)
# omega pequeno  -> avanza mas lento pero es mas estable
#
# Aqui se implementa una relajacion ADAPTATIVA: en cada paso se intenta
# con el omega actual; si el residuo ||F|| EMPEORA, omega se reduce a la
# mitad (line-search) hasta que mejore o se vuelva insignificante.
# Si el paso tuvo exito, omega se deja crecer un poco para el siguiente
# paso. Esta es la generalizacion естандar del metodo de relajacion
# simple cuando el sistema es "rigido" (derivadas muy grandes).

def relajacion_2d(x0, y0, tol=1e-6, omega0=0.5, max_iter=20000):
    x, y = x0, y0
    omega = omega0
    trayectoria = [(x, y)]
    res = residual_norm(x, y)
    omegas_usados = []

    for it in range(max_iter):
        gx, gy = g1(x, y), g2(x, y)
        om = omega
        for _ in range(80):                       # line-search interno
            xn = (1-om)*x + om*gx
            yn = (1-om)*y + om*gy
            rn = residual_norm(xn, yn)
            if rn < res or om < 1e-15:
                break
            om *= 0.5
        paso = max(abs(xn-x), abs(yn-y))
        x, y, res = xn, yn, rn
        omega = min(om*1.2, 0.9)                  # recupera omega
        trayectoria.append((x, y))
        omegas_usados.append(om)
        if paso < tol:
            return {
                "convergio": True, "x": x, "y": y, "iters": it+1,
                "residuo": res, "trayectoria": np.array(trayectoria),
                "omegas": omegas_usados,
            }
    return {
        "convergio": False, "x": x, "y": y, "iters": max_iter,
        "residuo": res, "trayectoria": np.array(trayectoria),
        "omegas": omegas_usados,
    }


def relajacion_2d_omega_fijo(x0, y0, omega, tol=1e-6, max_iter=20000):
    """Version 'de libro de texto' con omega CONSTANTE (sin adaptar),
    usada solo para comparar/ilustrar estabilidad frente a distintos omega."""
    x, y = x0, y0
    trayectoria = [(x, y)]
    for it in range(max_iter):
        xn = (1-omega)*x + omega*g1(x, y)
        yn = (1-omega)*y + omega*g2(x, y)
        if not (np.isfinite(xn) and np.isfinite(yn)) or abs(xn) > 1e8 or abs(yn) > 1e8:
            return {"convergio": False, "diverge": True, "iters": it,
                    "trayectoria": np.array(trayectoria)}
        paso = max(abs(xn-x), abs(yn-y))
        x, y = xn, yn
        trayectoria.append((x, y))
        if paso < tol:
            return {"convergio": True, "x": x, "y": y, "iters": it+1,
                    "trayectoria": np.array(trayectoria)}
    return {"convergio": False, "diverge": False, "iters": max_iter,
            "trayectoria": np.array(trayectoria)}


# ----------------------------------------------------------------------
# 5. PARTE B: ENERGIA POTENCIAL TOTAL SOBRE EL EJE X  Y  BISECCION
# ----------------------------------------------------------------------
#
# Sobre el eje (y=0), f2(x,0)=0 se cumple automaticamente (todo termino
# tiene un factor y), asi que el problema se reduce a 1 variable.
#
# La energia potencial TOTAL de q (Coulomb + externa) es la funcion
# cuyo gradiente cambiado de signo da la fuerza neta usada en la Parte A:
#
#   U(x) = k*Q1*q/x + k*Q2*q/(2-x) + q*A*x^4          0 < x < 2

def U(x):
    return k*Q1*q/x + k*Q2*q/(2 - x) + q*A*x**4


def g_bisec(x, U0):
    return U(x) - U0


def biseccion(a, b, U0, tol_x=1e-5, max_iter=200):
    """Biseccion clasica para resolver U(x)=U0 en [a,b].
    Requiere g(a) y g(b) de signos opuestos (condicion de Bolzano)."""
    ga, gb = g_bisec(a, U0), g_bisec(b, U0)
    if ga == 0:
        return {"raiz": a, "iters": 0, "valido": True}
    if gb == 0:
        return {"raiz": b, "iters": 0, "valido": True}
    if ga*gb > 0:
        return {"valido": False, "ga": ga, "gb": gb}

    historial = []
    for it in range(1, max_iter+1):
        m = 0.5*(a+b)
        gm = g_bisec(m, U0)
        historial.append((it, a, b, m, gm))
        if gm == 0 or (b-a)/2 < tol_x:
            return {"valido": True, "raiz": m, "iters": it,
                     "historial": historial, "ancho_final": b-a}
        if ga*gm < 0:
            b, gb = m, gm
        else:
            a, ga = m, gm
    return {"valido": True, "raiz": 0.5*(a+b), "iters": max_iter,
             "historial": historial, "ancho_final": b-a}


def escanear_cambios_de_signo(a, b, U0, n=4000):
    """Barrido grueso para localizar sub-intervalos [a_i,b_i] donde
    U(x)-U0 cambia de signo (paso previo, obligatorio, antes de bisectar)."""
    xs = np.linspace(a, b, n)
    gs = U(xs) - U0
    brackets = []
    for i in range(len(xs)-1):
        if np.sign(gs[i]) != np.sign(gs[i+1]) and np.isfinite(gs[i]) and np.isfinite(gs[i+1]):
            brackets.append((xs[i], xs[i+1]))
    return brackets


# ============================================================================
# 6. EJECUCION PRINCIPAL
# ============================================================================
if __name__ == "__main__":

    print("="*78)
    print("PARTE A: Metodo de Relajacion 2D")
    print("="*78)

    x0, y0 = 1.0, 0.5
    tol = 1e-6

    # (a) intento de punto fijo PURO (omega=1) para evidenciar inestabilidad
    print(f"\n[Diagnostico] g1(x0,y0) = {g1(x0,y0):.4f}  (x0 era {x0})")
    print(f"[Diagnostico] g2(x0,y0) = {g2(x0,y0):.4f}  (y0 era {y0})")
    print("-> el salto es enorme: el punto fijo PURO es inestable aqui.\n")

    # (b) relajacion con varios omega FIJOS (tabla comparativa de estabilidad)
    print("Tabla de estabilidad frente a omega (relajacion de omega FIJO):")
    for om in [1.0, 0.1, 0.01]:
        r = relajacion_2d_omega_fijo(x0, y0, om, tol=tol)
        if r.get("diverge"):
            print(f"  omega={om:<6} -> DIVERGE en la iteracion {r['iters']}")
        elif r["convergio"]:
            print(f"  omega={om:<6} -> converge en {r['iters']:4d} iters a "
                  f"({r['x']:.6f}, {r['y']:.6f})")
        else:
            print(f"  omega={om:<6} -> no converge en el limite de iteraciones")

    # (c) relajacion ADAPTATIVA (la que se usa como resultado oficial)
    resA = relajacion_2d(x0, y0, tol=tol)
    trayA = resA["trayectoria"]
    print(f"\n[Relajacion adaptativa] convergio={resA['convergio']}  "
          f"iters={resA['iters']}")
    print(f"  Punto de equilibrio hallado: "
          f"(x*, y*) = ({resA['x']:.6f}, {resA['y']:.6f})")
    print(f"  Residuo final ||F|| = {resA['residuo']:.3e}")

    # (d) analisis de estabilidad (jacobiano/autovalores) en el punto hallado
    #     y en el punto "cercano al eje" que tambien es raiz del sistema
    raiz_lejana = (resA['x'], resA['y'])

    # raiz PRECISA sobre el eje: f1(x,0)=0 (biseccion simple 1D auxiliar,
    # solo para tener el punto exacto donde comparar estabilidad)
    def _f1_eje(x):
        return f1(x, 0.0)
    ax_, bx_ = 0.5, 1.9
    for _ in range(80):
        mx = 0.5*(ax_+bx_)
        if _f1_eje(ax_)*_f1_eje(mx) <= 0:
            bx_ = mx
        else:
            ax_ = mx
    x_raiz_eje = 0.5*(ax_+bx_)
    raiz_eje = (x_raiz_eje, 0.0)

    print(f"\nRaiz precisa sobre el eje (f1(x,0)=0): x* = {x_raiz_eje:.8f}")
    print("Analisis de estabilidad del mapa de iteracion g=(g1,g2):")
    for nombre, (rx, ry) in [("raiz sobre el eje", raiz_eje),
                              ("raiz hallada por relajacion", raiz_lejana)]:
        J = jacobiano_g(rx, ry, h=1e-6)
        eig = np.linalg.eigvals(J)
        print(f"  {nombre} ({rx:.5f},{ry:.5f}): autovalores de J_g = {eig}")
        print(f"    |autovalores| = {np.abs(eig)}  "
              f"-> {'ESTABLE (todos <1)' if np.all(np.abs(eig)<1) else 'INESTABLE (algun |autovalor|>1)'}")
    print("  Nota: dg2/dy en la raiz del eje es una SINGULARIDAD matematica")
    print("  real (no solo un numero grande): como el numerador de la")
    print("  ecuacion en y es proporcional a y, cerca de y=0 se tiene")
    print("  g2(x,y) ~ C(x)^(1/3) * y^(1/3), cuya derivada diverge a")
    print("  infinito cuando y->0. Ningun omega>0 puede estabilizar eso.")

    print("\n" + "="*78)
    print("PARTE B: Biseccion 1D sobre el eje x")
    print("="*78)

    U0_pedido = 15e-3   # 15 mJ, valor literal del enunciado
    a, b = 0.1, 1.9
    Umin = U(np.linspace(a, b, 4000)).min()
    print(f"\nU0 pedido en el enunciado = {U0_pedido*1000:.1f} mJ")
    print(f"Minimo de U(x) en [{a},{b}] = {Umin*1000:.4f} mJ "
          f"(en x = {np.linspace(a,b,4000)[np.argmin(U(np.linspace(a,b,4000)))]:.4f})")

    brackets_pedido = escanear_cambios_de_signo(a, b, U0_pedido)
    if not brackets_pedido:
        print("-> NO existe cambio de signo: U0=15 mJ esta por DEBAJO del "
              "minimo de U(x) en ese intervalo.")
        print("   La biseccion NO puede aplicarse tal como esta planteada "
              "el enunciado (condicion de Bolzano no se cumple).")
    else:
        print("Brackets encontrados:", brackets_pedido)

    # Se demuestra el metodo con un U0 SI alcanzable dentro de (0.1,1.9).
    # U(x) tiene un unico minimo en x=1.3333 (visto en el barrido de arriba),
    # asi que se parte el intervalo pedido [0.1,1.9] en dos sub-intervalos
    # AMPLIOS alrededor de ese minimo, cada uno con un solo cruce de signo,
    # y se bisecta cada uno por separado (asi la biseccion arranca del
    # intervalo grande, no de uno ya angosto).
    U0_demo = 45e-3
    x_min = 1.3333
    brackets = [(a, x_min), (x_min, b)]
    print(f"\nSe demuestra el algoritmo con U0 = {U0_demo*1000:.0f} mJ "
          f"(si tiene solucion en el dominio). Brackets amplios: {brackets}")

    raices_biseccion = []
    for (ai, bi) in brackets:
        r = biseccion(ai, bi, U0_demo, tol_x=1e-5)
        raices_biseccion.append(r)
        print(f"  bracket [{ai:.3f},{bi:.3f}] -> raiz x* = {r['raiz']:.6f}  "
              f"en {r['iters']} iteraciones (ancho final {r['ancho_final']:.2e})")

    N_teorico = int(np.ceil(np.log2((b-a)/1e-5)))
    print(f"\nIteraciones TEORICAS de biseccion para tol=1e-5 en [{a},{b}]: "
          f"N = ceil(log2((b-a)/tol)) = {N_teorico}")
    print("(no depende de la funcion U(x), solo del ancho del intervalo "
          "y la tolerancia -> contraste clave con la Parte A)")

    # ------------------------------------------------------------------
    # 7. GRAFICAS
    # ------------------------------------------------------------------

    # --- Grafica 1: trayectoria de convergencia de la relajacion (Parte A)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(trayA[:, 0], trayA[:, 1], '-o', ms=3, lw=1,
            color='tab:blue', label='Trayectoria de relajacion')
    ax.plot(trayA[0, 0], trayA[0, 1], 's', ms=10, color='green',
            label=f'Inicio ({x0:.1f},{y0:.1f})')
    ax.plot(trayA[-1, 0], trayA[-1, 1], '*', ms=16, color='red',
            label=f"Equilibrio ({resA['x']:.3f},{resA['y']:.3f})")
    ax.plot(0, 0, 'X', ms=10, color='black')
    ax.annotate('Q1 (+4 uC)', (0, 0), textcoords="offset points",
                xytext=(8, 8))
    ax.plot(2, 0, 'X', ms=10, color='black')
    ax.annotate('Q2 (+1 uC)', (2, 0), textcoords="offset points",
                xytext=(8, 8))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Parte A - Trayectoria de convergencia (relajacion 2D)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'parteA_trayectoria.png'), dpi=150)
    plt.close(fig)

    # --- Grafica 2: mapa de potencial V(x,y) con los equilibrios
    #     V(x,y) = potencial total por unidad de carga = k Q1/r1 + k Q2/r2 + Vext
    xg = np.linspace(-4, 9, 500)
    yg = np.linspace(-4, 9, 500)
    Xg, Yg = np.meshgrid(xg, yg)
    r1g = np.sqrt(Xg**2 + Yg**2)
    r2g = np.sqrt((Xg-2)**2 + Yg**2)
    with np.errstate(divide='ignore'):
        Vg = k*Q1/np.maximum(r1g, 1e-3) + k*Q2/np.maximum(r2g, 1e-3) + A*(Xg**4+Yg**4)
    Vg_clip = np.clip(Vg, 0, np.percentile(Vg, 92))   # recorte visual (evita saturar por las singularidades)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cf = ax.contourf(Xg, Yg, Vg_clip, levels=40, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('V(x,y) = k Q1/r1 + k Q2/r2 + A(x^4+y^4)  [V]  (recortado)')
    ax.plot(0, 0, 'X', ms=10, color='white', mec='black')
    ax.plot(2, 0, 'X', ms=10, color='white', mec='black')
    ax.plot(*raiz_eje, '*', ms=16, color='cyan', mec='black',
            label=f'Raiz sobre el eje ({raiz_eje[0]:.3f},0)  [saddle]')
    ax.plot(*raiz_lejana, '*', ms=16, color='red', mec='black',
            label=f'Equilibrio de la relajacion ({raiz_lejana[0]:.3f},{raiz_lejana[1]:.3f})')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Mapa de potencial V(x,y) y puntos de equilibrio')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'mapa_potencial_equilibrios.png'), dpi=150)
    plt.close(fig)

    # --- Grafica 3: U(x) sobre el eje, con biseccion (Parte B)
    xx = np.linspace(0.1, 1.9, 800)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xx, U(xx)*1000, color='tab:purple', label='U(x) [mJ]')
    ax.axhline(U0_pedido*1000, color='gray', ls='--',
               label=f'U0 pedido = {U0_pedido*1000:.0f} mJ (fuera de rango)')
    ax.axhline(U0_demo*1000, color='tab:orange', ls='--',
               label=f'U0 demostracion = {U0_demo*1000:.0f} mJ')
    for r in raices_biseccion:
        ax.plot(r['raiz'], U(r['raiz'])*1000, 'o', ms=8, color='tab:red')
        ax.annotate(f"x*={r['raiz']:.4f}", (r['raiz'], U(r['raiz'])*1000),
                    textcoords="offset points", xytext=(0, 10), ha='center')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('U(x) [mJ]')
    ax.set_title('Parte B - Energia potencial total sobre el eje y biseccion')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'parteB_biseccion.png'), dpi=150)
    plt.close(fig)

    # --- Grafica 4: comparacion de convergencia (error vs iteracion)
    fig, ax = plt.subplots(figsize=(7, 5))
    err_relax = np.hypot(np.diff(trayA[:, 0]), np.diff(trayA[:, 1]))
    ax.semilogy(np.arange(1, len(err_relax)+1), err_relax,
                'o-', ms=3, color='tab:blue',
                label=f'Relajacion 2D (Parte A) - {resA["iters"]} iters')
    if raices_biseccion:
        hist = raices_biseccion[0]['historial']
        anchos = [ (h[2]-h[1]) for h in hist ]
        ax.semilogy(np.arange(1, len(anchos)+1), anchos,
                    's-', ms=4, color='tab:orange',
                    label=f'Biseccion 1D (Parte B) - {len(anchos)} iters')
    ax.set_xlabel('Iteracion')
    ax.set_ylabel('Tamano de paso / ancho de intervalo (escala log)')
    ax.set_title('Comparacion de velocidad de convergencia')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'comparacion_convergencia.png'), dpi=150)
    plt.close(fig)

    print(f"\nGraficas guardadas en {OUTPUT_DIR}")
    print("  - parteA_trayectoria.png")
    print("  - mapa_potencial_equilibrios.png")
    print("  - parteB_biseccion.png")
    print("  - comparacion_convergencia.png")