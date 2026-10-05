"""
problema3_kepler.py
===================
Problema 3 – Ecuación de Kepler:   f(E) = E - e sin(E) - M = 0.

Dada la anomalía media M y la excentricidad e, se halla la anomalía excéntrica E
con dos métodos y se comparan.

Método de relajación (punto fijo)
---------------------------------
    E = g(E) = M + e sin(E) ,   E_{k+1} = g(E_k).
Como |g'(E)| = |e cos E| <= e < 1, g es una contracción y el método converge
para cualquier semilla, con razón lineal  |g'(E*)| = e |cos E*|.
Para e pequeña converge rápido; para e → 1 se vuelve lento.

Método de bisección (búsqueda binaria)
--------------------------------------
f(0) = -M < 0 y f(2π) = 2π - M > 0, y f es monótona (f' = 1 - e cos E > 0), por
lo que hay una raíz única en [0, 2π]. El intervalo se reduce a la mitad en cada
paso, de modo que el nº de iteraciones ≈ log2(2π/ε) NO depende de e.

Uso:  python problema3_kepler.py
Salida: resultados por consola y las figuras
        p3_convergencia.png        (error vs iteración, escala semilog)
        p3_iteraciones_vs_e.png    (iteraciones vs excentricidad)
"""
import math
import matplotlib.pyplot as plt

M = 1.5          # anomalía media [rad]
TOL = 1e-6       # tolerancia


def relajacion(e, M, E0=None, tol=TOL, max_iter=100000):
    """Resuelve la ecuación de Kepler por iteración de punto fijo E ← M + e sin E.

    Parámetros
    ----------
    e : float     Excentricidad (0 <= e < 1).
    M : float     Anomalía media [rad].
    E0 : float    Semilla; por defecto E0 = M.
    tol : float   Se detiene cuando |E_{k+1} - E_k| < tol.
    max_iter : int  Tope de iteraciones.

    Retorna
    -------
    (E, iteraciones, historial) donde historial[k] = |E_{k+1} - E_k|.
    """
    E = M if E0 is None else E0
    historial = []
    for it in range(1, max_iter + 1):
        E_nuevo = M + e * math.sin(E)        # E = g(E)
        historial.append(abs(E_nuevo - E))   # cambio entre iteraciones
        if abs(E_nuevo - E) < tol:
            return E_nuevo, it, historial
        E = E_nuevo
    return E, max_iter, historial


def biseccion(e, M, a=0.0, b=2 * math.pi, tol=TOL):
    """Resuelve la ecuación de Kepler por bisección en el intervalo [a, b].

    En cada paso se calcula el punto medio y se conserva la mitad donde f
    cambia de signo. Se detiene cuando la mitad de la longitud del intervalo
    es menor que tol.

    Retorna
    -------
    (E, iteraciones)
    """
    f = lambda E: E - e * math.sin(E) - M
    it = 0
    while (b - a) / 2 > tol:
        it += 1
        medio = (a + b) / 2
        if f(a) * f(medio) <= 0:   # la raíz está en [a, medio]
            b = medio
        else:                      # la raíz está en [medio, b]
            a = medio
    return (a + b) / 2, it


def main():
    # ---------- Comparación para órbita planetaria y cometaria ----------
    for e in (0.1, 0.92):
        E, it, hist = relajacion(e, M)
        E_b, it_b = biseccion(e, M)
        print(f"e = {e}: relajación E = {E:.8f} ({it} it), |g'(E*)| = {abs(e * math.cos(E)):.4f}"
              f" | bisección E = {E_b:.8f} ({it_b} it)")
        plt.semilogy(range(1, len(hist) + 1), hist, "o-", label=f"relajación, e = {e}")
    plt.xlabel("iteración k")
    plt.ylabel("|E_{k+1} - E_k|")
    plt.title("Convergencia del método de relajación")
    plt.legend()
    plt.grid()
    plt.savefig("p3_convergencia.png", dpi=150)
    plt.close()

    # ---------- Iteraciones vs excentricidad, e en [0.1, 0.95] ----------
    excentricidades = [0.1 + 0.01 * i for i in range(86)]
    it_relajacion = [relajacion(e, M)[1] for e in excentricidades]
    it_biseccion = [biseccion(e, M)[1] for e in excentricidades]
    plt.plot(excentricidades, it_relajacion, label="relajación")
    plt.plot(excentricidades, it_biseccion, label="bisección")
    plt.xlabel("excentricidad e")
    plt.ylabel("iteraciones (tol = 1e-6)")
    plt.title("Costo de cada método vs excentricidad")
    plt.legend()
    plt.grid()
    plt.savefig("p3_iteraciones_vs_e.png", dpi=150)


if __name__ == "__main__":
    main()