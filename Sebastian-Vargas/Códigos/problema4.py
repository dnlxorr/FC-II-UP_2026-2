import math
import matplotlib.pyplot as plt
from lineal import gauss_pivoteo


def van_der_waals(v, P, a, b, R, T):
    return (P + a / v ** 2) * (v - b) - R * T


def derivada_vdw(v, P, a, b):
    return P - a / v ** 2 + 2 * a * b / v ** 3


def newton_1d(f, df, x0, tol=1e-14, max_iter=50):
    x = x0
    lista = [x]
    for n in range(1, max_iter + 1):
        xnuevo = x - f(x) / df(x)
        lista.append(xnuevo)
        if abs(xnuevo - x) < tol:
            return xnuevo, n, lista
        x = xnuevo
    raise RuntimeError("Newton 1D no convergio.")


def secante(f, x0, x1, tol=1e-14, max_iter=50):
    lista = [x0, x1]
    f0 = f(x0)
    f1 = f(x1)
    for n in range(2, max_iter + 2):
        den = f1 - f0
        if abs(den) < 1e-30:
            raise ValueError("Denominador demasiado pequeno en secante.")
        x2 = x1 - f1 * (x1 - x0) / den
        lista.append(x2)
        if abs(x2 - x1) < tol:
            return x2, n - 1, lista
        x0, x1 = x1, x2
        f0, f1 = f1, f(x2)
    raise RuntimeError("Secante no convergio.")


def F(x, y):
    return [x ** 3 - 3 * x * y ** 2 - 1.0,
            3 * x ** 2 * y - y ** 3]


def jacobiano(x, y):
    return [[3 * x ** 2 - 3 * y ** 2, -6 * x * y],
            [6 * x * y, 3 * x ** 2 - 3 * y ** 2]]


def relajacion_2d(x0, y0, w, tol=1e-12, max_iter=500):
    x, y = x0, y0
    lista = [(x, y)]
    for n in range(1, max_iter + 1):
        try:
            fx, fy = F(x, y)
            xn = x - w * fx
            yn = y - w * F(xn, y)[1]
        except OverflowError:
            return float("nan"), float("nan"), n, lista
        lista.append((xn, yn))
        if not math.isfinite(xn) or not math.isfinite(yn) or max(abs(xn), abs(yn)) > 1e100:
            return float("nan"), float("nan"), n, lista
        if max(abs(xn - x), abs(yn - y)) < tol:
            return xn, yn, n, lista
        x, y = xn, yn
    return x, y, max_iter, lista


def newton_2d(x0, y0, tol=1e-14, max_iter=30):
    x, y = x0, y0
    lista = [(x, y)]
    for n in range(1, max_iter + 1):
        Fv = F(x, y)
        J = jacobiano(x, y)
        paso = gauss_pivoteo(J, [-Fv[0], -Fv[1]])
        xn = x + paso[0]
        yn = y + paso[1]
        lista.append((xn, yn))
        if max(abs(xn - x), abs(yn - y)) < tol:
            return xn, yn, n, lista
        x, y = xn, yn
    return x, y, max_iter, lista


def main():
    # Parte A
    P = 2.0e6
    a = 0.3643
    b = 4.267e-5
    R = 8.314
    T = 300.0
    videal = R * T / P
    f = lambda v: van_der_waals(v, P, a, b, R, T)
    df = lambda v: derivada_vdw(v, P, a, b)

    vn, nn, lista_n = newton_1d(f, df, videal)
    vs, ns, lista_s = secante(f, videal, 0.9 * videal)
    Z = P * vn / (R * T)

    print("Van der Waals")
    print("v ideal =", videal)
    print("Newton =", vn, "iteraciones", nn)
    print("Secante =", vs, "iteraciones", ns)
    print("Z =", Z)

    # Parte B: relajacion
    print("\nRelajacion 2D")
    datos_rel = [
        (0.2, 0.8, 0.3),
        (0.2, 1.3, -0.4),
        (0.2, -0.4, 0.8),
        (-0.2, -0.4, 0.8),
        (-0.2, -0.6, -0.7),
        (-0.2, 0.8, 0.3),
    ]
    for w, x0, y0 in datos_rel:
        x, y, n, _ = relajacion_2d(x0, y0, w)
        print(w, x0, y0, n, x, y)

    # Newton 2D
    print("\nNewton 2D")
    semillas = [(0.8, 0.3), (-0.4, 0.8), (-0.6, -0.7), (2.0, 2.0)]
    trayectorias = []
    for x0, y0 in semillas:
        x, y, n, lista = newton_2d(x0, y0)
        trayectorias.append(lista)
        print((x0, y0), n, x, y)

    # Graficas de convergencia
    # Parte A: Van der Waals
    vstar = vn
    err_n = [abs(x - vstar) for x in lista_n]
    err_s = [abs(x - vstar) for x in lista_s]
    plt.figure()
    plt.semilogy(range(len(err_n)), err_n, "o-", label="Newton")
    plt.semilogy(range(len(err_s)), err_s, "s-", label="Secante")
    plt.xlabel("Iteracion")
    plt.ylabel("Error absoluto")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p4a_convergencia.png", dpi=180)
    plt.close()

    # Parte B: sistema 2D
    _, _, _, rel = relajacion_2d(0.8, 0.3, 0.2)
    _, _, _, new = newton_2d(0.8, 0.3)
    erel = [math.hypot(x - 1.0, y) for x, y in rel]
    enew = [math.hypot(x - 1.0, y) for x, y in new]
    plt.figure()
    plt.semilogy(range(len(erel)), erel, "o-", label="Relajacion")
    plt.semilogy(range(len(enew)), enew, "s-", label="Newton")
    plt.xlabel("Iteracion")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p4b_convergencia.png", dpi=180)
    plt.close()

    # Cuencas de Newton (malla sencilla)
    ngrid = 101
    xmin, xmax = -1.5, 1.5
    ymin, ymax = -1.5, 1.5
    cuadricula = []
    with open("../figuras/p4_cuencas.txt", "w", encoding="utf-8") as out:
        for j in range(ngrid):
            y0 = ymin + (ymax - ymin) * j / (ngrid - 1)
            fila = []
            for i in range(ngrid):
                x0 = xmin + (xmax - xmin) * i / (ngrid - 1)
                try:
                    x, y, n, _ = newton_2d(x0, y0, max_iter=30)
                    if math.hypot(x - 1, y) < 1e-5:
                        r = 0
                    elif math.hypot(x + 0.5, y - math.sqrt(3) / 2) < 1e-5:
                        r = 1
                    elif math.hypot(x + 0.5, y + math.sqrt(3) / 2) < 1e-5:
                        r = 2
                    else:
                        r = -1
                except (ValueError, OverflowError):
                    r = -1
                fila.append(r)
            cuadricula.append(fila)
            out.write(" ".join(str(r) for r in fila) + "\n")

    plt.figure()
    plt.imshow(cuadricula, origin="lower", extent=[xmin, xmax, ymin, ymax], interpolation="nearest", aspect="equal")
    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.tight_layout()
    plt.savefig("../figuras/p4_cuencas.png", dpi=180)
    plt.close()

if __name__ == "__main__":
    main()
