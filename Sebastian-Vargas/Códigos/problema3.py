import math
import matplotlib.pyplot as plt


def f(E, e, M):
    return E - e * math.sin(E) - M


def relajacion(e, M, tol=1e-6, max_iter=1000):
    E = M
    lista = [E]
    for n in range(1, max_iter + 1):
        nuevo = M + e * math.sin(E)
        lista.append(nuevo)
        if abs(nuevo - E) < tol:
            return nuevo, n, lista
        E = nuevo
    raise RuntimeError("No convergio.")


def biseccion(e, M, tol=1e-6, max_iter=1000):
    a = 0.0
    b = 2.0 * math.pi
    fa = f(a, e, M)
    fb = f(b, e, M)
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo.")
    lista = []
    for n in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c, e, M)
        lista.append((n, a, b, c, fc))
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) / 2.0 < tol:
            c = (a + b) / 2.0
            return c, n, lista
    raise RuntimeError("No convergio.")


def main():
    M = 1.5
    for e in [0.1, 0.92]:
        E, n, lista = relajacion(e, M)
        print("relajacion", e, E, n)
        for i, Ei in enumerate(lista):
            print(i, Ei)

    print("\nbiseccion e=0.92")
    E, n, lista = biseccion(0.92, M)
    for fila in lista[:8]:
        print(fila)
    print("... ultima", lista[-1])
    print("iteraciones", n, "raiz", E)

    # Barrido solicitado
    es = [0.10 + 0.01 * i for i in range(86)]
    it_rel = []
    it_bis = []
    for e in es:
        _, nr, _ = relajacion(e, M)
        _, nb, _ = biseccion(e, M)
        it_rel.append(nr)
        it_bis.append(nb)

    # M adicional para mostrar que la rapidez depende de M
    es_extra = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.92, 0.95]
    it_extra = []
    for e in es_extra:
        _, n, _ = relajacion(e, 0.01)
        it_extra.append(n)

    plt.figure()
    for e, estilo in [(0.1, "o-"), (0.92, "s-"), (0.92, "^-" )]:
        if e == 0.1:
            _, _, seq = relajacion(e, M)
            raiz, _, _ = biseccion(e, M, tol=1e-14)
            errores = [abs(x - raiz) for x in seq]
            plt.semilogy(range(len(errores)), errores, estilo, label="relajacion e=0.1")
            break
    _, _, seq = relajacion(0.92, M)
    raiz, _, _ = biseccion(0.92, M, tol=1e-14)
    errores = [abs(x - raiz) for x in seq]
    plt.semilogy(range(len(errores)), errores, "s-", label="relajacion e=0.92")
    raiz, _, lista = biseccion(0.92, M)
    errores_b = [abs(fila[3] - raiz) for fila in lista]
    plt.semilogy(range(1, len(errores_b) + 1), errores_b, "^-", label="biseccion e=0.92")
    plt.xlabel("Iteracion")
    plt.ylabel("Error absoluto")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p3_convergencia.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(es, it_rel, label="relajacion, M=1.5")
    plt.plot(es, it_bis, label="biseccion, M=1.5")
    plt.plot(es_extra, it_extra, "o", label="relajacion, M=0.01")
    plt.xlabel("Excentricidad e")
    plt.ylabel("Iteraciones")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p3_iteraciones.png", dpi=180)
    plt.close()

    print("\nTabla seleccionada")
    for e, ne in zip(es_extra, it_extra):
        j = int(round((e - 0.10) / 0.01))
        print(e, it_rel[j], it_bis[j], ne)

if __name__ == "__main__":
    main()
