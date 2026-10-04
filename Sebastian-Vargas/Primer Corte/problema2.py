import math
import matplotlib.pyplot as plt


def copiar_matriz(A):
    return [fila[:] for fila in A]


def norma(v):
    return math.sqrt(sum(x * x for x in v))


def producto_matriz_vector(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def producto_escalar(A, c):
    return [[c * x for x in fila] for fila in A]


def norma_fuera_diagonal(A):
    n = len(A)
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += A[i][j] ** 2
    return math.sqrt(s)


def jacobi(A, tol=1e-12, max_sweeps=100):
    A = copiar_matriz(A)
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    historia = [norma_fuera_diagonal(A)]

    for barrido in range(max_sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < tol:
                    continue

                phi = 0.5 * math.atan2(2.0 * A[p][q], A[q][q] - A[p][p])
                c = math.cos(phi)
                s = math.sin(phi)

                app = A[p][p]
                aqq = A[q][q]
                apq = A[p][q]

                A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
                A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
                A[p][q] = 0.0
                A[q][p] = 0.0

                for r in range(n):
                    if r != p and r != q:
                        arp = A[r][p]
                        arq = A[r][q]
                        A[r][p] = c * arp - s * arq
                        A[p][r] = A[r][p]
                        A[r][q] = s * arp + c * arq
                        A[q][r] = A[r][q]

                for r in range(n):
                    vrp = V[r][p]
                    vrq = V[r][q]
                    V[r][p] = c * vrp - s * vrq
                    V[r][q] = s * vrp + c * vrq

        historia.append(norma_fuera_diagonal(A))
        if historia[-1] < tol:
            break

    eigenvalores = [A[i][i] for i in range(n)]
    pares = sorted(zip(eigenvalores, range(n)), key=lambda z: z[0])
    valores = []
    vectores = []
    for val, j in pares:
        valores.append(val)
        vectores.append([V[i][j] for i in range(n)])
    return valores, vectores, historia


def producto_vectorial(x, y):
    return sum(a * b for a, b in zip(x, y))


def potencia(A, tolerancia=1e-12, max_iter=500):
    n = len(A)
    v = [1.0 + 0.1 * i for i in range(n)]
    vnorm = norma(v)
    v = [x / vnorm for x in v]
    valor_anterior = 0.0

    for it in range(1, max_iter + 1):
        w = producto_matriz_vector(A, v)
        nw = norma(w)
        if nw == 0:
            raise ValueError("Vector nulo en potencia.")
        v = [x / nw for x in w]
        Av = producto_matriz_vector(A, v)
        lam = producto_vectorial(v, Av)
        if abs(lam - valor_anterior) < tolerancia:
            return lam, v, it
        valor_anterior = lam
    return lam, v, max_iter


def deflacion_potencia(A, cantidad):
    B = copiar_matriz(A)
    resultados = []
    for _ in range(cantidad):
        lam, v, it = potencia(B)
        resultados.append((lam, v, it))
        for i in range(len(B)):
            for j in range(len(B)):
                B[i][j] -= lam * v[i] * v[j]
    resultados.sort(key=lambda z: z[0])
    return resultados


def residual(A, lam, v):
    Av = producto_matriz_vector(A, v)
    return max(abs(Av[i] - lam * v[i]) for i in range(len(v)))


def main():
    k = 200.0
    m = 0.5
    A = [
        [2 * k / m, -k / m, 0.0, 0.0],
        [-k / m, 2 * k / m, -k / m, 0.0],
        [0.0, -k / m, 2 * k / m, -k / m],
        [0.0, 0.0, -k / m, 2 * k / m],
    ]

    valores, vectores, historia = jacobi(A)
    print("Eigenvalores por Jacobi:")
    for i, lam in enumerate(valores, 1):
        print(i, lam, "omega =", math.sqrt(lam), "residuo =", residual(A, lam, vectores[i - 1]))

    print("\nEigenvectores por Jacobi:")
    for v in vectores:
        print([round(x, 8) for x in v])

    print("\nPotencia con deflacion:")
    resultados = deflacion_potencia(A, 4)
    for i, (lam, v, it) in enumerate(resultados, 1):
        print(i, lam, "iteraciones", it)

    exactos = []
    N = 4
    for n in range(1, N + 1):
        lam = (2 * k / m) * (1 - math.cos(n * math.pi / (N + 1)))
        exactos.append(lam)

    for i, lam in enumerate(exactos, 1):
        print("exacto", i, lam, math.sqrt(lam))

    plt.figure()
    x = [0, 1, 2, 3, 4, 5]
    for i, v in enumerate(vectores, 1):
        y = [0.0] + v + [0.0]
        plt.plot(x, y, "o-", label=f"modo {i}")
    plt.xlabel("Posicion")
    plt.ylabel("Amplitud relativa")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p2_modos.png", dpi=180)
    plt.close()

    plt.figure()
    plt.semilogy(range(len(historia)), historia, "o-")
    plt.xlabel("Barrido de Jacobi")
    plt.ylabel("Norma fuera de la diagonal")
    plt.tight_layout()
    plt.savefig("../figuras/p2_jacobi.png", dpi=180)
    plt.close()

    with open("../figuras/p2_resultados.txt", "w", encoding="utf-8") as f:
        for i, (lam, v, it) in enumerate(resultados, 1):
            f.write(f"{i} {lam:.15g} {math.sqrt(lam):.15g} {it}\n")

if __name__ == "__main__":
    main()
