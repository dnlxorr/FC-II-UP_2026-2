import math
import time
import matplotlib.pyplot as plt
from lineal import gauss_pivoteo, lu_tridiagonal, thomas_factorizado, inversa_por_columnas

L = 1.0
k = 45.0
TA = 100.0
TB = 20.0
N = 6

def matriz_barra(N):
    A = []
    for i in range(N):
        fila = [0.0] * N
        fila[i] = 2.0
        if i > 0:
            fila[i - 1] = -1.0
        if i < N - 1:
            fila[i + 1] = -1.0
        A.append(fila)
    return A

def terminos_derecha(qfun):
    dx = L / (N + 1)
    b = []
    for i in range(1, N + 1):
        x = i * dx
        valor = dx * dx * qfun(x) / k
        if i == 1:
            valor += TA
        if i == N:
            valor += TB
        b.append(valor)
    return b

def exacta_q1(x):
    return TA + (TB - TA) * x / L + 5000.0 * x * (L - x) / (2 * k)

def exacta_q2(x):
    c = 10000.0 / (k * L)
    C1 = (TB - TA) / L + c * L * L / 6.0
    return TA + C1 * x - c * x ** 3 / 6.0

def exacta_q3(x):
    return TA + (TB - TA) * x / L + (8000.0 / k) * (L * L / math.pi ** 2) * math.sin(math.pi * x / L)

def main():
    A = matriz_barra(N)
    dx = L / (N + 1)
    print("dx =", dx)
    print("A =")
    for fila in A:
        print(fila)

    # Actividad 2
    b1 = terminos_derecha(lambda x: 5000.0)
    T_gauss = gauss_pivoteo(A, b1)
    print("\nGauss para q=5000:")
    for i, Ti in enumerate(T_gauss, 1):
        x = i * dx
        print(i, x, Ti, abs(Ti - exacta_q1(x)))

    # Actividad 3: una sola factorizacion
    a = [0.0] * N
    d = [2.0] * N
    c = [0.0] * N
    for i in range(1, N):
        a[i] = -1.0
        c[i - 1] = -1.0

    l, u, c_super = lu_tridiagonal(a, d, c)
    fuentes = [
        ("q1=5000", lambda x: 5000.0, exacta_q1),
        ("q2=10000 x/L", lambda x: 10000.0 * x / L, exacta_q2),
        ("q3=8000 sin", lambda x: 8000.0 * math.sin(math.pi * x / L), exacta_q3),
    ]

    soluciones = []
    errores = []
    for nombre, qfun, exacta in fuentes:
        b = terminos_derecha(qfun)
        T = thomas_factorizado(l, u, c_super, b)
        soluciones.append((nombre, T, exacta))
        err = max(abs(T[i] - exacta((i + 1) * dx)) for i in range(N))
        errores.append(err)
        print("\n", nombre)
        for i, Ti in enumerate(T, 1):
            print(i, Ti)
        print("error maximo =", err)

    # Actividad 4: inversa
    Ainv = inversa_por_columnas(A)
    print("\nA^(-1):")
    for fila in Ainv:
        print([round(x, 6) for x in fila])

    # Comparacion de costos para varias n
    tamanos = [50, 100, 200]
    t_gauss = []
    t_thomas = []
    for n in tamanos:
        AA = matriz_barra(n)
        bb = [1.0] * n
        a = [0.0] * n
        d = [2.0] * n
        c = [0.0] * n
        for i in range(1, n):
            a[i] = -1.0
            c[i - 1] = -1.0
        ini = time.perf_counter()
        gauss_pivoteo(AA, bb)
        t_gauss.append(time.perf_counter() - ini)
        ini = time.perf_counter()
        ll, uu, cc = lu_tridiagonal(a, d, c)
        thomas_factorizado(ll, uu, cc, bb)
        t_thomas.append(time.perf_counter() - ini)

    # Graficas
    xs = [(i + 1) * dx for i in range(N)]
    plt.figure()
    for nombre, T, exacta in soluciones:
        plt.plot(xs, T, "o", label=nombre)
        xx = [j / 100 for j in range(101)]
        yy = [exacta(x) for x in xx]
        plt.plot(xx, yy, "-")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p1_perfiles.png", dpi=180)
    plt.close()

    plt.figure()
    for j in range(N):
        plt.plot(range(1, N + 1), [Ainv[i][j] for i in range(N)], "o-", label=f"columna {j + 1}")
    plt.xlabel("Nodo i")
    plt.ylabel(r"$(A^{-1})_{ij}$")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("../figuras/p1_inversa.png", dpi=180)
    plt.close()

    plt.figure()
    plt.loglog(tamanos, t_gauss, "o-", label="Gauss")
    plt.loglog(tamanos, t_thomas, "s-", label="Thomas")
    plt.xlabel("n")
    plt.ylabel("tiempo [s]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figuras/p1_costos.png", dpi=180)
    plt.close()

    print("\nTiempos")
    for n, tg, tt in zip(tamanos, t_gauss, t_thomas):
        print(n, tg, tt)

if __name__ == "__main__":
    main()
