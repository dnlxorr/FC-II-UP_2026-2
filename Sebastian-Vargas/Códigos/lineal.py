def gauss_pivoteo(A, b):
    """Eliminación de Gauss con pivoteo parcial."""
    n = len(A)
    A = [fila[:] for fila in A]
    b = b[:]

    for k in range(n - 1):
        fila_pivote = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[fila_pivote][k]):
                fila_pivote = i

        if fila_pivote != k:
            A[k], A[fila_pivote] = A[fila_pivote], A[k]
            b[k], b[fila_pivote] = b[fila_pivote], b[k]

        if abs(A[k][k]) < 1e-15:
            raise ValueError("La matriz es singular o casi singular.")

        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]

    if abs(A[n - 1][n - 1]) < 1e-15:
        raise ValueError("La matriz es singular o casi singular.")

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]
    return x


def lu_tridiagonal(a, d, c):
    """Factorización LU de una matriz tridiagonal.

    a[0] no se usa; c[-1] no se usa.
    Devuelve l y u, de modo que L tiene 1 en la diagonal.
    """
    n = len(d)
    l = [0.0] * n
    u = [0.0] * n
    u_super = c[:]

    u[0] = d[0]
    for i in range(1, n):
        if abs(u[i - 1]) < 1e-15:
            raise ValueError("Pivote nulo en Thomas.")
        l[i] = a[i] / u[i - 1]
        u[i] = d[i] - l[i] * c[i - 1]
    return l, u, u_super


def thomas_factorizado(l, u, c, b):
    """Resuelve LUx=b usando una factorización ya calculada."""
    n = len(u)
    y = [0.0] * n
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - l[i] * y[i - 1]

    x = [0.0] * n
    x[n - 1] = y[n - 1] / u[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / u[i]
    return x


def inversa_por_columnas(A):
    """Calcula A^{-1} resolviendo A x = e_j para cada columna."""
    n = len(A)
    columnas = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        columnas.append(gauss_pivoteo(A, e))

    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            inv[i][j] = columnas[j][i]
    return inv
