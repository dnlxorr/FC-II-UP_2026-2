"""
algebra_lineal.py
=================
Rutinas de álgebra lineal implementadas desde cero (sin numpy.linalg ni scipy),
tal como exige el Taller de Física Computacional II.

Contenido
---------
gauss_pivoteo(A, b)             Eliminación de Gauss con pivoteo parcial + sustitución inversa.
thomas_factorizar(a, d, c)      Descomposición LU de una matriz tridiagonal.
thomas_resolver(l, u, c, b)     Resolución de L U x = b con la factorización ya calculada.

Convención de almacenamiento
----------------------------
Las matrices densas se representan como listas de listas (A[i][j]).
Una matriz tridiagonal de tamaño n se guarda en tres listas:
    a : subdiagonal   (n-1 elementos)  -> A[i+1][i]
    d : diagonal      (n   elementos)  -> A[i][i]
    c : superdiagonal (n-1 elementos)  -> A[i][i+1]
"""


def gauss_pivoteo(A, b):
    """Resuelve el sistema lineal A x = b por eliminación de Gauss con pivoteo parcial.

    Algoritmo
    ---------
    1. Se forma la matriz aumentada [A | b].
    2. Para cada columna k = 0..n-2:
         a) Pivoteo parcial: se busca la fila p >= k con mayor |A[p][k]| y se
            intercambia con la fila k (evita dividir entre pivotes pequeños).
         b) Eliminación: para cada fila i > k se calcula el multiplicador
            m = A[i][k]/A[k][k] y se resta m * fila_k a la fila_i.
    3. El sistema queda triangular superior y se resuelve con sustitución inversa.

    Parámetros
    ----------
    A : list[list[float]]
        Matriz cuadrada n x n (no se modifica; se trabaja sobre una copia).
    b : list[float]
        Vector del lado derecho de longitud n.

    Retorna
    -------
    list[float]
        Vector solución x.

    Lanza
    -----
    ValueError
        Si algún pivote es prácticamente cero (matriz singular).

    Costo: O(n^3 / 3) operaciones.
    """
    n = len(b)
    # Matriz aumentada [A | b]; se copian las filas para no alterar la entrada.
    M = [fila[:] + [b[i]] for i, fila in enumerate(A)]

    # ---------- Eliminación hacia adelante ----------
    for k in range(n - 1):
        # (a) Pivoteo parcial: fila con el mayor valor absoluto en la columna k
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-14:
            raise ValueError("Matriz singular o casi singular")
        M[k], M[p] = M[p], M[k]  # intercambio de filas

        # (b) Anular los elementos debajo del pivote
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]            # multiplicador
            for j in range(k, n + 1):        # incluye la columna aumentada
                M[i][j] -= m * M[k][j]

    # ---------- Sustitución inversa ----------
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - suma) / M[i][i]
    return x


def thomas_factorizar(a, d, c):
    """Descomposición LU de una matriz tridiagonal (parte de factorización del algoritmo de Thomas).

    Se busca A = L U con
        L : bidiagonal inferior con unos en la diagonal y l_i debajo,
        U : bidiagonal superior con u_i en la diagonal y c_i arriba (la
            superdiagonal de A se conserva).
    Igualando coeficientes:
        u_0 = d_0
        l_i = a_i / u_i            (i = 1..n-1, con a_i de A[i][i-1])
        u_i = d_i - l_i * c_{i-1}

    Como la factorización solo depende de A, se calcula UNA vez y se reutiliza
    para cualquier vector b (costo O(n) en lugar de O(n^3)).

    Parámetros
    ----------
    a, d, c : list[float]
        Sub-diagonal (n-1), diagonal (n) y super-diagonal (n-1) de A.

    Retorna
    -------
    (l, u) : tuple[list[float], list[float]]
        l : multiplicadores de L (n-1 elementos).
        u : diagonal de U (n elementos).
    """
    n = len(d)
    u = [0.0] * n
    l = [0.0] * (n - 1)
    u[0] = d[0]
    for i in range(1, n):
        l[i - 1] = a[i - 1] / u[i - 1]
        u[i] = d[i] - l[i - 1] * c[i - 1]
    return l, u


def thomas_resolver(l, u, c, b):
    """Resuelve L U x = b usando la factorización de `thomas_factorizar`.

    Dos barridos:
        1) Sustitución directa   L y = b :  y_0 = b_0 ;  y_i = b_i - l_i * y_{i-1}
        2) Sustitución inversa   U x = y :  x_{n-1} = y_{n-1}/u_{n-1} ;
                                            x_i = (y_i - c_i * x_{i+1}) / u_i

    Parámetros
    ----------
    l, u : list[float]
        Resultado de `thomas_factorizar`.
    c : list[float]
        Superdiagonal de A (n-1 elementos).
    b : list[float]
        Lado derecho (n elementos).

    Retorna
    -------
    list[float]
        Vector solución x. Costo O(n).
    """
    n = len(u)

    # Sustitución directa: L y = b
    y = [0.0] * n
    y[0] = b[0]
    for i in range(1, n):
        y[i] = b[i] - l[i - 1] * y[i - 1]

    # Sustitución inversa: U x = y
    x = [0.0] * n
    x[-1] = y[-1] / u[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - c[i] * x[i + 1]) / u[i]
    return x