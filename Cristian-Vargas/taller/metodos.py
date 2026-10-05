# -*- coding: utf-8 -*-
"""
metodos.py  --  Física Computacional II, Universidad de Pamplona
=================================================================
Biblioteca de métodos numéricos implementados DESDE CERO.

Restricción del taller: no se usa numpy.linalg.solve, scipy.optimize ni
ninguna rutina de alto nivel que resuelva sistemas lineales/no lineales o
calcule eigenvalores.  NumPy se usa únicamente como contenedor de arreglos
(y matplotlib para graficar).  Todo el "trabajo numérico" está escrito
con bucles explícitos siguiendo la definición de cada algoritmo.

Contenido
---------
  Unidad 1  : eliminación de Gauss con pivoteo parcial, sustituciones,
              LU de Doolittle, algoritmo de Thomas, inversa por columnas,
              método de Jacobi (rotaciones) y método de la potencia con
              deflación.
  Unidad 2  : relajación (punto fijo) 1D, bisección, Newton 1D,
              secante, relajación 2D (tipo Gauss-Seidel) y Newton n-D.
"""
import math
import numpy as np


# ======================================================================
#                    UNIDAD 1  --  ÁLGEBRA LINEAL
# ======================================================================
def matvec(A, x):
    """Producto matriz-vector  y = A x  (bucles explícitos)."""
    n, m = len(A), len(x)
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += A[i][j] * x[j]
        y[i] = s
    return y


def matmul(A, B):
    """Producto matricial  C = A B  (bucles explícitos)."""
    n, p, m = len(A), len(B), len(B[0])
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i, j] = s
    return C


def norma2(x):
    """Norma euclídea de un vector."""
    return math.sqrt(sum(float(xi) ** 2 for xi in x))


def sustitucion_regresiva(U, c):
    """Resuelve U x = c con U triangular superior (sustitución inversa)."""
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = c[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def sustitucion_progresiva(L, b):
    """Resuelve L y = b con L triangular inferior (sustitución directa)."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s / L[i][i]
    return y


def gauss_pivoteo(A, b, pivoteo=True, guardar_etapas=False):
    """
    Eliminación de Gauss con pivoteo parcial + sustitución inversa.

    Parámetros
    ----------
    A, b            : sistema A x = b
    pivoteo         : si False, no se permutan filas (solo para demostración)
    guardar_etapas  : si True devuelve la matriz aumentada tras cada columna

    Devuelve  x, info   con info = {'intercambios', 'U', 'c', 'etapas'}
    """
    n = len(b)
    M = np.zeros((n, n + 1))                 # matriz aumentada [A | b]
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i][j]
        M[i, n] = b[i]
    intercambios, etapas = [], []
    for k in range(n - 1):
        if pivoteo:                          # ---- pivoteo parcial
            p = k
            for i in range(k + 1, n):
                if abs(M[i, k]) > abs(M[p, k]):
                    p = i
            if p != k:
                M[[k, p]] = M[[p, k]]
                intercambios.append((k + 1, p + 1))
        if M[k, k] == 0.0:
            raise ZeroDivisionError("Matriz singular (pivote nulo).")
        for i in range(k + 1, n):            # ---- eliminación hacia adelante
            m = M[i, k] / M[k, k]
            for j in range(k, n + 1):
                M[i, j] -= m * M[k, j]
            M[i, k] = 0.0
        if guardar_etapas:
            etapas.append(M.copy())
    if M[n - 1, n - 1] == 0.0:
        raise ZeroDivisionError("Matriz singular (pivote nulo).")
    U, c = M[:, :n], M[:, n]
    x = sustitucion_regresiva(U, c)
    return x, {"intercambios": intercambios, "U": U.copy(), "c": c.copy(),
               "etapas": etapas}


def lu_doolittle(A):
    """
    Descomposición LU de Doolittle (L con diagonal unitaria), sin pivoteo:
        u_{ij} = a_{ij} - sum_{k<i} l_{ik} u_{kj}          (j >= i)
        l_{ij} = (a_{ij} - sum_{k<j} l_{ik} u_{kj}) / u_jj (i >  j)
    """
    n = len(A)
    L, U = np.eye(n), np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for k in range(i):
                s += L[i, k] * U[k, j]
            U[i, j] = A[i][j] - s
        for j in range(i + 1, n):
            s = 0.0
            for k in range(i):
                s += L[j, k] * U[k, i]
            L[j, i] = (A[j][i] - s) / U[i, i]
    return L, U


def thomas_factor(a, b, c):
    """
    Factorización LU de una matriz tridiagonal (a: subdiagonal, b: diagonal,
    c: superdiagonal).  L es bidiagonal inferior (diag. unitaria, subdiag. l)
    y U bidiagonal superior (diag. u, superdiag. c):
        u_1 = b_1 ,   l_i = a_i / u_{i-1} ,   u_i = b_i - l_i c_{i-1}
    Coste O(N).  a y c tienen longitud N-1;  b tiene longitud N.
    """
    n = len(b)
    u, l = np.zeros(n), np.zeros(n - 1)
    u[0] = b[0]
    for i in range(1, n):
        l[i - 1] = a[i - 1] / u[i - 1]
        u[i] = b[i] - l[i - 1] * c[i - 1]
    return l, u


def thomas_resolver(l, u, c, d):
    """Con L y U ya calculadas resuelve A x = d:  L y = d  y  U x = y. O(N)."""
    n = len(u)
    y = np.zeros(n)
    y[0] = d[0]
    for i in range(1, n):                    # sustitución progresiva
        y[i] = d[i] - l[i - 1] * y[i - 1]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / u[n - 1]
    for i in range(n - 2, -1, -1):           # sustitución regresiva
        x[i] = (y[i] - c[i] * x[i + 1]) / u[i]
    return x


def inversa_por_columnas(l, u, c):
    """A^{-1}: se resuelven N sistemas  A x_j = e_j  reutilizando L y U."""
    n = len(u)
    Ainv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Ainv[:, j] = thomas_resolver(l, u, c, e)
    return Ainv


# ----------------------------------------------------------------------
#                 Eigenvalores: rotaciones de Jacobi
# ----------------------------------------------------------------------
def _norma_fuera_diag(A):
    n, s = len(A), 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += A[i, j] ** 2
    return math.sqrt(s)


def jacobi_eigen(A, tol=1e-13, max_barridos=60):
    """
    Método cíclico de Jacobi para matrices simétricas.
    En cada rotación (p,q) se anula a_pq con una rotación de ángulo theta:
        cot(2 theta) = (a_qq - a_pp)/(2 a_pq) ,  t = tan(theta)
    Devuelve (eigenvalores, matriz V con eigenvectores en columnas,
              historial de ||A_offdiag|| tras cada barrido).
    """
    n = len(A)
    A = np.array(A, dtype=float)
    V = np.eye(n)
    hist = [_norma_fuera_diag(A)]
    for _ in range(max_barridos):
        if hist[-1] < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if A[p, q] == 0.0:
                    continue
                theta = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                sgn = 1.0 if theta >= 0 else -1.0
                t = sgn / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                app, aqq, apq = A[p, p], A[q, q], A[p, q]
                A[p, p] = app - t * apq
                A[q, q] = aqq + t * apq
                A[p, q] = A[q, p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = A[r, p], A[r, q]
                        A[r, p] = A[p, r] = c * arp - s * arq
                        A[r, q] = A[q, r] = c * arq + s * arp
                for r in range(n):
                    vrp, vrq = V[r, p], V[r, q]
                    V[r, p] = c * vrp - s * vrq
                    V[r, q] = s * vrp + c * vrq
        hist.append(_norma_fuera_diag(A))
    lam = np.array([A[i, i] for i in range(n)])
    orden = sorted(range(n), key=lambda i: lam[i])
    return lam[orden], V[:, orden], hist


# ----------------------------------------------------------------------
#                 Eigenvalores: potencia + deflación
# ----------------------------------------------------------------------
def potencia(A, x0, tol=1e-13, max_iter=100000):
    """
    Método de la potencia: x_{k+1} = A x_k / ||A x_k||.
    Estimación de eigenvalor con el cociente de Rayleigh  lam = x^T A x.
    Devuelve lam, v, iteraciones, historial de lam.
    """
    x = np.array(x0, dtype=float)
    x = x / norma2(x)
    lam_old, hist = 0.0, []
    for k in range(1, max_iter + 1):
        y = matvec(A, x)
        lam = sum(x[i] * y[i] for i in range(len(x)))
        hist.append(lam)
        nrm = norma2(y)
        x = y / nrm
        if abs(lam - lam_old) < tol * max(1.0, abs(lam)) and k > 2:
            break
        lam_old = lam
    return lam, x, k, hist


def potencia_con_deflacion(A, x0, tol=1e-13):
    """
    Todos los pares propios de una matriz simétrica mediante potencia +
    deflación de Hotelling:  A_{j+1} = A_j - lam_j v_j v_j^T.
    Devuelve listas (lam, v, iteraciones, historiales) en orden de
    eigenvalor decreciente.
    """
    n = len(A)
    B = np.array(A, dtype=float)
    lams, vecs, its, hists = [], [], [], []
    for j in range(n):
        x = np.array(x0, dtype=float) + 0.1 * j * np.arange(1, n + 1) ** 0.5
        lam, v, k, h = potencia(B, x, tol)
        lams.append(lam); vecs.append(v); its.append(k); hists.append(h)
        for a in range(n):
            for b in range(n):
                B[a, b] -= lam * v[a] * v[b]
    return lams, vecs, its, hists


# ======================================================================
#                 UNIDAD 2  --  ECUACIONES NO LINEALES
# ======================================================================
def relajacion_1d(g, x0, tol=1e-6, max_iter=100000):
    """Iteración de punto fijo x_{k+1} = g(x_k).  Devuelve lista de iterados."""
    xs = [x0]
    for _ in range(max_iter):
        xs.append(g(xs[-1]))
        if abs(xs[-1] - xs[-2]) < tol:
            break
    return xs


def biseccion(f, a, b, tol=1e-6, max_iter=200):
    """Búsqueda binaria.  Devuelve lista de puntos medios (aproximaciones)."""
    fa = f(a)
    if fa * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")
    cs = []
    for _ in range(max_iter):
        c = (a + b) / 2
        cs.append(c)
        fc = f(c)
        if fc == 0 or (b - a) / 2 < tol:
            break
        if fa * fc < 0:
            b = c
        else:
            a, fa = c, fc
    return cs


def newton_1d(f, df, x0, tol=1e-12, max_iter=100):
    """Newton-Raphson: x_{k+1} = x_k - f(x_k)/f'(x_k)."""
    xs = [x0]
    for _ in range(max_iter):
        xs.append(xs[-1] - f(xs[-1]) / df(xs[-1]))
        if abs(xs[-1] - xs[-2]) < tol:
            break
    return xs


def secante(f, x0, x1, tol=1e-12, max_iter=100):
    """Secante: x_{k+1} = x_k - f_k (x_k - x_{k-1})/(f_k - f_{k-1})."""
    xs = [x0, x1]
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            break
        x2 = xs[-1] - f1 * (xs[-1] - xs[-2]) / (f1 - f0)
        xs.append(x2)
        f0, f1 = f1, f(x2)
        if abs(xs[-1] - xs[-2]) < tol:
            break
    return xs


def relajacion_2d(g1, g2, x0, y0, tol=1e-12, max_iter=100000):
    """
    Relajación de dos variables tipo Gauss-Seidel:
        x_{k+1} = g1(x_k, y_k) ,   y_{k+1} = g2(x_{k+1}, y_k)
    """
    P = [(x0, y0)]
    for _ in range(max_iter):
        x, y = P[-1]
        xn = g1(x, y)
        yn = g2(xn, y)
        P.append((xn, yn))
        if math.hypot(xn - x, yn - y) < tol:
            break
        if not (math.isfinite(xn) and math.isfinite(yn)) or abs(xn) > 1e8:
            break
    return P


def newton_nd(F, J, x0, tol=1e-13, max_iter=50):
    """
    Newton multivariable: en cada paso se resuelve  J(x_k) dx = -F(x_k)
    con la eliminación de Gauss con pivoteo parcial de este mismo módulo
    y se actualiza  x_{k+1} = x_k + dx.
    """
    X = [np.array(x0, dtype=float)]
    for _ in range(max_iter):
        x = X[-1]
        dx, _ = gauss_pivoteo(J(x), [-fi for fi in F(x)])
        X.append(x + dx)
        if norma2(dx) < tol:
            break
    return X
