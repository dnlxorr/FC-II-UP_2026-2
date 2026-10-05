r"""
=============================================================================
 TEMA: Eigenvalores y eigenvectores (metodo de potencias, potencia inversa
       y algoritmo QR)
 CURSO: Fisica Computacional II
=============================================================================

EJEMPLO FISICO: modos normales de N masas acopladas por resortes
-----------------------------------------------------------------
N masas iguales m unidas por N+1 resortes iguales de constante k, con los
dos extremos fijos a paredes. Con x_j el desplazamiento de la masa j, la
segunda ley de Newton da

    m x_j'' = -k (x_j - x_{j-1}) - k (x_j - x_{j+1})        (x_0 = x_{N+1} = 0)

que en forma matricial es   x'' = -A x ,   con   A = (k/m) tridiag(-1, 2, -1).

POR QUE APARECE UN PROBLEMA DE EIGENVALORES (justificacion)
-----------------------------------------------------------------
Se buscan soluciones donde TODAS las masas oscilan con la misma frecuencia:
x(t) = u cos(w t). Al sustituir, x'' = -w^2 u cos(w t), y la ecuacion de
movimiento se reduce a

        A u = w^2 u .

Es decir: w^2 es un eigenvalor de A y u su eigenvector (el "modo normal").
Como A es simetrica y definida positiva, sus eigenvalores son reales y
positivos (frecuencias reales) y sus eigenvectores son ortogonales. Esa
ortogonalidad es la que permite escribir cualquier movimiento como
superposicion de modos: x(t) = sum_n c_n u_n cos(w_n t), con c = V^T x(0).

SOLUCION ANALITICA (para medir el error real del metodo)
-----------------------------------------------------------------
Para esta matriz tridiagonal particular existe solucion cerrada:

    u_j^(n) = sin( j n pi / (N+1) ),
    w_n = 2 sqrt(k/m) sin( n pi / (2 (N+1)) ),     n = 1,...,N

(se verifica sustituyendo u^(n) en (A u)_j y usando la identidad
 sin(a-b) + sin(a+b) = 2 sin(a) cos(b)).

METODOS IMPLEMENTADOS
-----------------------------------------------------------------
(1) Potencias: x_{k+1} = A x_k / ||A x_k||  converge al eigenvector del
    eigenvalor de MAYOR modulo, con razon |l_2/l_1| por iteracion.
(2) Potencia inversa: aplicar potencias a A^{-1} (resolviendo A y = x en
    vez de invertir) converge al eigenvalor de MENOR modulo, con razon
    |l_1/l_2|. Aqui eso es el MODO FUNDAMENTAL (el mas importante fisicamente).
(3) Algoritmo QR: A_{k+1} = R_k Q_k, con A_k = Q_k R_k. Como
    A_{k+1} = Q_k^T A_k Q_k es una transformacion de semejanza, conserva los
    eigenvalores; las matrices A_k tienden a una matriz diagonal D, y el
    producto acumulado V = Q_1 Q_2 ... Q_k tiene como columnas los
    eigenvectores (porque A_k = V^T A V -> D implica A V = V D).
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# -----------------------------------------------------------------------
# 1) Sistema fisico
# -----------------------------------------------------------------------
N = 6                  # numero de masas
m = 0.5                # kg
k = 200.0              # N/m
w0sq = k / m           # k/m

A = w0sq * (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))

n_idx = np.arange(1, N + 1)
w_analitica = 2 * np.sqrt(w0sq) * np.sin(n_idx * np.pi / (2 * (N + 1)))
lam_analitica = w_analitica ** 2          # eigenvalores exactos, ascendentes

print("Matriz A = (k/m) tridiag(-1,2,-1):")
print(A)
print("\nEigenvalores exactos (rad/s)^2:", lam_analitica)


# -----------------------------------------------------------------------
# 2) Metodo de potencias
# -----------------------------------------------------------------------
def metodo_potencias(A, x0, n_iter=400):
    """Devuelve (lambda_final, x_final, historial de cocientes de Rayleigh,
    historial de iterados). Cociente de Rayleigh: lam = x^T A x / x^T x;
    para A simetrica converge al doble de rapido que el eigenvector."""
    x = x0 / np.linalg.norm(x0)
    lam_hist, x_hist = [], []
    for _ in range(n_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
        lam_hist.append(x @ A @ x)
        x_hist.append(x.copy())
    return lam_hist[-1], x, np.array(lam_hist), np.array(x_hist)


# -----------------------------------------------------------------------
# 3) Potencia inversa (usa LU con pivoteo, de los scripts anteriores)
# -----------------------------------------------------------------------
def lu_pivoteo(A):
    n = A.shape[0]
    U, L, P = A.astype(float).copy(), np.eye(n), np.eye(n)
    for kk in range(n - 1):
        fm = kk + np.argmax(np.abs(U[kk:, kk]))
        if fm != kk:
            U[[kk, fm]] = U[[fm, kk]]
            P[[kk, fm]] = P[[fm, kk]]
            L[[kk, fm], :kk] = L[[fm, kk], :kk]
        for i in range(kk + 1, n):
            mm = U[i, kk] / U[kk, kk]
            L[i, kk] = mm
            U[i, kk:] -= mm * U[kk, kk:]
    return P, L, U


def resolver_lu(P, L, U, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (P @ b)[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def potencia_inversa(A, x0, n_iter=40):
    """Potencias sobre A^{-1}. Se factoriza A UNA vez (LU) y en cada
    iteracion solo se hacen dos sustituciones (ver script 4)."""
    P, L, U = lu_pivoteo(A)
    x = x0 / np.linalg.norm(x0)
    lam_hist, x_hist = [], []
    for _ in range(n_iter):
        y = resolver_lu(P, L, U, x)       # y = A^{-1} x  sin calcular A^{-1}
        x = y / np.linalg.norm(y)
        lam_hist.append(x @ A @ x)
        x_hist.append(x.copy())
    return lam_hist[-1], x, np.array(lam_hist), np.array(x_hist)


# -----------------------------------------------------------------------
# 4) Algoritmo QR (Gram-Schmidt modificado para la factorizacion)
# -----------------------------------------------------------------------
def qr_gram_schmidt(A):
    """A = Q R con Q ortogonal y R triangular superior (Gram-Schmidt
    modificado: restar proyecciones sobre la marcha es mas estable que la
    version clasica)."""
    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    V = A.astype(float).copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for l in range(j + 1, n):
            R[j, l] = Q[:, j] @ V[:, l]
            V[:, l] -= R[j, l] * Q[:, j]
    return Q, R


def algoritmo_qr(A, tol=1e-12, max_iter=5000):
    """Devuelve eigenvalores (diagonal final), eigenvectores (columnas de V)
    y el historial de la norma de la parte triangular inferior de A_k."""
    n = A.shape[0]
    Ak = A.astype(float).copy()
    V = np.eye(n)
    hist = []
    for it in range(max_iter):
        Q, R = qr_gram_schmidt(Ak)
        Ak = R @ Q
        V = V @ Q
        off = np.linalg.norm(np.tril(Ak, -1))
        hist.append(off)
        if off < tol:
            break
    return np.diag(Ak).copy(), V, np.array(hist), Ak


# =========================================================================
# 5) Ejecucion y verificacion
# =========================================================================
lam_qr, V_qr, hist_qr, A_final = algoritmo_qr(A)
orden = np.argsort(lam_qr)
lam_qr, V_qr = lam_qr[orden], V_qr[:, orden]

print(f"\nAlgoritmo QR: convergio en {len(hist_qr)} iteraciones")
print("Eigenvalores QR       :", lam_qr)
print("Error relativo vs exactos:", np.abs(lam_qr - lam_analitica) / lam_analitica)

res_AV = np.linalg.norm(A @ V_qr - V_qr * lam_qr)
ort = np.linalg.norm(V_qr.T @ V_qr - np.eye(N))
print(f"\n||A V - V D||_2   = {res_AV:.3e}   (cada columna es eigenvector)")
print(f"||V^T V - I||_2   = {ort:.3e}   (modos ortonormales)")

# frecuencias en Hz y rad/s
w_qr = np.sqrt(lam_qr)
print("\nFrecuencias de los modos normales:")
for i in range(N):
    print(f"  modo {i+1}: w = {w_qr[i]:8.4f} rad/s (exacta {w_analitica[i]:8.4f})"
          f"  f = {w_qr[i]/(2*np.pi):7.3f} Hz")

# --- eigenvectores numericos vs analiticos (normalizados y con signo fijo) ---
j_idx = np.arange(1, N + 1)
modos_exactos = np.array([np.sin(j_idx * n_ * np.pi / (N + 1)) for n_ in n_idx]).T
modos_exactos /= np.linalg.norm(modos_exactos, axis=0)
err_modos = []
for i in range(N):
    v = V_qr[:, i]
    v = v * np.sign(v @ modos_exactos[:, i])        # el signo es arbitrario
    err_modos.append(np.linalg.norm(v - modos_exactos[:, i]))
    V_qr[:, i] = v
print("\nError ||u_num - u_exacto||_2 por modo:", np.array(err_modos))

# --- metodo de potencias (modo de mayor frecuencia) ---
rng = np.random.default_rng(1)
x0 = rng.random(N) + 0.1
lam_p, x_p, lam_hist_p, x_hist_p = metodo_potencias(A, x0, n_iter=250)
lam1, lam2 = lam_analitica[-1], lam_analitica[-2]
print(f"\nPotencias: lambda_max = {lam_p:.8f}  (exacto {lam1:.8f}), "
      f"razon teorica |l2/l1| = {lam2/lam1:.4f}")

# --- potencia inversa (modo fundamental) ---
lam_i, x_i, lam_hist_i, x_hist_i = potencia_inversa(A, x0, n_iter=30)
print(f"Potencia inversa: lambda_min = {lam_i:.8f}  (exacto {lam_analitica[0]:.8f}), "
      f"razon teorica |l1/l2| = {lam_analitica[0]/lam_analitica[1]:.4f}")

# =========================================================================
# 6) Evolucion temporal por superposicion de modos
# =========================================================================
x_ini = np.zeros(N)
x_ini[0] = 0.01                       # se desplaza 1 cm la primera masa
c = V_qr.T @ x_ini                    # coeficientes modales (V es ortogonal)
t = np.linspace(0, 0.6, 600)
X_t = (V_qr * c) @ np.cos(np.outer(w_qr, t))      # N x len(t)

# verificacion: la superposicion debe reproducir la condicion inicial
print(f"\n||x(0) superposicion - x_ini|| = {np.linalg.norm(X_t[:,0]-x_ini):.3e}")

# =========================================================================
# 7) Graficas
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# (a) convergencia del algoritmo QR
axes[0, 0].semilogy(range(1, len(hist_qr) + 1), hist_qr, color="#2E86AB")
axes[0, 0].set_xlabel("Iteracion k")
axes[0, 0].set_ylabel(r"$\|$parte triangular inferior de $A_k\|$")
axes[0, 0].set_title("Convergencia del algoritmo QR a forma diagonal")
axes[0, 0].grid(alpha=0.3, which="both")

# (b) error relativo de los eigenvalores
axes[0, 1].semilogy(n_idx, np.abs(lam_qr - lam_analitica) / lam_analitica + 1e-18,
                    "o-", color="#C1121F")
axes[0, 1].set_xlabel("Modo n")
axes[0, 1].set_ylabel("Error relativo del eigenvalor")
axes[0, 1].set_title("QR vs solucion analitica")
axes[0, 1].grid(alpha=0.3, which="both")

# (c) formas modales
x_pos = np.arange(0, N + 2)
for i in range(3):
    perfil = np.concatenate(([0], V_qr[:, i], [0]))
    axes[0, 2].plot(x_pos, perfil, "o-", label=f"modo {i+1} (QR)")
    exacto = np.concatenate(([0], modos_exactos[:, i], [0]))
    axes[0, 2].plot(x_pos, exacto, "k--", lw=0.8)
axes[0, 2].plot([], [], "k--", lw=0.8, label="exacto")
axes[0, 2].set_xlabel("Posicion de la masa j (0 y N+1: paredes)")
axes[0, 2].set_ylabel("Amplitud normalizada")
axes[0, 2].set_title("Tres primeros modos normales")
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(alpha=0.3)

# (d) potencias: error del eigenvalor y del eigenvector
iters_p = np.arange(1, len(lam_hist_p) + 1)
err_lam_p = np.abs(lam_hist_p - lam1) / lam1
ang_p = np.array([1 - abs(xx @ modos_exactos[:, -1]) for xx in x_hist_p])
axes[1, 0].semilogy(iters_p, err_lam_p + 1e-18, label="error de eigenvalor")
axes[1, 0].semilogy(iters_p, ang_p + 1e-18, label=r"$1-|\cos\theta|$ del eigenvector")
r = lam2 / lam1
axes[1, 0].semilogy(iters_p, r ** (2 * iters_p), "k:", label=r"$(\lambda_2/\lambda_1)^{2k}$")
axes[1, 0].set_ylim(1e-17, 2)
axes[1, 0].set_xlabel("Iteracion k")
axes[1, 0].set_title("Metodo de potencias (modo de mayor frecuencia)")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3, which="both")

# (e) potencia inversa
iters_i = np.arange(1, len(lam_hist_i) + 1)
err_lam_i = np.abs(lam_hist_i - lam_analitica[0]) / lam_analitica[0]
ri = lam_analitica[0] / lam_analitica[1]
axes[1, 1].semilogy(iters_i, err_lam_i + 1e-18, "o-", label="error de eigenvalor")
axes[1, 1].semilogy(iters_i, ri ** (2 * iters_i), "k:", label=r"$(\lambda_1/\lambda_2)^{2k}$")
axes[1, 1].set_ylim(1e-17, 2)
axes[1, 1].set_xlabel("Iteracion k")
axes[1, 1].set_title("Potencia inversa (modo fundamental)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, which="both")

# (f) movimiento de las masas por superposicion de modos
for j in [0, 2, 5]:
    axes[1, 2].plot(t, X_t[j] * 100, label=f"masa {j+1}")
axes[1, 2].set_xlabel("t (s)")
axes[1, 2].set_ylabel("Desplazamiento (cm)")
axes[1, 2].set_title(r"$x(t)=\sum_n c_n u_n \cos(\omega_n t)$")
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("7_eigenvalores_eigenvectores.png", dpi=150)
print("\nGrafica guardada en 7_eigenvalores_eigenvectores.png")