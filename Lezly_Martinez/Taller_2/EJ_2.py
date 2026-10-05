"""
problema2_modos.py
==================
Problema 2 – Modos normales de vibración de N = 4 masas acopladas.

Física
------
N masas iguales m unidas entre sí y a dos paredes fijas por resortes de
constante k. Para la masa j:  m x_j'' = k (x_{j-1} - 2 x_j + x_{j+1}).
Con x_j = v_j e^{iωt} resulta el problema de eigenvalores
        K v = ω² M v   ⇒   (M⁻¹K) v = λ v ,   λ = ω² ,
con M = m I y K = k · tridiag(-1, 2, -1). La matriz dinámica A = M⁻¹K es real
y simétrica, por lo que el método de Jacobi converge siempre.

Solución analítica (para comparar)
----------------------------------
    ω_n = 2 √(k/m) sin( nπ / (2(N+1)) ) ,   v_j^(n) = sin( j n π / (N+1) ).

Método numérico: rotaciones de Jacobi
-------------------------------------
Se anula iterativamente el mayor elemento fuera de la diagonal con una rotación
plana  A' = Jᵀ A J ; al converger la diagonal contiene los eigenvalores y las
columnas de V = J₁J₂··· los eigenvectores.

Uso:  python problema2_modos.py
Salida: eigenvalores/frecuencias por consola y la figura p2_modos.png
"""
import math
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parámetros del sistema
# ------------------------------------------------------------------
N = 4          # número de masas
m = 0.5        # masa de cada partícula [kg]
k = 200.0      # constante elástica [N/m]


def matriz_dinamica(n, m, k):
    """Construye A = M⁻¹K = (k/m)·tridiag(-1, 2, -1) de tamaño n x n."""
    return [[(2 * k / m if i == j else -k / m if abs(i - j) == 1 else 0.0)
             for j in range(n)] for i in range(n)]


def jacobi(A, tol=1e-12, max_iter=100):
    """Eigenvalores y eigenvectores de una matriz simétrica por rotaciones de Jacobi.

    En cada iteración:
      1) Se localiza el elemento fuera de diagonal de mayor módulo, A[p][q].
      2) Se calcula el ángulo que lo anula:
             θ = (A_qq - A_pp) / (2 A_pq)
             t = sgn(θ) / (|θ| + √(θ²+1))     (raíz de menor módulo de t²+2θt-1=0)
             c = 1/√(t²+1) ,  s = t c
      3) Se actualiza A ← Jᵀ A J (solo cambian las filas/columnas p y q) y se
         acumula V ← V J.
    El proceso termina cuando max|A_pq| < tol.

    Parámetros
    ----------
    A : list[list[float]]   Matriz simétrica (no se modifica).
    tol : float             Tolerancia sobre los elementos fuera de la diagonal.
    max_iter : int          Máximo de rotaciones.

    Retorna
    -------
    (eigenvalores, V) : eigenvalores es una lista; V[r][i] es la componente r
    del eigenvector i (los eigenvectores son las COLUMNAS de V, ortonormales).
    """
    n = len(A)
    A = [fila[:] for fila in A]                                   # copia de trabajo
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]  # V = I

    for _ in range(max_iter):
        # 1) Mayor elemento fuera de la diagonal
        p, q, mx = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > mx:
                    mx, p, q = abs(A[i][j]), i, j
        if mx < tol:           # ya es (casi) diagonal
            break

        # 2) Parámetros de la rotación
        theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
        t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta**2 + 1))
        c = 1 / math.sqrt(t**2 + 1)
        s = t * c

        # 3a) A ← A J  (actualiza columnas p y q)
        for r in range(n):
            arp, arq = A[r][p], A[r][q]
            A[r][p] = c * arp - s * arq
            A[r][q] = s * arp + c * arq
        # 3b) A ← Jᵀ A  (actualiza filas p y q)
        for r in range(n):
            apr, aqr = A[p][r], A[q][r]
            A[p][r] = c * apr - s * aqr
            A[q][r] = s * apr + c * aqr
        # 3c) V ← V J  (acumula los eigenvectores)
        for r in range(n):
            vrp, vrq = V[r][p], V[r][q]
            V[r][p] = c * vrp - s * vrq
            V[r][q] = s * vrp + c * vrq

    return [A[i][i] for i in range(n)], V


def main():
    A = matriz_dinamica(N, m, k)
    lam, V = jacobi(A)

    # Ordenar los modos de menor a mayor frecuencia
    orden = sorted(range(N), key=lambda i: lam[i])

    fig, ejes = plt.subplots(2, 2, figsize=(8, 6))
    for n_, idx in enumerate(orden):
        omega = math.sqrt(lam[idx])                       # ω = √λ
        v = [V[r][idx] for r in range(N)]                 # eigenvector del modo
        if v[0] < 0:                                      # convención de signo
            v = [-t for t in v]
        omega_teo = 2 * math.sqrt(k / m) * math.sin((n_ + 1) * math.pi / (2 * (N + 1)))
        print(f"Modo {n_+1}: λ = {lam[idx]:9.4f}   ω = {omega:8.4f} rad/s "
              f"(teórico {omega_teo:8.4f})   v = {[round(t, 4) for t in v]}")

        # Perfil espacial: se agregan las paredes fijas (desplazamiento 0)
        ax = ejes[n_ // 2][n_ % 2]
        ax.plot(range(N + 2), [0] + v + [0], "o-")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"Modo {n_+1},  ω = {omega:.2f} rad/s")
        ax.set_xlabel("posición de la masa j")
        ax.set_ylabel("amplitud")

    plt.tight_layout()
    plt.savefig("p2_modos.png", dpi=150)


if __name__ == "__main__":
    main()