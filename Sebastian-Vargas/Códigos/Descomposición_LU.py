def lu(A, b):
    n = len(A)

    # --- Descomposición LU (Doolittle) ---
    for k in range(n):
        for i in range(k + 1, n):
            A[i][k] /= A[k][k]                 # guardamos L en la parte inferior
            for j in range(k + 1, n):
                A[i][j] -= A[i][k] * A[k][j]   # actualizamos U

    # --- Sustitución hacia adelante: Ly = b ---
    y = [0] * n
    for i in range(n):
        y[i] = b[i] - sum(A[i][j] * y[j] for j in range(i))

    # --- Sustitución hacia atrás: Ux = y ---
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]

    return x


# ---------- PRUEBA ----------
A = [[4, 3,6],
     [6, 3,7],
     [5,8,3]]
b = [10, 12,17]

print("Solución:", lu(A, b))   # [1.0, 2.0]