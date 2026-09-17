def gauss(A, b):
    n = len(A)

    # --- Eliminación hacia adelante ---
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # --- Sustitución inversa ---
    x = [0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - suma) / A[i][i]

    return x


# ---------- PRUEBA ----------
A = [[2, 1, -1],
     [-3, -1, 2],
     [-2, 1, 2]]
b = [8, -11, -3]

print("Solución:", gauss(A, b))   # [2.0, 3.0, -1.0]