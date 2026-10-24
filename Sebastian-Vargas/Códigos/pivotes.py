import numpy as np


def gauss_pivot(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n - 1):

        # Partial pivoting
        i_max = np.argmax(
            np.abs(A[k:, k])
        ) + k

        if i_max != k:
            A[[k, i_max]] = A[[i_max, k]]
            b[[k, i_max]] = b[[i_max, k]]

        # Elimination
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # Back-substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] - A[i, i+1:] @ x[i+1:]
        ) / A[i, i]

    return x


# -------------------------
# Sistema que queremos resolver
# -------------------------

A = np.array([
    [4, -1, -1,0],
    [1,  4, 0, 1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
])

b = np.array([100, 100, 0, 0])


# Resolver
x = gauss_pivot(A, b)

# Mostrar resultado
print(x)