import numpy as np

# Matriz de coeficientes
A = np.array([
    [1, 4, 2],
    [5, 2, 1],
    [2, 8, 1]
], dtype=float)

# Vector de términos independientes
b = np.array([13, 20, 23], dtype=float)

# Hallar la inversa de A
A_inv = np.linalg.inv(A)

# Resolver I = A^(-1) b
I = A_inv @ b

print("Matriz inversa:")
print(A_inv)

print("\nSolución:")
print(I)