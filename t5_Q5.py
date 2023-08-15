import numpy as np

def repeated_mult_by_matrix (A , k , x):
    """A^K*x=b"""
    val, vecs = np.linalg.eig(A)
    c = np.linalg.solve(vecs, x)
    sum = 0
    for i in range(len(list(A))):
        sum += c[i] * (val[i] ** k) * vecs[:,i]
    return sum


if __name__ == '__main__':
    n = 5
    k = 10
    np.random.seed(seed=211)

    A = np.random.rand(n, n)
    x = np.random.rand(n)
    b = repeated_mult_by_matrix (A, k, x)
    b_check = np.linalg.matrix_power(A , k) @ x

    print(np.allclose(b, b_check))