import numpy as np
from scipy import linalg as sla

def forwardSub3(L, b):
    """Takes a lower triangular 3x3 matrix
       and solves for x vector from b;
       Lx = b      cost = O{ n^2 }
    """
    bREF = b.copy()
    x_0 = bREF[0]
    x_1 = bREF[1] - (L[1, 0] * x_0)
    x_2 = bREF[2] - (L[2, 0] * x_0) - (L[2, 1] * x_1)
    return np.array([x_0, x_1, x_2])

def backSub3(U, b):
    """Takes an upper triangular 3x3 matrix
       and solves for x vector from b;
       Ux = b      cost = O{ n^2 }
    """
    bREF = b.copy()
    x_2 = (1/U[2, 2]) * (bREF[2])
    x_1 = (1/U[1, 1]) * (bREF[1] - (U[1, 2] * x_2))
    x_0 = (1/U[0, 0]) * (bREF[0] - (U[0, 2] * x_2) - (U[0, 1] * x_1))
    return np.array([x_0, x_1, x_2])

if __name__ == '__main__':
    n = 3
    np.random.seed(seed=211)
    L, U = sla.lu(np.random.rand(n, n))[1:]
    b = np.random.rand(n)

    print(forwardSub3(L, b))
    print(np.linalg.solve(L, b))

    print(backSub3 (U , b))
    print(np.linalg.solve(U , b))