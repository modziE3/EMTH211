import numpy as np
from scipy import linalg as sla

def forwardSub3(L, b):
    """Takes a lower triangular mxm matrix
       and solves for x vector from b;
       Lx = b      cost = O{ n^2 }
    """
    bREF = b.copy()
    Lsize = len(list(L))
    x = np.ones(Lsize)
    for i in range(Lsize):
        subs = 0
        n = i - 1
        while n > -1:
            subs += L[i, n] * x[n]
            n -= 1
        x[i] = bREF[i] - subs
    return x

def backSub3(U, b):
    """Takes an upper triangular mxm matrix
       and solves for x vector from b;
       Ux = b      cost = O{ n^2 }
    """
    bREF = b.copy()
    Lsize = len(list(U))
    x = np.ones(Lsize)
    for i in range(Lsize):
        subs = 0
        k = Lsize -1 - i
        n = k + 1
        while n < Lsize:
            subs += U[k, n] * x [n]
            n += 1
        x[k] = (1/U[k, k]) * (bREF[k] - subs)
    return x

if __name__ == '__main__':
    n = 6
    np.random.seed(seed=211)
    L, U = sla.lu(np.random.rand(n, n))[1:]
    b = np.random.rand(n)

    print(forwardSub3(L, b))
    print(np.linalg.solve(L, b))

    print(backSub3 (U , b))
    print(np.linalg.solve(U , b))