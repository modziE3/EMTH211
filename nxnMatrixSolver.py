import numpy as np
from scipy import linalg as sla

def RowSwap(A, x, y, PastSwaps):
    """Takes a nxn matrix and swaps the desired rows
       x and y, ie.
       Rx <-> Ry
    """
    REF_A = A.copy()
    REF_Ax = A[x].copy()
    REF_Ay = A[y].copy()
    REF_A[x] = REF_Ay
    REF_A[y] = REF_Ax
    PastSwaps.append([x, y])
    return REF_A, PastSwaps

def myLU(A):
    """ Takes an nxn numpy array and computes its
        LU decomposition without partial pivoting
        and assuming that no row swaps are required . 
    """
    U = A.copy()
    n = len(list(A))
    L = np.eye(n)
    Swaps = []
    for i in range(n):
        for k in range(n)[1+i:]:
            L[k, i] = (U[k, i]/(U[i, i]))
            U[k] -= ((U[k, i]/(U[i, i])) * (U[i]))
        try:
            if U[i + 1, i + 1] == 0:
                row = i + 2
                while U[row, i + 1] == 0:
                    row += 1
                U, Swaps = RowSwap(U, i + 1, row, Swaps)
        except:
            pass
    if Swaps != []:
        L = L_RowSwapper(L, Swaps, True)
    return L , U, Swaps

def L_RowSwapper(L, Swaps, ForwardSwap):
    """Takes matrix L and row swaps with
       row swap numbers from matrix U.
    """
    L_Swaps = []
    n = len(Swaps)
    if ForwardSwap == True:
        for i in range(n):
            L, L_Swaps = RowSwap(L, int(Swaps[i][0]), int(Swaps[i][1]), L_Swaps)
    else:
        for i in range(n):
            L, L_Swaps = RowSwap(L, int(Swaps[n-i-1][0]), int(Swaps[n-i-1][1]), L_Swaps)
    return L

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

def Matrix_solver(A, b):
    """Solves an nxn matrix that can handle
       row swaps.
    """
    L, U, Swaps = myLU(A)
    Lsolver = L_RowSwapper(L, Swaps, False)
    y = forwardSub3(Lsolver, b)
    y = L_RowSwapper(y, Swaps, True)
    x = backSub3 (U , y)
    return(x)

if __name__ == '__main__':
    n = 1000
    np.random.seed(seed=211)
    #A = np.random.rand(n, n)
    #b = np.random.rand(n)
    A = np.array([[1., 2.,4.,4.],[1.,2.,4.,8.],[1.,2.,8.,8.],[1.,4.,8.,8.]])
    b = np.array([1.,2.,4.,8.])
    x = Matrix_solver(A, b)
    print(x)
    print(np.linalg.solve(A, b))