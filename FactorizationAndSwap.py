import numpy as np

def RowSwap(A, x, y):
    """Takes a nxn matrix and swaps the desired rows
       x and y, ie.
       Rx <-> Ry
    """
    REF_A = A.copy()
    REF_Ax = A[x].copy()
    REF_Ay = A[y].copy()
    REF_A[x] = REF_Ay
    REF_A[y] = REF_Ax
    return REF_A

def myLU ( A):
    """ Takes an nxn numpy array and computes its
    LU de co mp os it io n without partial pivoting
    and assuming that no row swaps are required . 
    """
    U = A.copy()
    n = len(list(A))
    L = np.eye(n)
    for i in range(n):
        for k in range(n)[1+i:]:
            L[k, i] = (U[k, i]/(U[i, i]))
            U[k] -= ((U[k, i]/(U[i, i])) * (U[i]))
    return L , U

A = np.array([np.array ([1.,2.,4.]),np.array([1.,2.,8.]),np.array([1.,4.,8.])])
B = RowSwap(A, 1, 2)
L, U = myLU(A)
print(f"{L}\n{U}\n")
L, U = myLU(B)
print(f"{L}\n{U}\n")