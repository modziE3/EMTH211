import numpy as np

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

if __name__ == '__main__':
    np . random . seed ( seed = 211 )
    A = np . random . rand (4 , 4)
    L , U = myLU (A)
    print (f'L is\n{L}\n U is\n{U}\nL @ U == A: {np. allclose (A, L @ U)}')
