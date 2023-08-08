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

if __name__ == '__main__':
    A = np.array([np.array ([1 ,2 ,3]) , np . array ([4 ,5 ,6]) , np . array ([7 ,8 , 9])])
    B = RowSwap(A, 1, 2)
    print(A)
    print(B)