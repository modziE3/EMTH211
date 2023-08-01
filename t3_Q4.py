import numpy as np

def myLU3 (A):
    """ Takes a 3x3 numpy array and computes its
        LU de co mp os it io n without partial pivoting
        and assuming that no row swaps are required . 
    """
    L = np.eye(3)
    L[1, 0] = (A[1, 0]/(A[0, 0]))
    L[2, 0] = (A[2, 0]/(A[0, 0]))
    A[1] -= ((A[1, 0]/(A[0, 0])) * (A[0]))
    A[2] -= ((A[2, 0]/(A[0, 0])) * (A[0]))
    L[2, 1] = (A[2, 1]/(A[1, 1]))
    A[2] -= ((A[2, 1]/(A[1, 1])) * (A[1]))
    return L , A

if __name__ == '__main__':
    A = np . array ([[1., 2.,-1.],[-2.,-5.,3.],[-1.,-3.,0.]])
    L , U = myLU3 (A )
    print (f"L is\n{L}\nU is\n{U}\nL @ U =\n{L @ U}\n= A")
    np . random . seed ( seed = 211 )
    B = np . random . rand (3 , 3)
    L , U = myLU3 (B )
    print (f"L is\n{L}\nU is\n{U}\nL @ U =\n{L @ U}\n= A")

