import numpy as np

def myRowEchelon3 (A):
    """ Takes a 3x3 matrix and uses Gaussian
        elimination to find its row echelon form . 
    """
    A[1] = A[1] - ((A[1, 0]/(A[0, 0])) * (A[0]))
    A[2] = A[2] - ((A[2, 0]/(A[0, 0])) * (A[0]))
    A[2] = A[2] - ((A[2, 1]/(A[1, 1])) * (A[1]))
    return A

A = np . array ([[2., -1. , 1.], [-2., 3., -1.], [4. , -15., 7.]]) # From question 1
np . random . seed ( seed = 211 ) # Sets the seed so everyone gets the same matrix !
B = np . random . rand (3 , 3) # A random matrix
print ( myRowEchelon3 (A))
print ( myRowEchelon3 (B))
