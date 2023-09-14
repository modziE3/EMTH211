import t5_Q8 as Sol
import numpy as np


def Jacobi(A, b, x0, max_iter):
    """Uses Jacobi Iteration to find the solution to the 
       Equation Ax=b. 
    """
    P = np.diagflat(np.diag(A))
    Q = A - P
    for k in range(max_iter):
        x0 = Sol.backSub(P, (-Q @ x0) + b)
    return x0


def Gauss_Seidel(A, b, x0, max_iter):
    """Uses Gauss-Seidel Iteration to find the solution to the 
       Equation Ax=b. 
    """
    P = np.triu(A)
    Q = A - P
    for k in range(max_iter):
        x0 = Sol.backSub(P, (-Q @ x0) + b)
    return x0


if __name__ == '__main__':
    B = np.array([[2.,0.5,-0.5],[0.,4.,-2.],[4.,0.,-4.]])
    c = np.array([4.,4.,0.])
    x = np.array([1.,0.,0.])
    num = 10
    print(Jacobi(B, c, x, num))
    print(Gauss_Seidel(B, c, x, num))
    print(Sol.Matrix_solver(B, c))