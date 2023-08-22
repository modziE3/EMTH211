import t5_Q8 as sl
import numpy as np

def repeated_mult (A_matrix, x0_vec, iteration_num):
    """takes a matrix A, initial vector x(0), and
       number of iterations n, and returns a matrix 
       whose rows are   x(0), x(1), ... , x(n) .
    """
    ini_matrixlist = []
    eig_vals, Perm_matrix = np.linalg.eig(A_matrix)
    Diag_matrix = np.eye(len(list(eig_vals)))
    for value_ind in range(len(list(eig_vals))):
        Diag_matrix[value_ind, value_ind] = eig_vals[value_ind]
    z_vec = sl.Matrix_solver(Perm_matrix, x0_vec)
    for power in range(iteration_num):
        ini_matrixlist.append(Perm_matrix @ ((Diag_matrix ** (power+1)) @ z_vec))
    return np.array(ini_matrixlist)

if __name__ == '__main__':
    A = np.array([[0.95,0.10],[0.05,0.90]])
    x0 = np.array([900.,300.])
    n=52
    print(repeated_mult(A, x0, n)[12])
