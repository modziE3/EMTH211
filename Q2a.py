import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

SIZES = np.linspace(2, 10000, 100, dtype=int)


def solving_timer(sizes, iter):
    """Takes a list of matrix sizes 'n' and finds the solve
       times for nxn triangular matrices. Each matrix is
       random. This process can be averaged over a certain
       amount of iterations 'iter'.
    """
    solve_times = []
    for k in range(iter):
        times = []
        for n in sizes :
            A = np.triu(np.random.rand(n, n))
            b = np.ones(n)
            t0 = time.time()
            la.solve_triangular(A, b)
            t1 = time.time()
            times.append(t1 - t0)
        solve_times.append(np.array(times))
    solve_times = np.array(solve_times)
    average_times = []
    for col in range(len(list(solve_times[0]))):
        average_times.append(solve_times[:,col].mean())
    return average_times

def matrix_size_solvetime_plot(sizes, times):
    """Takes a list of triangular matrix sizes and graphs
       them against their solve times.
       IE:  Ax = b  where A is a triangular matrix and b
       is a vector.
    """
    x_Sizes = np.array(sizes)
    y_times = np.array(times)
    axes = plt.axes()
    axes.plot(x_Sizes, y_times)
    axes.grid(True)
    axes.set_xlabel(f" Matrix size (n)")
    axes.set_ylabel(f" Solve Time / Seconds (s)")
    axes.set_title(f" nxn Triangular Matrix size and solve time relationship")
    plt.show()

if __name__ == '__main__':
    times = solving_timer(SIZES, 1)
    matrix_size_solvetime_plot(SIZES, times)