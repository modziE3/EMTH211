import numpy as np

def check_ind(vecs):
    """Takes a list of numpy arrays and determines whether
       they are linearly independent.
    """
    A = np.array(vecs)
    rank = np.linalg.matrix_rank(A)
    if rank != len(vecs):
        message = f"Rank is {rank}, while number of vectors is {len(vecs)}\nThese vectors are linearly dependent"
        ind = False
    else:
        message = f"Rank is {rank}, which is the same number of vectors {len(vecs)}\nThese vectors are linearly independent"
        ind = True
    print(message)
    return ind
    
if __name__ == '__main__':
    check_ind ([np.array ([1 ,2 ,3]) , np . array ([4 ,5 ,6]) , np . array ([7 ,8 , 9])])
    check_ind ([np.array ([1.,2.,3.]), np . array([-2., 1., 0.]), np . array ([10.,-5.,3.])])
    check_ind ([np.array ([1 ,2 ,3]) , np . array ([4 ,5 ,6])])
    check_ind ([np.array ([1 ,2 ,3]) , np . array ([4 ,8 , 12])])
    check_ind ([np.array ([1 ,4]) , np . array ([2 ,5]) , np . array ([3 ,6])])