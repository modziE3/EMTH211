import numpy as np

Q1B_TRANSMATRIX = np.array([[1/9., 4/13., 0., 0., 0., 0., 0., 4/25., 0.           ],
                            [4/9., 1/13., 4/17., 4/13., 0., 0., 0., 0., 0.        ],
                            [0., 4/13., 1/17., 4/13., 4/21., 0., 0., 4/25., 0.    ],
                            [0., 4/13., 4/17., 1/13., 4/21., 0., 0., 0., 0.       ],
                            [0., 0., 4/17., 4/13., 1/21., 4/13., 4/13., 4/25., 0. ],
                            [0., 0., 0., 0., 4/21., 1/13., 4/13., 4/25., 0.       ],
                            [0., 0., 0., 0., 4/21., 4/13., 1/13., 4/25., 0.       ],
                            [4/9., 0., 4/17., 0., 4/21., 4/13., 4/13., 1/25., 4/5.],
                            [0., 0., 0., 0., 0., 0., 0., 4/25., 1/5.              ]])

VALUES, VECTORS = np.linalg.eig(Q1B_TRANSMATRIX)


def is_stochastic(trans_matrix):
    """Checks a Markov Transition matrix 'trans_matrix'
       is stochastic by checking all the columns of the 
       matrixs sum to 1.
    """
    for col in range(len(list(trans_matrix))):
        if float(f"{(sum(trans_matrix[:, col])):.6f}") != 1:
            return False
    return True

def display_probability(eig_vals, eig_vecs):
    """Takes the Eigen vectors and values of a transition
       matrix to display the probability distribution vector.
    """
    dist_vector = one_norm_rescaler(eig_vecs[:, 0])
    for prob_num in range(len(list(dist_vector))):
        print(f"{prob_num+1}.  {(dist_vector[prob_num] * 100):.4f} %")

def one_norm_rescaler(vector):
    """Takes a vector and rescales it to get the
       one norm of the vector.
    """
    sum = 0
    for element in vector:
        sum+=element
    return ((1/sum) * vector)

if __name__ == '__main__':
    #print( is_stochastic( Q1B_TRANSMATRIX ) )
    #print(VECTORS[:, 0])
    #print(VALUES)
    print( display_probability(VALUES, VECTORS) )