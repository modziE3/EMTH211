import numpy as np

Q1C_TRANSMATRIX = np.array([[1/7., 2/7., 0., 0., 0., 0., 0., 2/13., 2/5.       ],
                            [2/7., 1/7., 2/9., 2/7., 0., 0., 0., 0., 0.        ],
                            [0., 2/7., 1/9., 2/7., 2/11., 0., 0., 2/13., 0.    ],
                            [0., 2/7., 2/9., 1/7., 2/11., 0., 0., 0., 0.       ],
                            [0., 0., 2/9., 2/7., 1/11., 2/7., 2/7., 2/13., 0.  ],
                            [0., 0., 0., 0., 2/11., 1/7., 2/7., 2/13., 0.      ],
                            [0., 0., 0., 0., 2/11., 2/7., 1/7., 2/13., 0.      ],
                            [2/7., 0., 2/9., 0., 2/11., 2/7., 2/7., 1/13., 2/5.],
                            [2/7., 0., 0., 0., 0., 0., 0., 2/13., 1/5.         ]])

VALUES, VECTORS = np.linalg.eig(Q1C_TRANSMATRIX)


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
    #print( is_stochastic( Q1C_TRANSMATRIX ) )
    #print(VECTORS[:, 0])
    #print(VALUES[0])
    print( display_probability(VALUES, VECTORS) )