import numpy as np

def homography_estimate(x1, y1, x2, y2):
    
    # Nb points
    n = len(x1)
    
    A = []
    B = []

    for i in range(n):
        A.append([x1[i], y1[i], 1, 0, 0, 0, -x1[i]*x2[i], -y1[i]*x2[i]])
        B.append(x2[i])
        
        A.append([0, 0, 0, x1[i], y1[i], 1, -x1[i]*y2[i], -y1[i]*y2[i]])
        B.append(y2[i])

    A = np.array(A)
    B = np.array(B)
    
    #fonction donnant solution exacte (pour n=4) 
    #ou donnant solution optimal au sens des moindres carrés (pour n>4)
    #fournit par un llm
    X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    H = np.array([
        [X[0], X[1], X[2]],
        [X[3], X[4], X[5]],
        [X[6], X[7], 1.0]
    ])

    return H


def homography_apply(H, x1, y1):

    x1 = np.array(x1)
    y1 = np.array(y1)
    
    num_x = H[0, 0] * x1 + H[0, 1] * y1 + H[0, 2]
    num_y = H[1, 0] * x1 + H[1, 1] * y1 + H[1, 2]
    
    denominator = H[2, 0] * x1 + H[2, 1] * y1 + H[2, 2]
    
    x2 = num_x / denominator
    y2 = num_y / denominator
    
    return x2, y2



 


