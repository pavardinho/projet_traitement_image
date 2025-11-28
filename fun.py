import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

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




def homography_extraction(I1, x, y, w, h):
    
    # ici je défini les coordonées des point d'ancrage de la future image (ordre : haut gauche, haut droit, bas droit, bas gauche)
    x_dest = np.array([0, w-1, w-1, 0])
    y_dest = np.array([0, 0, h-1, h-1])
    
    # calcul de l'homographie
    H = homography_estimate(x_dest, y_dest, x, y)
    
    # on applique l'homographie
    indices_x, indices_y = np.meshgrid(np.arange(w), np.arange(h))
    
    x_formate = indices_x.flatten()
    y_formate = indices_y.flatten()
    
    x_src, y_src = homography_apply(H, x_formate, y_formate)
    
    coords = np.vstack((y_src, x_src)) 
    
    # cas ou on est en 2d
    if I1.ndim == 2:
        pixel_values = map_coordinates(I1, coords, order=1, mode='constant', cval=0)
        # cval=0 met du noir si on tombe hors de l'image source
        
        I2 = pixel_values.reshape((h, w))
        
    # cas où on est en 3d
    elif I1.ndim == 3:
        I2 = np.zeros((h, w, 3), dtype=I1.dtype)
        for channel in range(3):
            pixel_values = map_coordinates(I1[:,:,channel], coords, order=1, mode='constant', cval=0)
            I2[:,:,channel] = pixel_values.reshape((h, w))
            
    return I2

 


