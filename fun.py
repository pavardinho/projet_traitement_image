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


def homography_projection(I1, I2, x, y):
  
    I2_result = I2.copy()
    
    h_src, w_src = I1.shape[:2]
    h_dst, w_dst = I2.shape[:2]
    
    # coins : (0,0); (w,0); (w,h); (0,h)
    x_src = [0, w_src - 1, w_src - 1, 0]
    y_src = [0, 0, h_src - 1, h_src - 1]

    H = homography_estimate(x_src, y_src, x, y)

    H_inv = np.linalg.inv(H)
    
    min_x = max(0, int(min(x)))
    max_x = min(w_dst, int(max(x)) + 1)
    min_y = max(0, int(min(y)))
    max_y = min(h_dst, int(max(y)) + 1)

    for v_dst in range(min_y, max_y):
        for u_dst in range(min_x, max_x):
            
            denominator = H_inv[2, 0] * u_dst + H_inv[2, 1] * v_dst + H_inv[2, 2]
            
            src_x = (H_inv[0, 0] * u_dst + H_inv[0, 1] * v_dst + H_inv[0, 2]) / denominator
            src_y = (H_inv[1, 0] * u_dst + H_inv[1, 1] * v_dst + H_inv[1, 2]) / denominator
            
            u_src = int(round(src_x))
            v_src = int(round(src_y))

            if 0 <= u_src < w_src and 0 <= v_src < h_src:
                I2_result[v_dst, u_dst] = I1[v_src, u_src]

    return I2_result

def homography_cross_projection(I, x1, y1, x2, y2):
    # On travaille sur une copie pour ne pas écraser les pixels dont on a besoin
    I_result = I.copy()

    I_source_3d = np.atleast_3d(I)
    
    h_img, w_img = I.shape[:2]
    
    # définition du "carré virtuel"
    size_temp = 400
    x_s = np.array([0, size_temp, size_temp, 0])
    y_s = np.array([0, 0, size_temp, size_temp])
    
    # calcul homographies de base avec le carre virtuel come destination
    H_as = homography_estimate(x1, y1, x_s, y_s)
    H_bs = homography_estimate(x2, y2, x_s, y_s)
    
    # Pour remplir A, on doit savoir où chercher dans B.
    # Chemin : Pixel dans A -> (H_as) -> Carré S -> (inv H_bs) -> Pixel dans B
    H_sb = np.linalg.inv(H_bs)
    H_fill_A = np.matmul(H_sb, H_as)
    
    # Pour remplir B, on doit savoir où chercher dans A.
    # Chemin : Pixel dans B -> (H_bs) -> Carré S -> (inv H_as) -> Pixel dans A
    H_sa = np.linalg.inv(H_as)
    H_fill_B = np.matmul(H_sa, H_bs)
    
    # fonction pour remplir une zone
    def fill_zone(x_corners, y_corners, H_transform, H_check_inside):

        x_min, x_max = int(np.min(x_corners)), int(np.max(x_corners))
        y_min, y_max = int(np.min(y_corners)), int(np.max(y_corners))
        

        x_min, x_max = max(0, x_min), min(w_img, x_max + 1)
        y_min, y_max = max(0, y_min), min(h_img, y_max + 1)
        
        # grille de pixels de la zone destination
        xv, yv = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        xf = xv.flatten()
        yf = yv.flatten()
        
        # Astuce du prof : On projette le pixel dans le Carré Virtuel.
        # S'il tombe entre [0, size_temp], c'est qu'il est dans le quad
        xt, yt = homography_apply(H_check_inside, xf, yf)
        mask = (xt >= 0) & (xt <= size_temp) & (yt >= 0) & (yt <= size_temp)
        
        if not np.any(mask): return # sinon zone vide
        
        # On garde que les pixels valides
        xf_valid = xf[mask]
        yf_valid = yf[mask]
        
        # On applique la matrice combinée
        x_src, y_src = homography_apply(H_transform, xf_valid, yf_valid)
        
        # interpolation depuis l'image originale
        coords = np.vstack((y_src, x_src))
        
        I_res_3d = np.atleast_3d(I_result)
        for i in range(I_res_3d.shape[2]):
            vals = map_coordinates(I_source_3d[:,:,i], coords, order=1, mode='nearest')
            I_res_3d[yf_valid, xf_valid, i] = vals
    
    # Remplir la zone 1 avec le contenu de la zone 2
    fill_zone(x1, y1, H_fill_A, H_as)
    
    # Remplir la zone 2 avec le contenu de la zone 1
    fill_zone(x2, y2, H_fill_B, H_bs)
    
    return I_result


