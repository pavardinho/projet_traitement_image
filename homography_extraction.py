import matplotlib.pyplot as plt
import numpy as np
    from scipy.ndimage import map_coordinates

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
    
    coords = np.vstack((y_src_flat, x_src_flat)) 
    
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
