import numpy as np
import matplotlib.pyplot as plt
# On importe la fonction depuis votre fichier fun.py
# Note : fun.py doit être dans le même dossier que main.py
from fun import homography_extraction

if __name__ == "__main__":
    print("=== TEST DE RECTIFICATION D'IMAGE ===")
    
    # 1. Création d'une image synthétique (Damier) pour tester
    # Cela permet de bien voir les déformations géométriques
    H_img, W_img = 400, 400
    checkers = np.zeros((H_img, W_img, 3), dtype=np.uint8)
    step = 50
    for y in range(0, H_img, step):
        for x in range(0, W_img, step):
            if (x // step + y // step) % 2 == 0:
                checkers[y:y+step, x:x+step, :] = 255 # Cases blanches
    
    # On ajoute un carré rouge au milieu pour l'orientation
    checkers[150:250, 150:250, 0] = 255 
    checkers[150:250, 150:250, 1:] = 0

    # 2. Affichage et Interaction
    plt.figure(figsize=(8, 8))
    plt.imshow(checkers)
    plt.title("CLIQUEZ 4 POINTS (Ordre: HG -> HD -> BD -> BG)\nEssayez de former un trapèze pour voir l'effet !")
    print("Veuillez cliquer 4 points dans l'image...")
    
    # Récupération des clics (Section 3.2 [111])
    pts = plt.ginput(4, timeout=-1)
    plt.close()

    if len(pts) == 4:
        x_clics = [p[0] for p in pts]
        y_clics = [p[1] for p in pts]
        
        # Dimensions de sortie (arbitraires ou ratio A4)
        w_out, h_out = 300, 300
        
        print("Calcul de l'homographie et extraction...")
        try:
            # Appel de votre fonction
            res = homography_extraction(checkers, x_clics, y_clics, w_out, h_out)
            
            # Affichage Comparatif
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Image originale avec les points cliqués
            ax[0].imshow(checkers)
            # On referme la boucle pour dessiner le quadrilatère
            ax[0].plot(x_clics + [x_clics[0]], y_clics + [y_clics[0]], 'r-o', linewidth=2)
            ax[0].set_title("Source (Zone sélectionnée)")
            
            # Résultat redressé
            ax[1].imshow(res.astype(np.uint8)) # Conversion sûre pour affichage
            ax[1].set_title(f"Résultat redressé ({w_out}x{h_out})")
            
            plt.show()
            
        except Exception as e:
            print(f"Erreur pendant l'exécution : {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Erreur : Vous n'avez pas cliqué 4 points.")


