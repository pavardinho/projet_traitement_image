import matplotlib.pyplot as plt
import numpy as np
import os
# On importe la fonction d'échange depuis votre fichier fun.py
from fun import homography_cross_projection

if __name__ == "__main__":
    print("=== TEST PROJECTIONS CROISÉES (IMAGE PANNEAUX) ===")
    
    # 1. Chargement de l'image
    image_path = 'panneaux.jpg'
    
    if not os.path.exists(image_path):
        print(f"ERREUR: L'image '{image_path}' est introuvable dans le dossier.")
        print("Veuillez vérifier le nom du fichier ou le placer au même endroit que ce script.")
        # Création d'une image de secours si le fichier manque
        print("-> Création d'une image de test générique...")
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        img[:,:] = 100 # Gris
        img[100:300, 100:300, 0] = 255 # Carré Rouge (Zone 1)
        img[100:300, 500:700, 2] = 255 # Carré Bleu (Zone 2)
    else:
        print(f"Chargement de {image_path}...")
        img = plt.imread(image_path)
        # Si l'image est en float [0,1], on la remet en uint8 [0,255] pour éviter des soucis d'affichage
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

    # 2. Affichage et Interaction
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title("CLIQUEZ 8 POINTS :\n4 sur le Panneau A (HG->HD->BD->BG)\npuis 4 sur le Panneau B (HG->HD->BD->BG)")
    
    print("\n--- INSTRUCTIONS ---")
    print("1. Cliquez sur les 4 coins du PREMIER panneau à échanger.")
    print("   (Ordre conseillé : Haut-Gauche -> Haut-Droit -> Bas-Droit -> Bas-Gauche)")
    print("2. Cliquez sur les 4 coins du SECOND panneau.")
    print("   (Même ordre)")
    print("--------------------\n")
    print("En attente de 8 clics...")
    
    # Récupération des clics
    # timeout=-1 signifie qu'on attend indéfiniment
    pts = plt.ginput(8, timeout=-1)
    plt.close() # On ferme la fenêtre de sélection

    if len(pts) == 8:
        # Séparation des points en deux groupes
        pts_zone1 = pts[0:4]
        pts_zone2 = pts[4:8]
        
        # Transformation en listes séparées x et y
        x1 = [p[0] for p in pts_zone1]
        y1 = [p[1] for p in pts_zone1]
        
        x2 = [p[0] for p in pts_zone2]
        y2 = [p[1] for p in pts_zone2]
        
        print("Points récupérés.")
        print("Calcul de l'échange en cours... (Patientez, cela peut prendre quelques secondes)")
        
        try:
            # Appel de votre fonction dans fun.py
            res = homography_cross_projection(img, x1, y1, x2, y2)
            
            print("Calcul terminé ! Affichage du résultat.")
            
            # Affichage Comparatif
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            
            # Image Originale avec tracé des zones
            ax[0].imshow(img)
            # On boucle les points pour dessiner les quadrilatères fermés
            ax[0].plot(x1 + [x1[0]], y1 + [y1[0]], 'r-', linewidth=2, label='Zone 1')
            ax[0].plot(x2 + [x2[0]], y2 + [y2[0]], 'c-', linewidth=2, label='Zone 2')
            ax[0].legend()
            ax[0].set_title("Image Originale (Zones cliquées)")
            ax[0].axis('off')
            
            # Résultat
            ax[1].imshow(res)
            ax[1].set_title("Résultat : Panneaux échangés")
            ax[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"\nERREUR PENDANT LE CALCUL : {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"\nErreur : Vous avez cliqué {len(pts)} points. Il en faut exactement 8.")