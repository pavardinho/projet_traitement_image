import matplotlib.pyplot as plt
import numpy as np
# On importe ta fonction depuis le fichier "functions.py"
# Si tu as nommé ton fichier autrement, change "functions" par le nom de ton fichier
from fun import homography_projection

def main():
    print("--- Chargement des images ---")
    
    # 1. Chargement des images
    # Utilisation de plt.imread (gère jpg, png...)

    I1 = plt.imread('../mona.jpg')       # Image Source (à projeter)
    I2 = plt.imread('../timesquare.jpg') # Image Destination (le décor)


    print(f"Source : {I1.shape} | Destination : {I2.shape}")

    # 2. Affichage pour la sélection des points
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(I2)
    plt.title("CLIQUEZ 4 POINTS sur l'écran géant (ou la zone cible)\nOrdre : Haut-Gauche -> Haut-Droite -> Bas-Droite -> Bas-Gauche")
    plt.axis('on')
    
    print("En attente de 4 clics sur l'image...")
    
    # ginput permet de récupérer les clics de souris
    # n=4 : on attend 4 clics
    # timeout=-1 : pas de limite de temps
    points = plt.ginput(n=4, timeout=-1)
    
    plt.close(fig) # On ferme la fenêtre de sélection

    # 3. Traitement des coordonnées
    if len(points) < 4:
        print("Annulation : Vous n'avez pas cliqué assez de points.")
        return

    # Séparation des X et Y
    # points est une liste de tuples [(x1,y1), (x2,y2)...]
    x_clic = [p[0] for p in points]
    y_clic = [p[1] for p in points]

    print("Points récupérés :")
    print(f"X : {np.round(x_clic)}")
    print(f"Y : {np.round(y_clic)}")
    print("Calcul de la projection en cours (cela peut prendre quelques secondes)...")

    # 4. Appel de TA fonction importée
    I_resultat = homography_projection(I1, I2, x_clic, y_clic)

    # 5. Affichage du résultat final
    plt.figure(figsize=(12, 6))
    
    # Affichage comparatif
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.title("Image Source (Mona Lisa)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(I_resultat)
    plt.title("Résultat : Projection sur Times Square")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Protection pour exécuter le main seulement si on lance ce fichier directement
if __name__ == "__main__":
    main()
