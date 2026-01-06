import matplotlib.pyplot as plt
import numpy as np
from fun import homography_extraction

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == "__main__":
    image_path = 'challenge1.png'
    img = plt.imread(image_path)
    
    # Redressement du mur
    print("--- ETAPE 1 : RECTIFICATION DU MUR ---")
    plt.figure()
    plt.imshow(img)
    plt.title("Cliquez les 4 coins du MUR ORANGE pour le redresser")
    pts_mur = plt.ginput(4, timeout=-1)
    plt.close()
    
    x_mur = [p[0] for p in pts_mur]
    y_mur = [p[1] for p in pts_mur]
    
    W_rect, H_rect = 1000, 700 
    img_rectifiee = homography_extraction(img, x_mur, y_mur, W_rect, H_rect)
    
    # Mesure des Aires
    nb_carres = 9
    liste_carres = [] 
    
    print("\n--- ETAPE 2 : MESURE  ---")
    print("Pour chaque carré : 1. Extrémité -> 2. COIN -> 3. Extrémité")
    
    # On garde la figure dans une variable pour pouvoir dessiner dessus
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rectifiee)
    ax.set_title(f"Cliquez sur les carrés pour mesurer leur aire")
    
    for i in range(nb_carres):
        numero = i + 1
        print(f"Mesure du Carré n°{numero}...")
        
        # 3 clics par carré pour eviter les cas ou un coin est coupé
        pts = plt.ginput(3, timeout=-1)
        
        if len(pts) == 3:
            # Calcul de l'aire
            cote_a = distance(pts[0], pts[1])
            cote_b = distance(pts[1], pts[2])
            aire = cote_a * cote_b
            
            liste_carres.append({'nom': f"Carré n°{numero}", 'aire': aire})
            
            # On dessine pour vérification
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            # Trace les côtés mesurés en rouge
            ax.plot(xs, ys, 'r-', linewidth=2) 
            # Écrit le numéro en gros au niveau du coin 
            ax.text(pts[1][0], pts[1][1], str(numero), color='white', fontsize=15, fontweight='bold')
            # Force la mise à jour de l'image
            plt.draw()
            
        else:
            print(f"Erreur sur le carré n°{numero}, aire mise à 0.")
            liste_carres.append({'nom': f"Carré n°{numero}", 'aire': 0})

    
    # Affichage dans le terminal
    if not liste_carres:
        print("Aucune mesure effectuée.")
        exit()

    # Liste du plus grand au plus petit
    classement = sorted(liste_carres, key=lambda k: k['aire'], reverse=True)
    
    print("\n" + "="*45)
    print(f"{'RANG':<5} | {'IDENTIFIANT':<15} | {'AIRE (px)':<12}")
    print("="*45)
    
    for rang, carre in enumerate(classement):
        print(f"{rang+1:<5} | {carre['nom']:<15} | {carre['aire']:.0f}")
    print("="*45)
    print("\nVous pouvez vérifier les résultats sur l'image.")
    print("Fermez la fenêtre graphique pour terminer le script.")
    
    plt.show()