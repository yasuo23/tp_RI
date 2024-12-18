import numpy as np

# Données originales de rappel (Ra) et précision (Pr)
rappel = [0.07, 0.13, 0.20, 0.27, 0.33, 0.40, 0.47, 0.53, 0.60, 0.67, 0.90]
precision = [1.00, 0.50, 0.75, 0.67, 0.71, 0.67, 0.64, 0.67, 0.64, 0.67, 0.01]

# Points de rappel interpolés (rj) spécifiés dans l'énoncé
rappel_interpole = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Initialisation d'une liste pour stocker les précisions interpolées
precision_interpolee = []

# Interpolation : trouver la précision maximale pour r >= rj
for rj in rappel_interpole:
    # Filtrer les précisions pour les valeurs de rappel >= rj
    precisions_filtrees = [pr for ra, pr in zip(rappel, precision) if ra >= rj]
    
    # Ajouter la précision maximale (ou 0 si aucune précision)
    precision_interpolee.append(max(precisions_filtrees) if precisions_filtrees else 0)

# Afficher les résultats
print("Ra (rappel interpolé) | Pr (précision interpolée)")
for rj, pr in zip(rappel_interpole, precision_interpolee):
    print(f"{rj:.1f}                 | {pr:.2f}")
