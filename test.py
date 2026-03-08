import numpy as np
from kmeans_lib.initialization import KMeanPlusPlusInit

# Données de test
np.random.seed(42)  # Pour la reproductibilité
X = np.random.rand(100, 2)

# Test KMeans++
kpp_init = KMeanPlusPlusInit()
centers = kpp_init.initialize(X, 5)

print(f"Shape des centroïdes: {centers.shape}")  # Doit être (5, 2)
assert centers.shape == (5, 2), "Erreur de shape !"

# Vérifier que les centroïdes sont bien des points de X
for center in centers:
    assert np.any(np.all(X == center, axis=1)), "Centroïde n'est pas dans X !"

print("K-Means++ Init OK !")