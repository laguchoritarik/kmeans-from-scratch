import numpy as np
from kmeans_lib.core import KMeans

# Données de test (2 clusters bien séparés)
np.random.seed(42)  # On fixe le seed avant d'appeler KMeans
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],   # Cluster 1
    np.random.randn(50, 2) + [5, 5]    # Cluster 2
])

# Test avec initialisation random (CORRECTÉ)
kmeans = KMeans(n_clusters=2, init='random', max_iter=100)  # ✅ 'init' pas 'init_strategy'
kmeans.fit(X)

print(f"Nombre d'itérations: {kmeans.n_iter_}")
print(f"Inertie: {kmeans.inertia_:.2f}")
print(f"Shape des centroïdes: {kmeans.cluster_centers_.shape}")
print(f"Shape des labels: {kmeans.labels_.shape}")

# Vérifications
assert kmeans.cluster_centers_.shape == (2, 2), "Erreur shape centroïdes !"
assert kmeans.labels_.shape == (100,), "Erreur shape labels !"

print("\n✅ KMeans fit() OK !")