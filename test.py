import numpy as np
from kmeans_lib.distances import compute_distances

X = np.array([[0, 0], [1, 1], [2, 2]])
centers = np.array([[0.5, 0.5], [1.5, 1.5]])

distances = compute_distances(X, centers)
print(distances)
print(f"Shape: {distances.shape}")  # Doit être (3, 2)