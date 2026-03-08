import numpy as np
from kmeans_lib.initialization import RandomInit

X = np.random.rand(50, 4)
init = RandomInit()
centers = init.initialize(X, 3)

print(f"Centers shape: {centers.shape}")  # Attendu: (3, 4)
assert centers.shape == (3, 4), "Erreur de shape !"
print("RandomInit OK !")