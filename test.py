import numpy as np
from kmeans_lib.core import KMeans

# ─────────────────────────────────────────────────────
# 1. Création des données d'entraînement
# ─────────────────────────────────────────────────────
np.random.seed(42)
X_train = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5]
])

# ─────────────────────────────────────────────────────
# 2. Entraînement du modèle
# ─────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100)
kmeans.fit(X_train)

print(f"✅ Modèle entraîné en {kmeans.n_iter_} itérations")
print(f"   Inertie: {kmeans.inertia_:.2f}")

# ─────────────────────────────────────────────────────
# 3. Test de predict() sur de NOUVELLES données
# ─────────────────────────────────────────────────────
X_test = np.array([[0.5, 0.5], [5.5, 5.5], [2.5, 2.5]])
predictions = kmeans.predict(X_test)

print(f"\n✅ Prédictions sur nouvelles données:")
for i, (point, label) in enumerate(zip(X_test, predictions)):
    print(f"   Point {i} {point} → Cluster {label}")

# ─────────────────────────────────────────────────────
# 4. Test de transform()
# ─────────────────────────────────────────────────────
distances = kmeans.transform(X_test)

print(f"\n✅ Distances aux centroïdes:")
for i, (point, dist) in enumerate(zip(X_test, distances)):
    print(f"   Point {i} → Distances: [{dist[0]:.3f}, {dist[1]:.3f}]")

# ─────────────────────────────────────────────────────
# 5. Test de score()
# ─────────────────────────────────────────────────────
score = kmeans.score(X_test)
print(f"\n✅ Score (négatif de l'inertie): {score:.2f}")

# ─────────────────────────────────────────────────────
# 6. Vérification d'erreur (modèle non fitted)
# ─────────────────────────────────────────────────────
kmeans_unfitted = KMeans(n_clusters=2)
try:
    kmeans_unfitted.predict(X_test)
    print("❌ Erreur: aurait dû lever une exception")
except ValueError as e:
    print(f"\n✅ Gestion d'erreur OK: {e}")

print("\n🎉 Toutes les tests sont passés !")