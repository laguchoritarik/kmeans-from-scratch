# K-Means From Scratch

Une implémentation de l'algorithme K-Means en Python avec NumPy uniquement.

## Installation

```bash
pip install kmeans-from-scratch

# K-Means From Scratch 🎯

[![PyPI version](https://badge.fury.io/py/kmeans-from-scratch-laguchori.svg)](https://pypi.org/project/kmeans-from-scratch-laguchori/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Une implémentation **from scratch** de l'algorithme K-Means en Python, utilisant uniquement **NumPy**. Parfaite pour apprendre le Machine Learning et les Design Patterns.

---

## ✨ Features

- ✅ **100% NumPy** : Aucune dépendance lourde (pas de scikit-learn, pandas, etc.)
- ✅ **API compatible scikit-learn** : `fit()`, `predict()`, `transform()`, `score()`
- ✅ **Deux stratégies d'initialisation** : `random` et `k-means++`
- ✅ **Vectorisé** : Performance optimisée grâce au broadcasting NumPy
- ✅ **Design Patterns** : Pattern Strategy pour l'initialisation, code modulaire et testable
- ✅ **Type hints** : Annotations de type pour une meilleure expérience IDE
- ✅ **Tests inclus** : Structure prête pour pytest

---

# Cloner le repo
git clone https://github.com/laguchoritarik/kmeans-from-scratch.git
cd kmeans-from-scratch

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer en mode développement
pip install -e .[dev]

# Exemple
import numpy as np
from kmeans_lib import KMeans

# Générer des données d'exemple
```python
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],   # Cluster 1
    np.random.randn(50, 2) + [5, 5]    # Cluster 2
])

# Créer et entraîner le modèle
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100)
kmeans.fit(X)

# Prédire les clusters
labels = kmeans.predict(X)
print(f"Labels: {labels[:10]}")  # Premier 10 prédictions

# Accéder aux centroïdes
print(f"Centroïdes:\n{kmeans.cluster_centers_}")

# Calculer l'inertie (qualité du clustering)
print(f"Inertie: {kmeans.inertia_:.2f}")
```python
