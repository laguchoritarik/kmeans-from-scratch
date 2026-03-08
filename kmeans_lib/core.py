from .initialization import RandomInit,KMeanPlusPlusInit
from .distances import compute_distances
import numpy as np
class KMeans:
    def __init__(self,n_clusters=8,init='random',max_iter=300,tol=1e-4):
        self.n_clusters=n_clusters
        if(init=='random'):
            self.init_strategy_=RandomInit()
        elif(init=='k-means++'):
            self.init_strategy_=KMeanPlusPlusInit()
        else:
            raise ValueError(f"init method '{init}' not supported")
       # self.init=init
        self.max_iter=max_iter
        self.tol=tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    def fit(self, X):
            n_samples, n_features = X.shape
            
            # ─────────────────────────────────────────────────────
            # ÉTAPE 1 : Initialiser les centroïdes (déjà fait)
            # ─────────────────────────────────────────────────────
            self.cluster_centers_ = self.init_strategy_.initialize(X, self.n_clusters)
            
            # ─────────────────────────────────────────────────────
            # ÉTAPE 2 : Boucle d'entraînement
            # ─────────────────────────────────────────────────────
            for i in range(self.max_iter):
                # 2a. ASSIGNATION : Calculer les distances et assigner les clusters
                distances = compute_distances(X, self.cluster_centers_)
                self.labels_ = np.argmin(distances, axis=1)  # Cluster le plus proche
                
                # 2b. MISE À JOUR : Recalculer les centroïdes
                new_centers = np.zeros_like(self.cluster_centers_)
                for k in range(self.n_clusters):
                    # Points assignés au cluster k
                    points_in_cluster = X[self.labels_ == k]
                    
                    # Si le cluster est vide, garder l'ancien centroïde
                    if len(points_in_cluster) == 0:
                        new_centers[k] = self.cluster_centers_[k]
                    else:
                        # Nouveau centroïde = moyenne des points
                        new_centers[k] = np.mean(points_in_cluster, axis=0)
                
                # 2c. VÉRIFIER LA CONVERGENCE
                # Calculer le déplacement des centroïdes
                center_shift = np.sqrt(np.sum((new_centers - self.cluster_centers_) ** 2))
                
                # Mettre à jour les centroïdes
                self.cluster_centers_ = new_centers
                self.n_iter_ = i + 1
                
                # Si le déplacement est inférieur à la tolérance → convergence
                if center_shift < self.tol:
                    break
                
            # ─────────────────────────────────────────────────────
            # ÉTAPE 3 : Calculer l'inertie (somme des distances au carré)
            # ─────────────────────────────────────────────────────
            distances = compute_distances(X, self.cluster_centers_)
            self.inertia_ = np.sum(distances[np.arange(n_samples), self.labels_] ** 2)
            
            return self
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            New data to predict.
        
        Returns:
        --------
        labels : np.ndarray, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # Vérifier que le modèle est entraîné
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() before predict().")
        
        from kmeans_lib.distances import compute_distances
        
        # Calculer les distances aux centroïdes
        distances = compute_distances(X, self.cluster_centers_)
        
        # Retourner l'index du centroïde le plus proche
        return np.argmin(distances, axis=1)


    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            New data to transform.
        
        Returns:
        --------
        distances : np.ndarray, shape (n_samples, n_clusters)
            Distance of each sample to each cluster center.
        """
        # Vérifier que le modèle est entraîné
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted. Call fit() before transform().")
        
        from kmeans_lib.distances import compute_distances
        
        # Retourner la matrice des distances
        return compute_distances(X, self.cluster_centers_)
    def score(self, X):
        """
        Opposite of the value of X on the K-means objective.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Test instances.
        
        Returns:
        --------
        score : float
            Negative inertia (lower is better).
        """
        from kmeans_lib.distances import compute_distances
        
        labels = self.predict(X)
        distances = compute_distances(X, self.cluster_centers_)
        inertia = np.sum(distances[np.arange(len(X)), labels] ** 2)
        
        return -inertia  # Négatif car scikit-learn utilise un score à maximiser

