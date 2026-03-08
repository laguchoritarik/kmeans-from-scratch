class KMeans:
    def __init__(self,n_clusters=8,init='random',max_iter=300,tol=1e-4):
        self.n_clusters=n_clusters
        self.init_=init
        self.max_iter=max_iter
        self.tol=tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    def fit(self, X):
        """
        Compute k-means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster.
        """
        # Logic will be implemented in Step 3 & 4
        pass
        return self
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.
        """
        # Logic will be implemented in Step 5
        pass
    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to transform.
        """
        # Logic will be implemented in Step 5
        pass

