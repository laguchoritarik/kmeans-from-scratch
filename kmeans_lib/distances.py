import numpy as np

def compute_distances(X, centers):
    """
    Compute Euclidean distances between each sample in X and each center.
    
    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        The data points.
    centers : np.ndarray, shape (n_clusters, n_features)
        The cluster centers.
    
    Returns:
    --------
    distances : np.ndarray, shape (n_samples, n_clusters)
        Distance matrix where distances[i, j] is the distance between
        X[i] and centers[j].
    """
    # TODO: Implémente ici avec NumPy (sans boucle for)
    return np.sqrt(np.sum((X[:,np.newaxis,:]-centers[np.newaxis:,:])**2,axis=2))