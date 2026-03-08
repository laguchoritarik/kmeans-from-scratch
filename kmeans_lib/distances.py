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
    n_samples,n_features=X.shape
    n_centers,_=centers.shape
    distances=np.zeros((len(X),len(centers)))
    for i in range(n_samples):
        for j in range(n_centers):
            d=np.dot(X[i]-centers[j],(X[i]-centers[j]).T)
            #d=np.sqrt(d)
            distances[i,j]=d
    distances=np.sqrt(distances)
    return distances