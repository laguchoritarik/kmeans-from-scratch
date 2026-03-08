import numpy as np
from abc import ABC,abstractmethod
from .distances import compute_distances
class InitializationStrategy(ABC):
    @abstractmethod
    def initialize(self,X:np.ndarray,n_clusters:int)->np.ndarray:
        pass
class RandomInit(InitializationStrategy):
    def initialize(self,X, n_clusters):
        np.random.seed(25)
        if(n_clusters>len(X)):
            raise ValueError("n_clusters cannot be greater than number of samples")
        indices=np.random.choice(len(X),size=n_clusters,replace=False)
        return X[indices]
class KMeanPlusPlusInit(InitializationStrategy):
    def initialize(self,X, n_clusters)->np.ndarray:
        n_samples,n_features=X.shape
        center_indices=[]
        center_indices.append(np.random.randint(0,len(X)))
        for _ in range(1,n_clusters):
            distances=compute_distances(X,X[center_indices])
            min_distances=np.min(distances,axis=1)
            probabilities=min_distances**2
            probabilities=probabilities/np.sum(probabilities)
            center_indices.append(np.random.choice(n_samples,p=probabilities))
        return X[center_indices]
    