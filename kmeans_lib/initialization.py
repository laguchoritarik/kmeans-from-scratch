import numpy as np
from abc import ABC,abstractmethod
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
    def initialize(self,X, n_clusters):
        return super().initialize(n_clusters)
    