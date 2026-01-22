import yaml
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

class Clustering:
    """
    Handles clustering operations and evaluation metrics for microbiome data.
    """
    def __init__(self, config_path="Config/clustering.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.metric = self.config.get('metric', 'euclidean')
        self.max_clusters = self.config.get('max_clusters', 20)
        self.njobs = self.config.get('n_jobs', -1)


    def compute_elbow_silhouette(self, X):
        """
        Computes inertia and silhouette scores for a range of k values using K-Medoids.
        
        Args:
            X: Data matrix or distance matrix.
            max_clusters: Maximum number of clusters to test.
            metric: Distance metric ('euclidean', 'precomputed', etc.).
            
        Returns:
            ks: Range of k values tested.
            inertias: List of inertia values (sum of distances to medoids).
            silhouettes: List of mean silhouette scores.
            tick_values: Suggested tick marks for plotting.
        """
        inertias = []
        silhouettes = []

        def run_single_k(k, X, metric):
            # KMedoids fit
            km = KMedoids(n_clusters=k, random_state=42, metric=metric).fit(X)
            inertia = km.inertia_
            
            # Silhouette (only for k > 1)
            score = np.nan
            if k > 1:
                score = silhouette_score(X, km.labels_, metric=metric)
            return inertia, score

        # Running the loop in parallel
        results = Parallel(n_jobs=self.njobs)(
            delayed(run_single_k)(k, X, self.metric) for k in range(1, self.max_clusters + 1)
        )
        
        ks = range(1, self.max_clusters + 1)
        inertias, silhouettes = zip(*results)
        tick_values = np.arange(2, self.max_clusters + 1, 2)

        return ks, list(inertias), list(silhouettes), tick_values, self.metric
