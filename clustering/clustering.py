import yaml
import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency

from joblib import Parallel, delayed

from logger.logger import logger
from utils.serialization import serializer
from visualization.plotting import plotter

class Clustering:
    """
    Handles clustering operations and evaluation metrics for microbiome data.
    """


    def __init__(self, config_path="config/clustering.yaml"):

        self.logger = logger
        self.config_path = config_path
        self.serializer = serializer
        self.plotter = plotter

        try:
            with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")

    def _bray_matrix(self, X):
        """
        Computes the Bray-Curtis distance matrix for the given data.
        Args:
            X: Scaled data matrix.
        Returns:
            bray_matrix: Bray-Curtis distance matrix.
        """
        bray_dist = pdist(X, metric='braycurtis')
        return squareform(bray_dist)

    def _k_medoids_fit(self, X, n_clusters, metric, random_state):
        """
        Fits K-Medoids clustering.
        Args:
            X: Data matrix or distance matrix.
            n_clusters: Number of clusters.
            metric: Distance metric ('euclidean', 'precomputed', etc.).
            random_state: Random state for reproducibility.
        Returns:
            kmedoids: Fitted KMedoids object.
        """
        kmedoids = KMedoids(n_clusters=n_clusters, metric=metric, random_state=random_state)
        self.logger.info(f"Fitting K-Medoids with k={n_clusters}, metric={metric}")
        return kmedoids.fit_predict(X)


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

        config = self.config.get('elbow_silhouette', {})

        max_clusters = config.get('max_clusters', 20)
        distance = config.get('distance', 'euclidean')
        metric = distance if distance == 'euclidean' else 'precomputed'
        njobs = config.get('n_jobs', -1)
        random_state = config.get('random_state', 42)
        visualize = config.get('visualize', False)

        inertias = []
        silhouettes = []

        if distance == 'braycurtis':
            X = self._bray_matrix(X)

        def run_single_k(k, X, metric):
            # KMedoids fit
            km = KMedoids(n_clusters=k, random_state=random_state, metric=metric).fit(X)
            inertia = km.inertia_

            # Silhouette (only for k > 1)
            score = np.nan
            if k > 1:
                score = silhouette_score(X, km.labels_, metric=metric)
            return inertia, score

        self.logger.info(f"Computing Elbow and Silhouette scores up to k={max_clusters} using metric={metric}")
        # Running the loop in parallel
        results = Parallel(n_jobs=njobs)(
            delayed(run_single_k)(k, X, metric) for k in range(1, max_clusters + 1)
        )

        ks = range(1, max_clusters + 1)
        inertias, silhouettes = zip(*results)
        tick_values = np.arange(2, max_clusters + 1, 2)

        self.plotter.plot_elbow_and_silhouette(ks, list(inertias), list(silhouettes),
                                                tick_values, distance, visualize)

        return inertias, silhouettes

    def kmedoids(self, X):
        """
        Performs K-Medoids for different k values.
        Args:
            X: Data matrix or distance matrix.
        Returns:
            cluster_df: DataFrame with counts of samples per cluster for each k.
        """

        config = self.config.get('kmedoids', {})

        k_values = config.get('k_values', [2])
        metric = config.get('metric', 'euclidean')
        random_state = config.get('random_state', 42)
        save = config.get('save', True)
        save_format = config.get('save_format', 'csv')

        cluster_counts = {}

        for k in k_values:
            if metric == "braycurtis":
                X = pd.DataFrame(self._bray_matrix(X)).values

            labels = self._k_medoids_fit(X, k, metric, random_state)
            counts = pd.Series(labels).value_counts().sort_index()
            cluster_counts[k] = counts

        cluster_df = pd.DataFrame(cluster_counts).fillna(0).astype(int)

        cluster_df.index.name = "cluster_idx"
        cluster_df = cluster_df.rename(columns={k: f"k_{k}" for k in cluster_df.columns})

        if save:
            self.serializer.save_file(
                data=cluster_df,
                subfolder="kmedoids",
                exp_type="kmedoids",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )

        return cluster_df

    def metadata_analysis(self, X, dataset, meta_cols):
        """
        Performs k-medoids clustering and analyzes categorical metadata associated with clusters.
        Args:
        - X: Scaled data matrix
        - dataset: DataFrame with metadata
        - meta_cols: metadata columns list
        Returns:
        - summary_df: DataFrame with metadata analysis results
        """

        config = self.config.get('metadata_analysis', {})

        k = config.get('k', 5)
        metric = config.get('metric', 'euclidean')
        random_state = config.get('random_state', 42)
        save = config.get('save', True)
        save_format = config.get('save_format', 'csv')
        useless_metadata = config.get('useless_metadata', [])
        visualize = config.get('visualize', True)

        labels = self._k_medoids_fit(X, k, metric, random_state)

        # Add cluster labels to dataset
        dataset = dataset.copy()
        dataset['cluster'] = labels

        self.logger.info("Filtering and cleaning categorical metadata columns")
        # Filter categorical metadata columns
        
        categorical_meta = [
            col for col in meta_cols if col not in useless_metadata
        ]

        # Clean categorical metadata
        for col in categorical_meta:
            dataset[col] = dataset[col].astype(str).str.strip().str.capitalize()

        # chi-quadro analysis and plots
        summary_results = []
        contingencies = []
        cols = []
        for col in categorical_meta:
            contingency = pd.crosstab(dataset['cluster'], dataset[col])
            if save:
                time_str = self.serializer.save_file(
                    data=contingency,
                    subfolder=f"metadata_analysis/k_{k}_contingencies",
                    exp_type=f"{col}_contingency",
                    exp_group="clustering",
                    save_format=save_format,
                    distance_type=metric
                )

            self.logger.info(f"Performing chi-squared test for metadata column: {col}")
            # chi-squared test
            chi2, p, _, _ = chi2_contingency(contingency)
            summary_results.append({"metadata": col, "chi2": chi2, "p_value": p})

            # cluster proportion plot
            contingency_prop = contingency / contingency.sum().sum()
            contingencies.append(contingency_prop)
            cols.append(col)


        self.plotter.plot_contingency(contingencies, cols, metric, visualize)

        # Salva tabella riassuntiva chi2
        summary_df = pd.DataFrame(summary_results)

        if save:
            self.serializer.save_file(
                data=summary_df,
                subfolder=f"{time_str}/metadata_analysis/k_{k}",
                exp_type="chi2_summary",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric,
                should_save_time=False
            )
        
        return summary_df
