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
from utils.utils import validate_config_param_type, validate_list_of_strings
from visualization.plotting import plotter

class Clustering:
    """
    Handles clustering operations and evaluation metrics for microbiome data.
    """

    def __init__(self, config_path="config/clustering.yaml"):
        """
        Initializes the Clustering class with configuration, serializer, and plotter.
        
        Parameters
        ----------
            config_path (str): Path to the configuration file.
        """
        self.logger = logger
        self.config_path = config_path
        self.serializer = serializer
        self.plotter = plotter

        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading config from file {self.config_path}: {e}")

    def _bray_matrix(self, data):
        """
        Computes the Bray-Curtis distance matrix for the given data.
        """
        return squareform(pdist(data, metric='braycurtis'))


    def _k_medoids_fit(self, data, n_clusters, metric, random_state):
        """
        Fits K-Medoids clustering.
        """
        self.logger.info(f"Fitting K-Medoids with k={n_clusters}, metric={metric}")
        return KMedoids(n_clusters, metric, random_state=random_state).fit_predict(data)


    def _run_single_k(self, n_clusters, data, metric, random_state):
        """
        Runs K-Medoids for a single k value.
        """
        kmedoids = KMedoids(n_clusters, metric, random_state=random_state).fit(data)

        score = silhouette_score(data, kmedoids.labels_, metric=metric) if n_clusters > 1 else np.nan

        return kmedoids.inertia_, score


    def compute_elbow_silhouette(self, data: np.ndarray) -> tuple[tuple, tuple]:
        """
        Computes inertia and silhouette scores for a range of k values using K-Medoids.

        Parameters
        ----------
            data (np.ndarray): Data matrix or distance matrix.

        Returns
        ----------
            **tuple ((tuple, tuple))**
                - inertias: List of inertia values (sum of distances to medoids).
                - silhouettes: List of mean silhouette scores.
        """

        config = self.config.get('elbow_silhouette', {})

        max_clusters = config.get('max_clusters', 20)
        validate_config_param_type("max_clusters", max_clusters, int)
        
        if max_clusters < 2:
            self.logger.warning(f"Clustering for less than 2 clusters. Skipping Elbow and Silhouette analysis")
            return [], []

        distance = config.get('distance', 'euclidean')
        
        if distance not in ['euclidean', 'braycurtis', 'unifrac']:
            self.logger.error(
                f"Unsupported distance metric: {distance}. Supported metrics are 'euclidean', 'unifrac', 'braycurtis'"
            )
            raise ValueError()
        
        metric = distance if distance == 'euclidean' else 'precomputed'

        njobs = config.get('n_jobs', -1) 
        validate_config_param_type("n_jobs", njobs, int)
        if njobs < -1:
            self.logger.error(f"Njobs must be either -1 or a positive integer, got {njobs}")
            raise ValueError()
        
        random_state = config.get('random_state', 42)        
        validate_config_param_type("random_state", random_state, int)
        
        visualize = config.get('visualize', False)
        validate_config_param_type("visualize", visualize, bool)

        if distance == 'braycurtis':
            data = self._bray_matrix(data)

        self.logger.info(f"Computing Elbow and Silhouette scores up to k={max_clusters} using metric={metric}")

        cluster_range = range(1, max_clusters + 1)

        results = Parallel(n_jobs=njobs)(
            delayed(self._run_single_k)(cluster_idx, data, metric, random_state) for cluster_idx in cluster_range
        )

        inertias, silhouettes = zip(*results)

        tick_values = np.arange(2, max_clusters + 1, 2)

        self.plotter.plot_elbow_and_silhouette(cluster_range, list(inertias), list(silhouettes),
                                                tick_values, distance, visualize)
        return inertias, silhouettes


    def kmedoids(self, data: np.ndarray) -> pd.DataFrame:
        """
        Performs K-Medoids for different k values.

        Parameters
        ----------
            data (np.ndarray): Data matrix or distance matrix.
        Returns
        ----------
            cluster_df: DataFrame with counts of samples per cluster for each k.
        """

        config = self.config.get('kmedoids', {})

        k_values = config.get('k_values', [2])
        validate_config_param_type("k_values", k_values, list)

        for k in k_values:
            validate_config_param_type("k_s", k, int)
            if k < 1:
                self.logger.error(f"k must be a positive integer, got {k}")
                raise ValueError()
        
        metric = config.get('metric', 'euclidean')
        validate_config_param_type("metric", metric, str)
        if metric not in ['euclidean', 'braycurtis', 'unifrac']:
            self.logger.error(
                f"Unsupported distance metric: {metric}. Supported metrics are 'euclidean', 'unifrac', 'braycurtis'"
            )
            raise ValueError()

        random_state = config.get('random_state', 42)
        validate_config_param_type("random_state", random_state, int)

        save = config.get('save', True)
        validate_config_param_type("save", save, bool)

        save_format = config.get('save_format', 'csv')
        validate_config_param_type("save_format", save_format, str)
        if save_format not in ['csv', 'tsv', 'xlsx']:
            self.logger.error(f"Unsupported save format: {save_format}. Supported formats are 'csv', 'tsv', and 'xlsx'")
            raise ValueError()

        cluster_counts = {}

        for n_clusters in k_values:
            if metric == "braycurtis":
                data = pd.DataFrame(self._bray_matrix(data)).values


            labels = self._k_medoids_fit(data, n_clusters, metric, random_state)
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

    def metadata_analysis(self, data: np.ndarray, dataset: pd.DataFrame, meta_cols: list) -> pd.DataFrame:
        """
        Performs k-medoids clustering and analyzes categorical metadata associated with clusters.
        
        Parameters
        ----------
            data (np.ndarray): Scaled data matrix
            dataset (pd.DataFrame): DataFrame with metadata
            meta_cols (list): metadata columns list

        Returns:
            summary_df: DataFrame with metadata analysis results
        """

        config = self.config.get('metadata_analysis', {})

        n_clusters = config.get('n_clusters', 5)
        validate_config_param_type("n_clusters", n_clusters, int)
        if n_clusters < 1:
            self.logger.error(f"n_clusters must be a positive integer, got {n_clusters}")
            raise ValueError()

        metric = config.get('metric', 'euclidean')
        validate_config_param_type("metric", metric, str)
        if metric not in ['euclidean', 'braycurtis', 'unifrac']:
            self.logger.error(
                f"Unsupported distance metric: {metric}. Supported metrics are 'euclidean', 'unifrac', 'braycurtis'"
            )
            raise ValueError()

        random_state = config.get('random_state', 42)
        validate_config_param_type("random_state", random_state, int)

        save = config.get('save', True)
        validate_config_param_type("save", save, bool)

        save_format = config.get('save_format', 'csv')
        validate_config_param_type("save_format", save_format, str)
        if save_format not in ['csv', 'tsv', 'xlsx']:
            self.logger.error(f"Unsupported save format: {save_format}. Supported formats are 'csv', 'tsv', and 'xlsx'")
            raise ValueError()
        
        useless_metadata = config.get('useless_metadata', [])
        validate_list_of_strings(useless_metadata)

        visualize = config.get('visualize', True)
        validate_config_param_type("visualize", visualize, bool)

        labels = self._k_medoids_fit(data, n_clusters, metric, random_state)

        # Add cluster labels to dataset
        dataset = dataset.copy()
        dataset['cluster'] = labels

        self.logger.info("Filtering and cleaning categorical metadata columns")

        # Filter categorical metadata        
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
                    subfolder=f"metadata_analysis/k_{n_clusters}_contingencies",
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

        self.plotter.plot_contingencies(contingencies, cols, metric, visualize)

        summary_df = pd.DataFrame(summary_results)

        if save:
            self.serializer.save_file(
                data=summary_df,
                subfolder=f"{time_str}/metadata_analysis/k_{n_clusters}",
                exp_type="chi2_summary",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric,
                should_save_time=False
            )
        
        return summary_df
