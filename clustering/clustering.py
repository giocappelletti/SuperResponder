from typing import Literal
import yaml
import warnings

import numpy as np
import pandas as pd

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, adjusted_rand_score, fowlkes_mallows_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
from sklearn.model_selection import ShuffleSplit
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import chi2_contingency

from joblib import Parallel, delayed

from logger.logger import logger
from fileio.serialization import serializer
from utils.validators import validate_config
from visualization.plotting import plotter
from fileio.df_loader import DataLoader


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


    def _k_medoids_fit(self, data, n_clusters, metric, random_state):
        """
        Fits K-Medoids clustering.
        """
        self.logger.info(f"Fitting K-Medoids with k={n_clusters}, metric={metric}")
        kmedoids = KMedoids(n_clusters, metric, random_state=random_state)
        labels = kmedoids.fit_predict(data)
        return kmedoids, labels


    def _run_single_k(self, n_clusters, data, metric, random_state, silhouette=False):
        """
        Runs K-Medoids for a single k value.
        """
        kmedoids = KMedoids(n_clusters, metric, random_state=random_state).fit(data)

        if silhouette:
            score = silhouette_score(data, kmedoids.labels_, metric=metric) if n_clusters > 1 else np.nan
            return kmedoids.inertia_, score
        
        return kmedoids.medoid_indices_
    

    def _analysis(self, data: np.ndarray, dataset: pd.DataFrame, resp_analisys=False):
        """
        Read analysis data from file and computes k medoids.
        """
        params = validate_config(self.config, 'analysis')
        
        n_clusters = params['n_clusters']
        metric = params['metric']
        random_state = params['random_state']
        save = params['save']
        save_format = params['save_format']
        useless_metadata = params['useless_metadata']
        visualize = params['visualize']

        distance = metric if metric == 'euclidean' else 'precomputed'

        if metric == "braycurtis":
            data = squareform(pdist(data, metric='braycurtis'))


        _, labels = self._k_medoids_fit(data, n_clusters, distance, random_state)

        # Add cluster labels to dataset
        dataset = dataset.copy()
        dataset['cluster'] = labels
        
        return dataset, n_clusters, distance, save, save_format, useless_metadata, visualize


    def _compute_contingency(self, dataset: pd.DataFrame, col: str):
        
        contingency = pd.crosstab(dataset['cluster'], dataset[col])

        chi2, p, _, _ = chi2_contingency(contingency)

        contingency_prop  = contingency / contingency.sum().sum()

        return contingency_prop, chi2, p


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

        params = validate_config(self.config, 'elbow_silhouette')
        distance = params['metric']
        max_clusters = params['n_clusters']
        njobs = params['njobs']
        random_state = params['random_state']
        visualize = params['visualize']
        save = params['save']


        metric = distance if distance == 'euclidean' else 'precomputed'

        if distance == 'braycurtis':
            data = squareform(pdist(data, metric='braycurtis'))

        self.logger.info(f"Computing Elbow and Silhouette scores up to k={max_clusters} using metric={metric}")

        cluster_range = range(1, max_clusters + 1)

        results = Parallel(n_jobs=njobs)(
            delayed(self._run_single_k)(cluster_idx, data, metric, random_state, silhouette=True) for cluster_idx in cluster_range
        )

        inertias, silhouettes = zip(*results)

        tick_values = np.arange(2, max_clusters + 1, 2)

        self.plotter._plot_elbow_and_silhouette(cluster_range, list(inertias), list(silhouettes),
                                                tick_values, distance, visualize, save)
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

        params = validate_config(self.config, 'kmedoids')
        k_values = params['k_values']
        metric = params['metric']
        random_state = params['random_state']
        save = params['save']
        save_format = params['save_format']

        cluster_counts = {}

        working_data = data
        working_metric = metric
        if metric == "braycurtis":
            working_data = squareform(pdist(data, metric='braycurtis'))
            working_metric = 'precomputed'
        elif metric == 'unifrac':
            working_metric = 'precomputed'

        for n_clusters in k_values:
            _, labels = self._k_medoids_fit(working_data, n_clusters, working_metric, random_state)
            counts = pd.Series(labels).value_counts().sort_index()
            cluster_counts[n_clusters] = counts

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
        else:
            self.logger.info(f"Kmedoids groups: {cluster_df}")

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

        dataset, n_clusters, metric, save, save_format, useless_metadata, \
            visualize = self._analysis(data, dataset)


        if useless_metadata is None:
            self.logger.info("Using all metedata columns") 
            categorical_meta = meta_cols
        
        else:
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
            
            contingency, chi2, p = self._compute_contingency(dataset, col)
           
            if save:
                time_str = self.serializer.save_file(
                    data=contingency,
                    subfolder=f"metadata_analysis/k_{n_clusters}_contingencies",
                    exp_type=f"{col}_contingency",
                    exp_group="clustering",
                    save_format=save_format,
                    distance_type=metric
                )

            # chi-squared test
            summary_results.append({"metadata": col, "chi2": chi2, "p_value": p})

            # cluster proportion plot
            contingencies.append(contingency)
            cols.append(col)

        self.plotter._plot_contingencies(contingencies, cols, metric, visualize, save)

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


    def response_analysis(self, 
                          data_matrix: np.ndarray,
                          dataset: str | pd.DataFrame, 
                          orig_dataset: str | pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
        """
        Performs k-medoids clustering and analyzes "response" label associated with clusters.
        
        Parameters
        ----------
            data_matrix (np.ndarray): Data matrix
            dataset (pd.DataFrame): DataFrame with metadata
            orig_dataset: Path to raw dataset or pre_loaded dataframe

        Returns
        ----------
            tuple (contingency, chi2, p)
                """

        if isinstance(orig_dataset, str):
            orig_dataset = DataLoader().load_dataset(orig_dataset, drop_response=False, sanitize=False)

        if isinstance(dataset, str):
            dataset = DataLoader().load_dataset(dataset, drop_response=False, sanitize=False, index_col=0)

        dataset, _, distance, save, save_format, _, \
            visualize = self._analysis(data_matrix, dataset, resp_analisys=True)
        
        n_clusters = 2

        assert "response" in orig_dataset.columns, \
            "Response column not found in raw dataset!"

        dataset['response'] = orig_dataset['response'].loc[dataset.index]
        
        contingency, chi2, p = self._compute_contingency(dataset, "response")

        if save:
            self.serializer.save_file(
                data=contingency,
                subfolder=f"response_analysis/k_{n_clusters}_contingencies",
                exp_type=f"response_contingency",
                exp_group="clustering",
                save_format=save_format,
                distance_type=distance
            )
        
        self.logger.info(f"Response chi2: {chi2}, p-value: {p}")

        self.plotter._plot_contingencies([contingency], ["response"], distance, visualize, save)

        return contingency, chi2, p


    def modenesi_analysis(self, data: np.ndarray, ref_data: np.ndarray, unifrac_dataframe: pd.DataFrame = None,
                          orig_dataframe: pd.DataFrame = None, modenesi_dataframe: pd.DataFrame = None) -> pd.DataFrame:

        params = validate_config(self.config, 'modenesi')
        k_values = params['k_values']
        metric = params['metric']
        random_state = params['random_state']
        save = params['save']
        save_format = params['save_format']
        njobs = params['njobs']
        
        distance = metric 

        if distance == "unifrac":
            assert unifrac_dataframe is not None, \
                "A Unifrac dataset must be passed to the function if Unifrac distance is used"
            assert orig_dataframe is not None, \
                "orig_dataframe must be passed to the function if Unifrac distance is used"
            assert modenesi_dataframe is not None, \
                "modenesi_dataframe must be passed to the function if Unifrac distance is used"

        if unifrac_dataframe is not None and distance != "unifrac":
            self.logger.info("Unifrac Dataset was passed to the function, assuming Unifrac distance type")
            distance = 'unifrac'

        fitting_data = ref_data
        
        if distance == "braycurtis":
            fitting_data = squareform(pdist(ref_data, metric='braycurtis'))
            metric = 'precomputed'

        elif distance == 'unifrac':
            fitting_data = data
            metric = 'precomputed'

        cluster_distributions = {}

        self.logger.info(f"Computing medoids for k ={k_values} using metric={metric}")

        medoid_indices = Parallel(n_jobs=njobs)(
            delayed(self._run_single_k)(cluster_idx, fitting_data, metric, random_state) 
            for cluster_idx in k_values
        )

        for i, k in enumerate(k_values):
            current_medoid_indices = medoid_indices[i]
            
            if distance in ['euclidean', 'braycurtis']:
                medoid_points = ref_data[current_medoid_indices]
                dist_to_medoids = cdist(data, medoid_points, metric=distance)
            
            else:
                medoid_samples = orig_dataframe.index[current_medoid_indices].tolist()
                dist_to_medoids = unifrac_dataframe.loc[modenesi_dataframe.index, medoid_samples].to_numpy()

            modenesi_clusters = np.argmin(dist_to_medoids, axis=1)
            cluster_distributions[k] = pd.Series(modenesi_clusters).value_counts().sort_index()

        cluster_df = pd.DataFrame(cluster_distributions).fillna(0).astype(int)

        cluster_df.index.name = "cluster_idx"
        cluster_df = cluster_df.rename(columns={k: f"k_{k}" for k in cluster_df.columns})

        if save:
            self.serializer.save_file(
                data=cluster_df,
                subfolder="modenesi",
                exp_type="modenesi",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )
        else:
            self.logger.info(f"Modenesi groups: {cluster_df}")

        return cluster_df


    def pca_mds(self, data: np.ndarray, pca_type: Literal['pca', 'pcoa', 'kpca']):
        """
        Compute PCA with euclidean distance.

        Parameters
        ----------
            data (np.ndarray): Data matrix.
            pca_type (str): Type of PCA to use.
        Returns
        ----------
            fitted_pca: PCA object.
            medoids: List of found medoids.
        """
        assert pca_type in ['pca', 'pcoa', 'kpca'], \
            f"pca_type must be either 'pca', 'pcoa' or 'kpca', got {pca_type}"

        params = validate_config(self.config, 'pca_mds')
        
        n_components = params['n_components']
        k_values = params['k_values']
        metric = params['metric']
        random_state = params['random_state']
        visualize = params['visualize']
        save = params['save']
        kernel = params['kernel']
        gamma = params['gamma']

        distance = metric if metric == 'euclidean' else 'precomputed'

        fitted = None

        if distance == "euclidean":
            assert pca_type in ['pca', 'pcoa'], \
                f"pca_type must be either 'pca' or 'pcoa', got {pca_type}"

            if pca_type == 'pca':
                pca = PCA(n_components=n_components, random_state=random_state)
                fitted = pca.fit_transform(data)
            
            elif pca_type == 'kpca':
                fitted = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma).fit_transform(data)
            
            else:
                self.logger.error(f"With distance {metric} pca_type must be either 'pca' or 'kpca', got {pca_type}")


        else:
            if metric == 'braycurtis':
                data = squareform(pdist(data, metric=metric))
            
            if pca_type == 'pca':
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    fitted = MDS(n_components=n_components, metric="precomputed", random_state=42, n_init=4).fit_transform(data)
            
            elif pca_type == 'pcoa':
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    fitted = pcoa(DistanceMatrix(data)).samples.iloc[:, :n_components].values

            else:
                self.logger.error(f"With distance {metric} pca_type must be either 'pca' or 'pcoa', got {pca_type}")


        assert fitted is not None, \
            f"Cannot compute results, check config file for possible problems.\n \
            Using distance {distance} with pca_type {pca_type}"            

        medoids = []
        all_labels = []

        for k in k_values:
            kmedoids, labels = self._k_medoids_fit(data, k, distance, random_state)
            
            if distance == 'euclidean' and pca_type == 'pca':
                medoids.append(pca.transform(data[kmedoids.medoid_indices_]))
            
            else:
                medoids.append(fitted[kmedoids.medoid_indices_])
                
            all_labels.append(labels)
        
        self.plotter._plot_pca_mds(k_values, fitted, medoids, all_labels, metric, visualize, pca_type, n_components, save)
        
        return fitted, medoids
    

    def cluster_stability_ari_fm(self, data: np.ndarray):
        """
        Performs k-medoids on n_splits folds and calculates cluster stability with ARI and Fowlkes-Mallows.

        Generates separate heatmaps for ARI and FM.

        Params
        ----------
            data: data array or distance matrix (if metric='precomputed')
            k: number of clusters
            n_splits: number of folds
            test_size: size of the test fold
            metric: 'Euclidean' or 'precomputed'
            output_dir: output base directory
            distance_name: name of the distance
        Returns
        ----------
            None    
        """

        params = validate_config(self.config, 'stability_ari')

        n_clusters = params['n_clusters']
        metric = params['metric']
        n_splits = params['n_splits']
        test_size = params['test_size']
        random_state = params['random_state']
        save = params['save']
        save_format = params['save_format']
        visualize = params['visualize']

        distance = metric if metric == 'euclidean' else 'precomputed'

        if metric == 'braycurtis':
            data = squareform(pdist(data, metric='braycurtis'))

        n_samples = data.shape[0]

        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        all_labels = []

        for fold, (train_idx, _) in enumerate(splitter.split(data)):
            
            train_data = data[train_idx] if distance == 'euclidean' else data[np.ix_(train_idx, train_idx)]

            indices = self._run_single_k(n_clusters, train_data, distance, random_state=fold)

            if distance == 'euclidean':
                medoids = train_data[indices]
                labels_complete = [np.argmin(np.linalg.norm(medoids - data[i], axis=1)) for i in range(n_samples)]

            elif distance == 'precomputed':
                labels_complete = [np.argmin(data[i, train_idx][indices]) for i in range(n_samples)]
            
            all_labels.append(np.array(labels_complete))

        ari_results = []
        fm_results = []

        for i in range(n_splits):
            for j in range(i+1, n_splits):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                fm = fowlkes_mallows_score(all_labels[i], all_labels[j])
                ari_results.append((i, j, ari))
                fm_results.append((i, j, fm))

        ari_matrix = np.zeros((n_splits, n_splits))
        fm_matrix = np.zeros((n_splits, n_splits))

        for i in range(n_splits):
            for j in range(n_splits):
                ari_matrix[i, j] = adjusted_rand_score(all_labels[i], all_labels[j])
                fm_matrix[i, j] = fowlkes_mallows_score(all_labels[i], all_labels[j])

        ari_results = pd.DataFrame(ari_results, columns=["Fold_1", "Fold_2", "ARI"])
        fm_results = pd.DataFrame(fm_results, columns=["Fold_1", "Fold_2", "FM"])
        ari_matrix = pd.DataFrame(ari_matrix)
        fm_matrix = pd.DataFrame(fm_matrix)

        plotter._plot_stability_ari([(ari_matrix, "ARI"), (fm_matrix, "Fowlkes-Mallows")], metric, n_clusters, visualize)

        if save:
            self.serializer.save_file(
                data=ari_results,
                subfolder="stability",
                exp_type="ari_pairwise",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )
            self.serializer.save_file(
                data=fm_results,
                subfolder="stability",
                exp_type="fm_pairwise",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )
            self.serializer.save_file(
                data=ari_matrix,
                subfolder="stability",
                exp_type="ari_matrix",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )
            self.serializer.save_file(
                data=fm_matrix,
                subfolder="stability",
                exp_type="fm_matrix",
                exp_group="clustering",
                save_format=save_format,
                distance_type=metric
            )

        return ari_results, fm_results, ari_matrix, fm_matrix