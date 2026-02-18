from typing import Literal
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import yaml
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform, pdist
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

from utils.validators import validate_config
from logger.logger import logger




class DimensionalityReduction:
    """
    Handles dimensionality reduction operations.
    """

    def __init__(self, config_path = "config/dimensionality_reduction.yaml"):
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.logger = logger

    
    def PCA(self, data: np.ndarray):
        """
        Computes Principal Component Analysys after reading config file.
        
        Parameters
        ----------
            data: np.ndarray
                Data matrix.
        Returns
        ----------
            components: np.ndarray 
                PCA components
            var_ratio: np.ndarray
                Explained variance ratio    
            cum_var_ratio: np.ndarray
                Cumulative explained variance ratio
            pca: PCA
                PCA object (needed for clustering)
        """
        params = validate_config(self.config, 'pca')
        n_components = params['n_components']
        
        pca = PCA(n_components=n_components)

        self.logger.info(f"Computing PCA with n_components={n_components}")

        components = pca.fit_transform(data)

        var_ratio = pca.explained_variance_ratio_

        return components, var_ratio, np.cumsum(var_ratio), pca
    

    def KPCA(self, data: np.ndarray):
        """
        Computes Kernel PCA after reading config file.
        
        Parameters
        ----------
            data: np.ndarray
                Data matrix.
        Returns
        ----------
            components: np.ndarray 
                KPCA components
            per_comp: np.ndarray
                Variance per component    
            cum_per_comp: np.ndarray
                Cumulative variance per component
        """
        params = validate_config(self.config, 'kernelpca')
        n_components = params['n_components']
        kernel = params['kernel']
        gamma = params['gamma']

        kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
        
        self.logger.info(f"Computing KPCA with n_components={n_components}, kernel={kernel}, gamma={gamma}")

        components = kpca.fit_transform(data)
        per_comps = kpca.eigenvalues_ / kpca.eigenvalues_.sum()

        return components, per_comps, np.cumsum(per_comps)
    

    def PCOA(self, data: np.ndarray, distance: Literal['braycurtis', 'jensenshannon'] = None):
        """
        Computes Principal Coordinate Analysis after reading config file.
        
        Parameters
        ----------
            data: np.ndarray
                Data matrix.
            distance: Literal['braycurtis', 'jensenshannon'], default None
                Distance type. Overrides config file specidified distance, useful for testing.
        Returns
        ----------
            components: np.ndarray 
                PCOA components
            values: np.ndarray
                Components values 
            prop_expl: np.ndarray or pd.Series
                Proportions explained
            cum_prop_expl: np.ndarray or pd.Series
                Cumulative Proportions explained
        """
        params = validate_config(self.config, 'pcoa')
        distance = params['metric'] if distance is None else distance
        
        assert distance in ['braycurtis', 'jensenshannon'], \
            f"Distance must be either 'braycurtis' or 'jensenshannon', got {distance}"

        data = squareform(pdist(data, metric=distance))

        self.logger.info(f"Computing PCOA with distance={distance}")

        # Avoid printing negative eigenvalue warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            components = pcoa(DistanceMatrix(data))

        prop_expl = components.proportion_explained

        return components, components.samples.values, prop_expl, np.cumsum(prop_expl)
    

    def TSNE(self, data: np.ndarray):
        """
        Computes T-Stochastic Neighbour Embedding after reading config file.
        
        Parameters
        ----------
            data: np.ndarray
                Data matrix.
        Returns
        ----------
            results: np.ndarray 
                Fitted TSNE
        """
        params = validate_config(self.config, 'tsne')

        n_components = params['n_components']
        perplexity = params['perplexity']
        learning_rate = params['learning_rate']
        random_state = params['random_state']
        max_iter = params['max_iter']
        metric = params['metric']

        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                    random_state=random_state, max_iter=max_iter, metric=metric)
        
        self.logger.info(f"Computing TSNE with n_components={n_components}, perplexity={perplexity}, "
                         f"learning_rate={learning_rate}, random_state={random_state}, "
                         f"max_iter={max_iter}, metric={metric}")

        return tsne.fit_transform(data)


    def PLS_DA(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray):
        """
        Computes Partial Least Squares Discriminant Analysis after reading config file.
        
        Parameters
        ----------
            data: np.ndarray
                Data matrix.
            train_labels: np.ndarray
                Train labels (y) to fit the model.
            test_data: np.ndarray
                Test data on which to transform.
        Returns
        ----------
            results: np.ndarray 
                Transformed PLS data.
        """
        params = validate_config(self.config, 'pls_regression')
        n_components = params['n_components']

        pls = PLSRegression(n_components=n_components)

        self.logger.info(f"Computing PLS with n_components={n_components}")

        pls.fit(train_data, train_labels)

        return pls.transform(test_data)


    def LDA(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray,):
        """
        Computes Linear Discriminant Analysis after reading config file.

        Parameters
        ----------
            train_data: np.ndarray
                Train data matrix.
            test_data: np.ndarray
                Test Data matrix.
            train_labels: np.ndarray
                Labels (y) to fit the model.
        Returns
        ----------
            results: np.ndarray 
                Transformed and flattened LDA data.
        """

        lda = LinearDiscriminantAnalysis()

        self.logger.info(f"Computing LDA")

        return lda.fit(train_data, train_labels).transform(test_data).flatten()

        
