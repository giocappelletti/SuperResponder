import warnings
from shap import summary_plot
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Literal
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from logger import logger

class Plotter:
    """
    Handles the visualization of clustering evaluation metrics with metric-specific labeling.

    Parameters
    ----------
        serializer: Serializer
            The Serializer instance used to save plots.
    """

    def __init__(self, serializer):
        self.logger = logger
        self.serializer = serializer


    def _compute_rows_cols(self, num_values):
        """
        Computes the number of rows and columns for subplots.
        """        

        # Max 3 columns is usually better for readability with legends
        ncols = max(1, min(3, num_values))
        nrows = max(1, (num_values + ncols - 1) // ncols)

        return nrows, ncols


    def _init_plot(self, values, title, n_components = None):
        """
        Initialize plot by computing rows and columns and setting up subplots.
        """
        num_values = len(values)

        nrows, ncols = self._compute_rows_cols(num_values)

        # Determine if 3D projection is needed
        projection = '3d' if n_components == 3 else None

        # Close all existing figures to prevent implicit figure creation issues
        # when plt.show() is called repeatedly in a loop.
        plt.close('all')
        
        fig, axes = plt.subplots(nrows, 
                                 ncols, 
                                 figsize = (ncols * 8, nrows * 5),
                                 squeeze = False, 
                                 subplot_kw = {'projection': projection})
        
        fig.canvas.manager.set_window_title(title)
      
        return fig, axes.flatten(), num_values
    

    def _close_plot(self, visualize, save, metric_name, subfolder, exp_name, fig, tight_layout = True):
        """
        Sets tight layout and closes the current plot.
        """
        if tight_layout:
            fig.tight_layout(h_pad = 2, w_pad = 2)

        if visualize:
            self.logger.info("Displaying plot")
            plt.show()
        
        if save:
            self.serializer.save_plot(fig, subfolder, exp_name, metric_name, bbox_inches = 'tight') 
            
        plt.close(fig)


    def _plot_elbow_and_silhouette(self, 
                                   ks, 
                                   inertias, 
                                   silhouettes, 
                                   tick_values, 
                                   metric_name, 
                                   visualize = True, 
                                   save = False):
        """
        Plots Elbow and Silhouette results with the specific metric name integrated,
        using _init_plot for consistent subplot initialization.
        """
        
        # Initialize subplots using _init_plot
        fig, axes, _ = self._init_plot([None, None], f"Clustering Analysis")

        for i in range(len(axes)):
            axes[i].grid(True, linestyle = "--", alpha = 0.6)
            axes[i].set_xticks(tick_values)
            axes[i].set_xlabel("Number of clusters (k)")

        # Elbow Plot
        axes[0].plot(ks, inertias, marker = 'o', color = 'royalblue', linewidth = 2)
        axes[0].set_ylabel("Inertia")
        axes[0].set_title(f"Elbow Method\nDistance: {metric_name}") 

        # Silhouette Plot
        axes[1].plot(ks, silhouettes, marker = 's', color = 'forestgreen', linewidth = 2)
        axes[1].set_ylabel("Average Silhouette Score")
        axes[1].set_title(f"Silhouette Method\nDistance: {metric_name}")

        self._close_plot(visualize, save, metric_name, "clustering", "elbow_silhouette", fig)
            

    def _plot_contingencies(self, contingencies, cols, distance_name, visualize = True, save = False):
        """
        Plots computed contingencies.
        """

        fig, axes, num_contingencies = self._init_plot(contingencies, f"Metadata Distribution - {distance_name}")
        
        for i in range(num_contingencies):
            col = cols[i]
            ax = axes[i]
            contingency_prop = contingencies[i]
            
            num_categories = len(contingency_prop.columns)
            show_legend = num_categories <= 15 
            
            contingency_prop.plot(kind = 'bar', stacked = True, ax = ax, legend = show_legend) 
            
            ax.set_title(f"{col} Distribution\n(Distance: {distance_name})", fontsize = 12)
            ax.set_ylabel("Proportion on total")
            ax.set_xlabel("Cluster")
            
            ax.tick_params(axis = 'x', rotation = 45)

            if show_legend:
                ax.legend(title = col, loc = 'upper left', bbox_to_anchor = (1, 1), fontsize = 'small')
            else:
                ax.set_title(f"{col} (Legend hidden - {num_categories} cats)\nDistance: {distance_name}", color = 'red')
        
        for j in range(num_contingencies, len(axes)):
            fig.delaxes(axes[j])

        self._close_plot(visualize, save, distance_name, "clustering", "metadata_analysis", fig)


    def _plot_pca_mds(self, k_values, fitted, medoids, labels, metric, visualize, pca_type, n_components, save):
        """
        Generates 2D graphs for Euclidean PCA or Unifac/Bray-Curtis MDS.
        """

        fig, axes, num_values = self._init_plot(k_values, f"{pca_type.upper()} - {metric} distance", n_components)

        for i, k in enumerate(k_values):
            ax = axes[i]
            current_labels = labels[i] if isinstance(labels, (list, tuple)) else labels

            fitted_coords = [fitted[:, j] for j in range(n_components)]
            medoid_coords = [medoids[i][:, j] for j in range(n_components)]

            scatter = ax.scatter(*fitted_coords, c = current_labels, cmap = "tab10", alpha = 0.7)
            ax.scatter(*medoid_coords, color = "black", marker = "X", s = 150, label = "Medoids")
            
            ax.set_title(f"Cluster {pca_type.capitalize()} {n_components}D — {metric} — k = {k}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            if n_components == 3:
                ax.set_zlabel("Dim 3")
            ax.legend(*scatter.legend_elements(), title = "Cluster")
        
        for j in range(num_values, len(axes)):
            fig.delaxes(axes[j])

        self._close_plot(visualize, save, f"{pca_type}_{n_components}D", "clustering", f"{pca_type}_{n_components}D", fig)
        

    def _plot_stability_ari(self, matrices_data, distance, k, visualize = True):
        """
        Plots stability heatmaps for ARI and Fowlkes-Mallows metrics.
        matrices_data: list of (matrix, name) tuples, e.g., [(ari_matrix, "ARI"), (fm_matrix, "Fowlkes-Mallows")]
        """

        fig, axes, num_plots = self._init_plot(matrices_data, f"Clustering Stability - {distance} distance - k = {k}")

        for i, (matrix, name) in enumerate(matrices_data):
            ax = axes[i]
            sns.heatmap(matrix, annot = True, fmt = ".3f", cmap = "coolwarm", square = True, ax = ax)
            ax.set_title(f"Cluster Stability ({name})\nDistance: {distance}, k = {k}")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Fold")

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])


        self._close_plot(visualize, False, f"stability", "clustering", f"stability_k{k}", fig)
    

    def _plot_heatmap_full(self, 
                           title, 
                           data, 
                           target_name, 
                           visualize, 
                           save, 
                           set_index = None, 
                           index = None, 
                           fmt = ".2f", 
                           labels = None):
        """
        Plots a full heatmap.
        """
        
        fig, _, _ = self._init_plot([None], title)
        df = data.set_index('Column')[[index]] if set_index else data
        
        if labels is None:
            labels = False
            
        sns.heatmap(df, 
                    annot = True, 
                    cmap = 'coolwarm', 
                    fmt = fmt, 
                    linewidths = 0.5, 
                    cbar_kws = {'label': index}, 
                    xticklabels = labels, 
                    yticklabels = labels)
        
        plt.title(title)

        self._close_plot(visualize, 
                         save,
                         f"heatmap_{target_name}", 
                         "heatmap", 
                         f"heatmap_{target_name}", 
                         fig)


    def _plot_barplot(self, title, data, x, y, ylim, model_name, visualize, save):
        """
        Plots a barplot.
        """
        
        fig, _, _ = self._init_plot([None], title)
        sns.barplot(data = data, x = x, y = y, color = 'blue')
        plt.ylim(ylim)
        
        self._close_plot(visualize,
                         save,
                         f"barplot_{model_name}",
                         "models",
                         f"barplot_{model_name}",
                         fig)

    
    def _plot_histplot(self, title, data, bins, xlabel, ylabel, model_name, visualize, save):
        """
        Plots a histogram.
        """

        fig, _, _ = self._init_plot([None], title)
        sns.histplot(data, bins = bins, kde = True, color = 'skyblue')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        self._close_plot(visualize,
                         save,
                         f"histplot_{model_name}",
                         "models",
                         f"histplot_{model_name}",
                         fig)
        
    
    def _plot_boxplot(self, title, data, x, y, xlabel, ylabel, xticks, model_name, visualize, save):
        """
        Plots a boxplot.
        """

        fig, _, _ = self._init_plot([None], title)
        sns.boxplot(x = x, y = y, data = data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(xticks[0], xticks[1])
        self._close_plot(visualize,
                         save,
                         f"boxplot_{model_name}",
                         "models",
                         f"boxplot_{model_name}",
                         fig)


    def _plot_metadata_correlation(self, 
                                  p_matrix: pd.DataFrame, 
                                  chi2_matrix: pd.DataFrame, 
                                  visualize: bool = True, 
                                  save: bool = False):
        """
        Plots heatmaps for p-values and chi-squared statistics from metadata correlation analysis.
        """

        title = "Chi-squared P-values Heatmap"

        self._plot_heatmap_full(title, p_matrix, '-log10(p-value)', 'metadata', visualize, save, False)
        
        self._plot_heatmap_full(title.replace("P-", ""), chi2_matrix, 'Chi-squared', 'metadata', visualize, save, False)
        

    def _plot_cramers_v(self,
                       data: pd.DataFrame,
                       target_name: str,
                       visualize: bool = True,
                       save: bool = False):
        """
        Plots Cramer's V analysis results with a bar plot and heatmaps.
        """

        fig = plt.figure(figsize = (20, 10)) 
        gs = fig.add_gridspec(2, 3) 

        # Plot 1: P-value bar plot
        ax0 = fig.add_subplot(gs[0, :]) 
        sns.barplot(data = data, x = 'Column', y = 'P-value', color = 'blue', ax = ax0)
        ax0.axhline(0.05, color = 'red', linestyle = '--', label = 'P-value threshold (0.05)')
        ax0.set_title(f"P-value for each column")
        ax0.tick_params(axis = 'x', rotation = 45)
        ax0.legend()
        ax0.set_xlabel("") # Remove x-label to reduce clutter
        ax0.set_ylabel("P-value")
        
        # Prepare data for heatmaps
        df_p_value = data.set_index('Column')[['P-value']]
        df_chi2 = data.set_index('Column')[['Chi-Squared']]
        df_cramers_v = data.set_index('Column')[['Cramers_V']]

        # Plot 2: P-value Heatmap
        ax1 = fig.add_subplot(gs[1, 0]) 
        sns.heatmap(df_p_value, 
                    annot = True, 
                    cmap = 'coolwarm',
                    fmt = ".2f", 
                    linewidths = 0.5, 
                    cbar_kws = {'label': 'P-value'},
                    ax = ax1)
        ax1.set_title(f"P-value Heatmap")

        # Plot 3: Chi-Squared Heatmap
        ax2 = fig.add_subplot(gs[1, 1]) 
        sns.heatmap(df_chi2, 
                    annot = True, 
                    cmap = 'coolwarm', 
                    fmt = ".2f", 
                    linewidths = 0.5, 
                    cbar_kws = {'label': 'Chi-Squared'}, 
                    ax = ax2)
        ax2.set_title(f"Chi-Squared Heatmap")

        # Plot 4: Cramer's V Heatmap
        ax3 = fig.add_subplot(gs[1, 2]) 
        sns.heatmap(df_cramers_v, 
                    annot = True, 
                    cmap = 'coolwarm', 
                    fmt = ".2f", 
                    linewidths = 0.5, 
                    cbar_kws = {'label': "Cramer's V"}, 
                    ax = ax3)
        ax3.set_title(f"Cramer's V Heatmap")

        # Close the entire figure
        self._close_plot(visualize, 
                         save, 
                         f"cramers_v_analysis_{target_name.lower()}", 
                         "correlation", 
                         f"cramers_v_analysis_{target_name.lower()}", 
                         fig)


    def plot_DR(self, 
                data: dict, 
                method: str, 
                n_components: int, 
                visualize: bool = True, 
                save: bool = False,
                per_comp: pd.Series = None, 
                labels_dict: dict = None,
                color_map: list = None):
        """
        Plots dimensionality reduction results.
        
        Parameters
        ----------
            data: dict
                A dictionary where keys are method names (e.g., 'PCA', 'KPCA')
                and values are numpy arrays of the transformed data.
            method: str 
                The overall method name for the plot title and filename.
            n_components: int
                The number of components to plot (2 or 3).
            visualize: bool
                Whether to display the plot.
                save: bool
                Whether to save the plot to a file.
            per_comp: pd.Series
                principal components percents.
        """        
        
        fig, axes, num_plots = self._init_plot(data, f"{method.upper()}", n_components)

        # Support plotting multiple DR methods in a single window
        for i, (key, princ_vals) in enumerate(data.items()):            

            if i >= num_plots: # Safety break if more data items than allocated subplots
                self.logger.warning(f"Skipping plot for '{key}' as there are no more subplots available.")
                break
            
            ax = axes[i]
            
            if not isinstance(princ_vals, np.ndarray) or princ_vals.shape[1] < n_components:
                self.logger.warning(f"Skipping plot for '{key}' due to invalid data format or insufficient components.")
                ax.set_title(f"{key} (Data Invalid/Insufficient Components)")
                #continue

            ax.set_title(key)
            coords = [princ_vals[:, j] for j in range(n_components)]

            scatter_c_arg = None
            current_labels = None
            if labels_dict is not None and key in labels_dict:
                current_labels = labels_dict[key]

            if current_labels is not None and len(current_labels) == princ_vals.shape[0]:
                if color_map is not None:
                    unique_labels = np.unique(current_labels)
                    if len(color_map) < len(unique_labels):
                        self.logger.warning(
                            f"Not enough colors in 'color_map' for all unique labels ({len(unique_labels)}). "
                            f"Provided {len(color_map)} colors. Using default colormap."
                        )
                        # Fallback to default matplotlib colormap if not enough colors
                        scatter_c_arg = current_labels
                    else:
                        # Map numerical labels (0, 1, ...) to colors from color_map
                        scatter_c_arg = [color_map[label] for label in current_labels]
                else:
                    # No color_map provided, use current_labels with a default colormap
                    scatter_c_arg = current_labels

                # Pass cmap only if c is numerical, otherwise it will be ignored or cause issues
                scatter = ax.scatter(*coords, alpha=0.85, c=scatter_c_arg, cmap='viridis' if color_map is None else None)

                if color_map is not None and isinstance(scatter_c_arg[0], str):
                    # Manually create legend handles when color_map was successfully applied as strings
                    unique_labels = np.unique(current_labels)
                    legend_handles = []
                    legend_labels = []
                    for label_val in unique_labels:
                        if label_val < len(color_map): # Ensure index is valid
                            color = color_map[label_val]
                            legend_handles.append(Line2D([0], [0], marker = 'o', color = 'w', markerfacecolor = color, markersize = 10))
                            legend_labels.append("Responder" if label_val == 1 else "Non-Responder") # Using numerical label as string
                    ax.legend(handles = legend_handles, labels = legend_labels, title = "Labels")
                
                else:
                    # Use scatter.legend_elements() for numerical 'c' values and a colormap
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category = UserWarning)
                        ax.legend(*scatter.legend_elements(), title = "Labels")
            else:
                # Default to gray if no labels or labels don't match
                ax.scatter(*coords, alpha = 0.85, c = ['gray'] * princ_vals.shape[0])

            if per_comp is not None:
                ax.set_xlabel(f"1st comp {per_comp[0] * 100}")
                ax.set_ylabel(f"2nd comp {per_comp[1] * 100}")
                if n_components == 3:
                    ax.set_zlabel(f"3rd comp {per_comp[2] * 100}")
            else:
                ax.set_xlabel("1st comp")
                ax.set_ylabel("2nd comp")
                if n_components == 3:
                    ax.set_zlabel("3rd comp")
            
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        self._close_plot(visualize, save, method, "dimensionality_reduction", method, fig)


    def plot_cumulative_vars(self, 
                             data: dict,
                             threshold : float = 0.9,
                             save = False,
                             visualize = True):
        """
        Plots the cumulative explained variance for different dimensionality reduction methods.

        Parameters
        ----------
            data: dict
                Dictionary where keys are method names and values are arrays of cumulative variance.
            threshold: float, default 0.9
                The variance threshold to highlight on the plot.
            save: bool, default False
                Whether to save the plot to disk.
            visualize: bool, default True
                Whether to display the plot.
                
        """
        # Filter out useless results
        valid_variances = {k: np.asarray(v)
                        for k, v in data.items()
                        if v is not None and len(v) > 0}

        if len(valid_variances) < 1:
            self.logger.warning("Cannot create cumulative variance plot with not enough valid variances")
            return
        
        max_n = max(v.shape[0] for v in valid_variances.values())

        fig, _, _ = self._init_plot([None], "Cumulative Explained Variance")

        plt.grid(True, linestyle = '--', alpha = 0.3, zorder = 0)

        for method, variances in valid_variances.items():
            x = np.arange(1, len(variances) + 1)
            label = ""
            if method == "PCOA_bc":
                label = "PCoA (Bray-Curtis)"
            elif method == "PCOA_js":
                label = "PCoA (Jensen-Shannon)"
            else:
                label = method

            plt.plot(x, variances, label = label, linewidth = 2, zorder = 3)

            # Number of components to reach threshold and annotate
            threshold_reached_indices = np.where(variances >= threshold)[0]
            if threshold_reached_indices.size > 0:
                n_comp = threshold_reached_indices[0] + 1

                plt.scatter(n_comp, variances[n_comp - 1], color = 'darkred', s = 40, zorder = 5)

                plt.annotate(f'{n_comp}',
                            xy = (n_comp, variances[n_comp - 1]),
                            xytext = (6, -10),
                            textcoords = 'offset points',
                            fontsize = 9,
                            color = 'darkred',
                            bbox = dict(boxstyle='round,pad=0.3',
                            facecolor = 'white',
                            edgecolor = 'white',
                            alpha = 0.8))
            else:
                self.logger.warning(f"Threshold {threshold*100}% not reached for method '{method}'. No annotation will be plotted.")

        plt.axhline(y = threshold, color = 'black', linestyle = ':', linewidth = 1.5, alpha = 0.8, label = f'Threshold {threshold*100}%')

        plt.xticks(np.arange(1, max_n + 1, max(1, max_n // 10)))
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.xlabel('Components', fontsize = 12)
        plt.ylabel('Cumulative Variance', fontsize = 12)
        plt.legend()
        
        self._close_plot(visualize, 
                         save, 
                         "dimensionality_reduction",
                         "cumulative_variance", 
                         "dimensionality_reduction",
                         fig)


    def plot_LDA(self,
                 data: pd.DataFrame,
                 palette: dict = None,
                 transform_method: str = None,
                 visualize: bool = True,
                 save: bool = False):
        """
        Plots the results of Linear Discriminant Analysis using a Kernel Density Estimate (KDE).

        Parameters
        ----------
            data: pd.DataFrame
                DataFrame containing the LDA components and labels.
            palette: dict, optional
                Color mapping for the labels.
            transform_method: str, optional
                The name of the transformation method used (e.g., 'CLR').
            visualize: bool, default True
                Whether to display the plot.
            save: bool, default False
                Whether to save the plot to disk.
                
        """

        fig, _, _ = self._init_plot([None], "LDA")

        if palette is None:
            palette = {0: "red", 1: "green"}

        sns.kdeplot(
            data = data,
            x = 'Comp 1',
            hue = 'Response',
            fill = True,
            alpha = 0.4,
            palette = palette,
            legend = False
        )

        legend_elements = [
            Line2D([0], [0], marker = 'o', color = 'w', label = 'Non Responder', markerfacecolor = 'red', markersize = 7),
            Line2D([0], [0], marker = 'o', color = 'w', label = 'Responder', markerfacecolor = 'green', markersize = 7)
        ]

        plt.legend(handles = legend_elements)
        plt.xlabel('Component 1')
        plt.ylabel('Density')
        plt.title(f'Linear Discriminant Analysis ({transform_method.upper()})')
        
        self._close_plot(visualize, save, "LDA", "clustering", "LDA", fig)


    def _plot_confusion_matrix(self, confusion_matrix, visualize, save):
        """
        Plots a confusion matrix.
        """
        fig, ax, _ = self._init_plot([None], "Confusion Matrix")

        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot(ax = ax[0])
        ax[0].set_title("Confusion Matrix")
       
        self._close_plot(visualize, save, "confusion_matrix", "classification", "confusion_matrix", fig)
        

    def _plot_confusion_matrices(self,
                                y_true_train, 
                                y_pred_train,
                                y_true_val, 
                                y_pred_val,
                                model_name,
                                val_set_name = "Validation",
                                visualize = True,
                                save = False):
        """
        Plots confusion matrices for training and validation/CV sets.
        """
        fig, axes, _ = self._init_plot([None, None], f"Confusion Matrices for {model_name}")

        # Train Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(y_true_train, y_pred_train, ax = axes[0], cmap = 'Blues')
        axes[0].set_title('Train Set')

        # Validation/CV Confusion Matrix
        if y_true_val is not None and y_pred_val is not None:
            ConfusionMatrixDisplay.from_predictions(y_true_val, y_pred_val, ax = axes[1], cmap = 'Blues')
            axes[1].set_title(f'{val_set_name} Set')
        else:
            # Hide the second subplot if no validation data
            fig.delaxes(axes[1])

        self._close_plot(visualize, save, f"confusion_matrix_{model_name}", "classification", model_name, fig)


    def _plot_metrics_comparison(self, metrics_df: pd.DataFrame, model_name: str, visualize = True, save = False):
        """
        Plots a bar chart comparing different performance metrics.
        """
        fig, ax, _ = self._init_plot([None], f"Metrics Comparison for {model_name}")

        metrics_df.plot(kind='bar', ax = ax[0])
        ax[0].set_title(f'Train vs. Validation/CV Metrics')
        ax[0].set_xlabel('Metric')
        ax[0].set_ylabel('Score')
        ax[0].set_ylim(0, 1)
        ax[0].tick_params(axis = 'x', rotation = 45)
        ax[0].grid(True, linestyle = '--', alpha = 0.6)

        self._close_plot(visualize, save, f"metrics_comparison_{model_name}", "classification", model_name, fig)


    def _plot_performance_curves(self,
                         curve_type: Literal['roc', 'pr'],
                         x_data,
                         y_data,
                         metric_train_value, # AUC or AP
                         baseline_value = None, # For PR, this is the horizontal line value. For ROC, it's ignored (uses [0,1] line)
                         cv_results: dict = None,
                         model_name: str = "Model",
                         visualize: bool = True,
                         save: bool = False):
        """
        Plots ROC or Precision-Recall curves for training and optionally for cross-validation results.
        """

        fig, ax, _ = self._init_plot([None], f"{curve_type.upper()} Curve for {model_name}")

        # Determine labels and title based on curve_type
        if curve_type == 'roc':
            x_label = 'False Positive Rate'
            y_label = 'True Positive Rate'
            title = 'ROC Curve'
            train_label = f'Model (AUC = {metric_train_value:.2f})'
            cv_label_format = 'Cross-Val (AUC = {:.2f} ± {:.2f})'
            baseline_plot_args = ([0, 1], [0, 1]) # Diagonal line for ROC
            baseline_label = 'Random'
            legend_loc = 'lower right'
            filename_prefix = "roc_curve"
        elif curve_type == 'pr':
            x_label = 'Recall'
            y_label = 'Precision'
            title = 'Precision-Recall Curve'
            train_label = f'Train (AP = {metric_train_value:.2f})'
            cv_label_format = 'Cross-Val (AP = {:.2f})'
            baseline_plot_args = ([0, 1], [baseline_value, baseline_value]) # Horizontal line for PR
            baseline_label = f'Baseline ({baseline_value:.2f})'
            legend_loc = 'upper right'
            filename_prefix = "pr_curve"
        else:
            self.logger.error(f"Invalid curve_type: {curve_type}. Must be 'roc' or 'pr'.")
            self._close_plot(visualize, save, f"invalid_curve_type_{model_name}", "classification", model_name, fig)
            return

        # Plot Train Curve if data is available
        if x_data is not None and y_data is not None:
            ax[0].plot(x_data, y_data, label = train_label, lw = 2)

        # Plot CV Curve if results are available
        if cv_results:
            if curve_type == 'roc':
                mean_x_cv = cv_results.get('mean_fpr')
                mean_y_cv = cv_results.get('mean_tpr')
                mean_metric_cv = cv_results.get('mean_auc')
                std_metric_cv = cv_results.get('std_auc')
                if all(v is not None for v in [mean_x_cv, mean_y_cv, mean_metric_cv, std_metric_cv]):
                    ax[0].plot(mean_x_cv, mean_y_cv, lw = 2, label = cv_label_format.format(mean_metric_cv, std_metric_cv))
            
            else: # pr
                mean_x_cv = cv_results.get('mean_recall')
                mean_y_cv = cv_results.get('mean_precision')
                mean_metric_cv = cv_results.get('mean_ap')
                if all(v is not None for v in [mean_x_cv, mean_y_cv, mean_metric_cv]):
                    ax[0].plot(mean_x_cv, mean_y_cv, lw = 2, label = cv_label_format.format(mean_metric_cv))

        # Plot Baseline
        ax[0].plot(*baseline_plot_args, 'k--', lw = 1, label = baseline_label)

        ax[0].set_xlabel(x_label)
        ax[0].set_ylabel(y_label)
        ax[0].set_title(title)
        ax[0].legend(loc = legend_loc)
        ax[0].grid(True, linestyle = '--', alpha = 0.6)

        if curve_type == 'pr':
            ax[0].set_xlim([0.0, 1.0])
            ax[0].set_ylim([0.0, 1.05])

        self._close_plot(visualize, save, f"{filename_prefix}_{model_name}", "classification", model_name, fig)


    def _plot_models_metrics(self,
                             cm,
                             df_barplot,
                             confidences,
                             df_boxplot,
                             labels,
                             model_name,
                             visualize,
                             save):
        """
        Plots metrics from trained models.
        """

        self._plot_heatmap_full(
            "Confusion Matrix",
            cm,
            model_name,
            visualize,
            save,
            False,
            fmt = "d",
            labels = labels,
        )

        self._plot_barplot("Metrics",
                           df_barplot,
                           'Metric',
                           'Value',
                           0.1,
                           model_name,
                           visualize,
                           save)

        self._plot_histplot("Confidences Histogram",
                            confidences,
                            10,
                            "Max prob",
                            "Count",
                            model_name,
                            visualize,
                            save)

        self._plot_boxplot("Confidences: correct vs wrong",
                           df_boxplot,
                           "correct",
                           "confidence",
                           "Correctness",
                           "Confidence",
                           [[0,1], ["Wrong", "Correct"]],
                           model_name,
                           visualize,
                           save)
        
                                  
    def _plot_feature_importance(self, data, visualize, save, imp_type = "PLS"):
        """
        Plots feature importance.
        """

        fig, ax, _ = self._init_plot([None], "Feature Importance")
        ax[0].barh(data["feature"], data["coeffs"], color = "skyblue")
        ax[0].set_xlabel("Score")
        ax[0].set_ylabel("Feature")
        ax[0].set_title(f"Top Feature Importance by scores ({imp_type})")
        ax[0].invert_yaxis()
        self._close_plot(visualize, save, "feature_importance", "classification", "feature_importance", fig, False)


    def _shap_barplot_beeswarm(self, shap_values, data, visualize):
        """
        Plots SHAP values using both a bar plot and a beeswarm plot.
        """
        
        if visualize:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
            summary_plot(shap_values, data, plot_type = "bar", max_display = 20)
            summary_plot(shap_values, data, max_display = 20)
        
