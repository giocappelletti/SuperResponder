import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from logger.logger import logger
from utils.utils import sanitize_filename


class Plotter:
    """
    Handles the visualization of clustering evaluation metrics with metric-specific labeling.

    Parameters
    ----------
        output_dir (str): Plots save directory.
    """
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        self.logger = logger

    def _visualize(self):
        """
        Displays the current plot.
        """
        self.logger.info("Displaying plot")
        plt.show()

    def _save(self, subfolder, exp_name, metric_name, bbox_inches=None):
        """
        Saves the plot to a file with metric-specific naming.
        """
        out_path = os.path.join(self.output_dir, subfolder, f"{time.strftime('%Y%m%d_%H%M%S')}", exp_name)
        os.makedirs(out_path, exist_ok=True)
        
        safe_name = sanitize_filename(metric_name)
        save_path = os.path.join(out_path, f"{exp_name}_{safe_name}.png")
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches=bbox_inches) 
            self.logger.info(f"Plot saved to: {save_path}")
        
        except:
            self.logger.warning(f"Error saving plot to: {save_path}")

    def _compute_rows_cols(self, num_values):
        """
        Computes the number of rows and columns for subplots.
        """        
        # Max 3 columns is usually better for readability with legends
        ncols = max(1, min(3, num_values))
        nrows = max(1, (num_values + ncols - 1) // ncols)

        return nrows, ncols


    def _init_plot(self, values, title, n_components = None):
        
        num_values = len(values)

        nrows, ncols = self._compute_rows_cols(num_values)

        # Determine if 3D projection is needed
        projection = '3d' if n_components == 3 else None
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5), squeeze=False, subplot_kw={'projection': projection})
        fig.canvas.manager.set_window_title(title)
      
        return fig, axes.flatten(), num_values
    

    def _close_plot(self, visualize, save, metric_name, subfolder, exp_name, fig):
        """
        Closes the current plot.
        """
        plt.tight_layout()

        if visualize:
            self._visualize() 
        if save:
            self._save(subfolder, exp_name, metric_name, bbox_inches='tight')

        plt.close(fig)


    def _plot_elbow_and_silhouette(self, ks, inertias, silhouettes, tick_values, 
                                  metric_name, visualize=True, save=False):
        """
        Plots Elbow and Silhouette results with the specific metric name integrated,
        using _init_plot for consistent subplot initialization.
        
        Parameters
        ----------
            ks, inertias, silhouettes, tick_values: Data from Clustering class.\n
            metric_name (string):  name of the metric.
            visualize (bool):  wether to show or save the plot.
        """
        
        # Initialize subplots using _init_plot
        fig, axes, _ = self._init_plot([None, None], f"Clustering Analysis") # Pass a dummy list of 2 items to ensure 2 subplots

        for i in range(len(axes)):
            axes[i].grid(True, linestyle="--", alpha=0.6)
            axes[i].set_xticks(tick_values)
            axes[i].set_xlabel("Number of clusters (k)")

        # Elbow Plot
        axes[0].plot(ks, inertias, marker='o', color='royalblue', linewidth=2)
        axes[0].set_ylabel("Inertia")
        axes[0].set_title(f"Elbow Method\nDistance: {metric_name}") 

        # Silhouette Plot
        axes[1].plot(ks, silhouettes, marker='s', color='forestgreen', linewidth=2)
        axes[1].set_ylabel("Average Silhouette Score")
        axes[1].set_title(f"Silhouette Method\nDistance: {metric_name}")

        self._close_plot(visualize, save, metric_name, "clustering", "elbow_silhouette", fig)
            

    def _plot_contingencies(self, contingencies, cols, distance_name, visualize=True, save=False):
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
            
            contingency_prop.plot(kind='bar', stacked=True, ax=ax, legend=show_legend) 
            
            ax.set_title(f"{col} Distribution\n(Distance: {distance_name})", fontsize=12)
            ax.set_ylabel("Proportion on total")
            ax.set_xlabel("Cluster")
            
            ax.tick_params(axis='x', rotation=45)

            if show_legend:
                ax.legend(title=col, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            else:
                ax.set_title(f"{col} (Legend hidden - {num_categories} cats)\nDistance: {distance_name}", color='red')
        
        for j in range(num_contingencies, len(axes)):
            fig.delaxes(axes[j])

        self._close_plot(visualize, False, distance_name, "clustering", "metadata_analysis", fig)


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

            scatter = ax.scatter(*fitted_coords, c=current_labels, cmap="tab10", alpha=0.7)
            ax.scatter(*medoid_coords, color="black", marker="X", s=150, label="Medoids")
            
            ax.set_title(f"Cluster {pca_type.capitalize()} {n_components}D — {metric} — k={k}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            if n_components == 3:
                ax.set_zlabel("Dim 3")
            ax.legend(*scatter.legend_elements(), title="Cluster")
        
        for j in range(num_values, len(axes)):
            fig.delaxes(axes[j])

        self._close_plot(visualize, save, f"{pca_type}_{n_components}D", "clustering", f"{pca_type}_{n_components}D", fig)
        

    def _plot_stability_ari(self, matrices_data, distance, k, visualize=True):
        """
        Plots stability heatmaps for ARI and Fowlkes-Mallows metrics.
        matrices_data: list of (matrix, name) tuples, e.g., [(ari_matrix, "ARI"), (fm_matrix, "Fowlkes-Mallows")]
        """
        fig, axes, num_plots = self._init_plot(matrices_data, f"Clustering Stability - {distance} distance - k={k}")

        for i, (matrix, name) in enumerate(matrices_data):
            ax = axes[i]
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm", square=True, ax=ax)
            ax.set_title(f"Cluster Stability ({name})\nDistance: {distance}, k={k}")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Fold")

        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._visualize() if visualize else self._save("clustering", f"stability_k{k}", f"{distance}")
        plt.close(fig)
            
# One time initialization of plotter object
plotter = Plotter()