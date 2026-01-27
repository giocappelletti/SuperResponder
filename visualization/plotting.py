import os
import time
import matplotlib.pyplot as plt

from logger.logger import logger
from utils.serialization import Serialization

class Plotter:
    """
    Handles the visualization of clustering evaluation metrics with metric-specific labeling.
    """
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        self.logger = logger

    def _visualize(self):
        self.logger.info("Displaying plot")
        plt.show()

    def _save(self, subfolder, exp_name, metric_name, bbox_inches=None):
        """Saves the plot to a file with metric-specific naming"""
        out_path = os.path.join(self.output_dir, subfolder, f"{time.strftime('%Y%m%d_%H%M%S')}", exp_name)
        os.makedirs(out_path, exist_ok=True)
        
        safe_name = Serialization.sanitize_filename(metric_name)
        save_path = os.path.join(out_path, f"{exp_name}_{safe_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches=bbox_inches) 
        self.logger.info(f"Plot saved to: {save_path}")

    def plot_elbow_and_silhouette(self, ks, inertias, silhouettes, tick_values, 
                                  metric_name, visualize=True):
        """
        Plots Elbow and Silhouette results with the specific metric name integrated.
        
        Args:
            ks, inertias, silhouettes, tick_values: Data from Clustering class.
            metric_name: String name of the metric (e.g., 'Bray-Curtis', 'UniFrac'). [cite: 81, 124]
            visualize: Boolean to show or save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # 1. Set the Window Title (the one on the OS title bar)
        fig.canvas.manager.set_window_title(f"Clustering Analysis")

        # Elbow Plot
        axes[0].plot(ks, inertias, marker='o', color='royalblue', linewidth=2)
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("Inertia")
        # Integrating metric name into the subplot title
        axes[0].set_title(f"Elbow Method\nDistance: {metric_name}") 
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].set_xticks(tick_values)

        # Silhouette Plot
        axes[1].plot(ks, silhouettes, marker='s', color='forestgreen', linewidth=2)
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Average Silhouette Score")
        # Integrating metric name into the subplot title
        axes[1].set_title(f"Silhouette Method\nDistance: {metric_name}")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].set_xticks(tick_values)

        plt.tight_layout()

        self._visualize() if visualize else self._save("clustering", "elbow_silhouette", metric_name)
        plt.close(fig)
            

    def plot_contingency(self, contingencies, cols, distance_name, visualize=True):
        """
        Plots a heatmap for the given contingency table.
        
        Args:
            contingency_df: DataFrame representing the contingency table.
            title: Title for the heatmap.
            visualize: Boolean to show or save the plot.
        """
        num_contingencies = len(contingencies)
        
        # Determine grid dimensions: max 4 columns, calculate rows needed
        ncols = min(num_contingencies, 4)
        nrows = (num_contingencies + ncols - 1) // ncols # Ceiling division
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5)) # Adjust figsize dynamically
        fig.canvas.manager.set_window_title(f"Metadata Analysis")
        
        # Ensure axes is always an array for consistent iteration
        if num_contingencies == 1:
            axes = [axes]
        else:
            axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
            
        for i in range(num_contingencies):
            col = cols[i]
            ax = axes[i]
            contingency_prop = contingencies[i]
            contingency_prop.plot(kind='bar', stacked=True, ax=ax) # Correct way to plot pandas DataFrame on an Axes
            ax.set_title(f"{col} Distribution by Cluster\nDistance: {distance_name}")
            ax.set_ylabel("Proportion on total")
            ax.set_xlabel("Cluster")
            ax.legend(title=col, loc='upper left', bbox_to_anchor=(1,1))
        
        # Hide any unused subplots if num_contingencies is not a perfect multiple of ncols
        for j in range(num_contingencies, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()

        self._visualize() if visualize else self._save("clustering", "contingencies_distribution",
                                                        distance_name, bbox_inches='tight')   
        plt.close()

# One time initialization of plotter object
plotter = Plotter()