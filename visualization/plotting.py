import os
import time
import matplotlib.pyplot as plt

from logger.logger import logger
from utils.serialization import Serialization
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
        
        plt.savefig(save_path, dpi=300, bbox_inches=bbox_inches) 
        self.logger.info(f"Plot saved to: {save_path}")

    def plot_elbow_and_silhouette(self, ks, inertias, silhouettes, tick_values, 
                                  metric_name, visualize=True):
        """
        Plots Elbow and Silhouette results with the specific metric name integrated.
        
        Parameters
        ----------
            ks, inertias, silhouettes, tick_values: Data from Clustering class.\n
            metric_name: String name of the metric.
            visualize: Boolean to show or save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # 1. Set the Window Title (the one on the OS title bar)
        fig.canvas.manager.set_window_title(f"Clustering Analysis")

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

        plt.tight_layout()

        self._visualize() if visualize else self._save("clustering", "elbow_silhouette", metric_name)

        plt.close(fig)
            

    def plot_contingencies(self, contingencies, cols, distance_name, visualize=True):
        num_contingencies = len(contingencies)
        
        # Max 3 columns is usually better for readability with legends
        ncols = 3 
        nrows = (num_contingencies + ncols - 1) // ncols
        
        # Increased width per column to accommodate legends
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5))
        fig.canvas.manager.set_window_title(f"Metadata Distribution - {distance_name}")
        
        if num_contingencies == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i in range(num_contingencies):
            col = cols[i]
            ax = axes[i]
            contingency_prop = contingencies[i]
            
            # Check number of unique categories to decide on legend
            num_categories = len(contingency_prop.columns)
            show_legend = num_categories <= 15 # Hide legend if too many items
            
            contingency_prop.plot(kind='bar', stacked=True, ax=ax, legend=show_legend) 
            
            ax.set_title(f"{col} Distribution\n(Distance: {distance_name})", fontsize=12)
            ax.set_ylabel("Proportion on total")
            ax.set_xlabel("Cluster")
            
            # Rotate x-axis labels if they are long
            ax.tick_params(axis='x', rotation=45)

            if show_legend:
                ax.legend(title=col, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
            else:
                ax.set_title(f"{col} (Legend hidden - {num_categories} cats)\nDistance: {distance_name}", color='red')
        
        # Hide unused subplots
        for j in range(num_contingencies, len(axes)):
            fig.delaxes(axes[j])
            
        # Use rect to prevent subplot titles from hitting the figure top
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        if visualize:
            plt.show()
        else:
            self._save("clustering", "contingencies_distribution",
                       distance_name, bbox_inches='tight')   
        plt.close()

        
# One time initialization of plotter object
plotter = Plotter()