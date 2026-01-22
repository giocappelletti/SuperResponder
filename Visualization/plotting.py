import os
import matplotlib.pyplot as plt

class Plotter:
    """
    Handles the visualization of clustering evaluation metrics with metric-specific labeling.
    """
    def __init__(self, output_dir="clustering_plots"):
        self.output_dir = output_dir

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
        fig.canvas.manager.set_window_title(f"Clustering Analysis - {metric_name}")

        # Elbow Plot
        axes[0].plot(ks, inertias, marker='o', color='royalblue', linewidth=2)
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("Inertia")
        # Integrating metric name into the subplot title
        axes[0].set_title(f"Elbow Method\n(Metric: {metric_name})") 
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].set_xticks(tick_values)

        # Silhouette Plot
        axes[1].plot(ks, silhouettes, marker='s', color='forestgreen', linewidth=2)
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Average Silhouette Score")
        # Integrating metric name into the subplot title
        axes[1].set_title(f"Silhouette Method\n(Metric: {metric_name})")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].set_xticks(tick_values)

        plt.tight_layout()

        if visualize:
            plt.show()
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            # 2. Integrate metric name into the filename for clear organization
            safe_name = metric_name.lower().replace(" ", "_")
            save_path = os.path.join(self.output_dir, f"clustering_{safe_name}.png")
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
            plt.close(fig)