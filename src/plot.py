from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metrics import normalize_values
from schema import ArchCoupled


def plot_pareto_front(pareto_front: List[ArchCoupled], min_max: Dict, path="pareto_front.png") -> None:
    """
    Plots the pareto front.

    Args:
        pareto_front (List[ArchCoupled]): List of architectures in the pareto front.
        min_max (Dict): Dictionary containing the min and max values for val_accuracy and train_time.
        path (str, optional): Path to save plot. Defaults to "pareto_front.png".
        
    """
    accs = [point.val_accuracy for point in pareto_front]
    times = np.array([point.train_time for point in pareto_front])
    normalized_times = normalize_values(times, worst_value=min_max['max_train_time'], best_value=min_max['min_train_time'])

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Create the scatter plot
    sns.scatterplot(x=normalized_times, y=accs, marker='o', color='b')

    # Add labels and title
    plt.xlabel("Training Times (Inversely Normalized)")
    plt.ylabel("Validation Accuracies")


    plt.title("Pareto Front")
    plt.savefig(path)
    
    pass