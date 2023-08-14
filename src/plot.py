import seaborn as sns 
import matplotlib.pyplot as plt


def plot_pareto_front(pareto_front, path="pareto_front.png"):
    accs = [point.val_accuracy for point in pareto_front]
    times = [-point.train_time for point in pareto_front]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Create the scatter plot
    sns.scatterplot(x=times, y=accs, marker='o', color='b')

    # Add labels and title
    plt.xlabel("Training Times")
    plt.ylabel("Validation Accuracies")


    plt.title("Pareto Front")
    plt.savefig(path)
    
    pass