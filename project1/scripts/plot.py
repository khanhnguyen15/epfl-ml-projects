import numpy as np
import matplotlib.pyplot as plt

def ridge_cross_validation_visualization(lambdas, accuracies):
    """visualization the curves of mse_tr and mse_te."""
    colors = ['r', 'b', 'y', 'g']
    labels = ['group_0', 'group_1', 'group_2', 'group_3']
    for i in range(len(accuracies)):
        plt.semilogx(lambdas, accuracies[i], marker=".", color=colors[i], label=labels[i])
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("ridge_cross_validation")
    
def poly_cross_validation_visualization(polys, accuracies):
    """visualization the curves of mse_tr and mse_te."""
    colors = ['r', 'b', 'y', 'g']
    labels = ['group_0', 'group_1', 'group_2', 'group_3']
    for i in range(len(accuracies)):
        plt.plot(polys, accuracies[i], marker=".", color=colors[i], label=labels[i])
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("polynomial_cross_validation")
