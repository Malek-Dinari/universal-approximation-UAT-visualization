import numpy as np
import torch
import matplotlib.pyplot as plt

def universal_approximation_function(x):
    # Placeholder for a neural network or approximation function
    return torch.sin(x)

def plot_approximation():
    x = np.linspace(-5, 5, 100)
    y = universal_approximation_function(torch.tensor(x)).numpy()

    plt.plot(x, y)
    plt.title("Universal Approximation Theorem Visualization")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_approximation()
