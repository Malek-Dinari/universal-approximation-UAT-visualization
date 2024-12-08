import numpy as np

def target_function(x):
    """Defines the target function."""
    return np.sin(2 * np.pi * x)

def generate_dataset():
    """Generates the dataset for target function."""
    x = np.linspace(-1, 1, 200)
    y = target_function(x)
    return x, y

if __name__ == "__main__":
    x, y = generate_dataset()
    np.savez("data/dataset.npz", x=x, y=y)
