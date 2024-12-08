import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import torch
import torch.nn as nn
import torch.optim as optim
import os
import imageio

# Ensure the directory exists before saving the figure
frame_directory = 'C:\\Users\\Lenovo\\Desktop\\programming\\TASKS-and-PROJECTS-2024-25\\Personal-tomfoolery\\universal-approximation-UAT-visualization\\assets'

# Check if the directory exists, if not create it
if not os.path.exists(frame_directory):
    os.makedirs(frame_directory)

# Function to approximate
def f(x):
    return np.sin(2 * np.pi * x)

# B-spline approximation
def b_spline_approximation(x, knots, coeffs, degree=3):
    """
    Fit a B-spline to the data using given knots and coefficients.
    Returns the B-spline values for the input x.
    """
    spline = BSpline(knots, coeffs, degree)
    return spline(x)

# Train B-spline approximation with optimization
def train_b_spline_approximation(x, y, epochs=200, knots=10, lr=0.01):
    """
    Train the B-spline approximation and generate frames.
    Optimizes the coefficients using gradient descent.
    """
    # Create knots (evenly spaced)
    x_knots = np.linspace(0, 1, knots)
    
    # Initialize coefficients randomly
    coeffs = torch.randn(knots, requires_grad=True, dtype=torch.float32)
    
    # Optimizer
    optimizer = optim.Adam([coeffs], lr=lr)
    frames = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Approximate using B-splines
        y_pred = b_spline_approximation(x, x_knots, coeffs)
        
        # Loss function: Mean Squared Error
        loss = torch.mean((y_pred - y) ** 2)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Plot and save frames every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            plt.figure(figsize=(6, 4))
            plt.plot(x, y, label='Target Function', color='blue')
            plt.plot(x, y_pred.detach().numpy(), label=f'B-Spline Approximation (Epoch {epoch + 1})', color='red')
            plt.legend()
            plt.title("Universal Approximation Theorem")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.grid()
            plt.tight_layout()
            frame_path = f"{frame_directory}/frame_{epoch + 1}.png"
            plt.savefig(frame_path)
            frames.append(frame_path)
            plt.close()

    return frames

# Main loop for training and visualization
if __name__ == "__main__":
    # Generate data
    x = np.linspace(0, 1, 100)
    y = f(x)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Train B-spline approximation and generate frames
    frames = train_b_spline_approximation(x_tensor, torch.tensor(y, dtype=torch.float32))

    # Create GIF
    images = []
    for frame in frames:
        images.append(imageio.imread(frame))
    gif_path = 'spline_approximation.gif'
    imageio.mimsave(gif_path, images, duration=0.5)  # Duration in seconds for each frame

    print("GIF saved as", gif_path)
