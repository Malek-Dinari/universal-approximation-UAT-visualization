import numpy as np
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import imageio
import os
import matplotlib
import matplotlib.pyplot as plt

# Ensure directory exists
if not os.path.exists('assets'):
    os.makedirs('assets')

# Target function: sin(2 * pi * x)
target_vals = np.sin(2 * np.pi * np.linspace(0, 1, 100))

# B-Spline basis functions and control points
def b_spline_basis(i, degree, x, knots):
    """ Evaluate the i-th B-spline basis function of given degree at x. """
    if degree == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    else:
        left = (x - knots[i]) / (knots[i + degree] - knots[i]) if knots[i + degree] != knots[i] else 0.0
        right = (knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1]) if knots[i + degree + 1] != knots[i + 1] else 0.0
        return left * b_spline_basis(i, degree - 1, x, knots) + right * b_spline_basis(i + 1, degree - 1, x, knots)

def b_spline_interpolation(x, control_points, degree, knots):
    """ Perform B-spline interpolation given the control points, degree, and knots. """
    y = np.zeros_like(x)
    for j in range(len(x)):
        for i in range(len(control_points)):
            y[j] += control_points[i] * b_spline_basis(i, degree, x[j], knots)
    return y

# Example B-Spline Interpolation Initialization
n_knots = 8    # Number of knots (including boundary knots)
degree = 3     # Degree of the B-spline (cubic splines)
n_control_points = n_knots - degree - 1  # Number of control points
knots = np.linspace(0, 1, n_knots)

# Define a simple K-A Network
class KANetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, control_points):
        super(KANetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        # Make control points learnable
        self.control_points = nn.Parameter(torch.randn(control_points))  # Learnable control points

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)

# Training function for both models and generating GIFs
def train_and_generate_gif(model, optimizer, criterion, x_train, y_train, epochs=200, filename='output.gif'):
    fig, ax = plt.subplots()
    line_target, = ax.plot([], [], label=r'Target Function $f(x) = \sin(2\pi x)$', color='black')
    line_control, = ax.plot([], [], label='Control Points', color='red')
    line_approx, = ax.plot([], [], label='Approximated Function', color='blue')

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()

    frames = []
    mse_values = []  # To store MSE values for plotting
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)

        # Compute loss
        loss = criterion(y_pred, y_train)
        mse_values.append(loss.item())  # Store MSE value

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update control points (for B-spline) or network approximation
        line_target.set_data(np.linspace(0, 1, 100), target_vals)
        line_control.set_data(np.linspace(0, 1, n_control_points), model.control_points.detach().numpy())  # Use learned control points
        line_approx.set_data(np.linspace(0, 1, 100), b_spline_interpolation(np.linspace(0, 1, 100), model.control_points.detach().numpy(), degree, knots))

        # Render the figure
        fig.canvas.draw()  # Force drawing of the figure

        # Convert to RGBA array
        frame = np.array(fig.canvas.buffer_rgba())
        frames.append(frame)

    # Save frames as a gif
    imageio.mimsave(os.path.join('assets', filename), frames, duration=0.1)
    print(f"Saved {filename}.")

    return mse_values  # Return the MSE values

# Prepare data for training
x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
y_vals = np.sin(2 * np.pi * x_vals)  # Target function: sin(2*pi*x)
x_train = torch.tensor(x_vals, dtype=torch.float32)
y_train = torch.tensor(y_vals, dtype=torch.float32)

# Initialize and train K-A Network
ka_model = KANetwork(input_dim=1, hidden_dim=50, control_points=n_control_points)
ka_optimizer = optim.Adam(ka_model.parameters(), lr=0.001)
ka_criterion = nn.MSELoss()

# Train and save K-A model GIF and MSE values
mse_values_ka = train_and_generate_gif(ka_model, ka_optimizer, ka_criterion, x_train, y_train, epochs=200, filename='K-A_Network.gif')

# Plot MSE for K-A Network
plt.figure(figsize=(8, 6))
plt.plot(mse_values_ka, label='K-A Network MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE Progression for K-A Network')
plt.legend()
plt.grid(True)
plt.savefig('assets/mse_ka_network.png')
plt.show()