import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Target function
def f_target(x):
    return np.sin(2 * np.pi * x)

# Generate data
n_samples = 500
x = np.linspace(0, 1, n_samples)
y = f_target(x)

# Convert to tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define the model
def build_model():
    """Builds a simple neural network for UAT demonstration."""
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )
    return model

# Initialize model, loss, and optimizer
model = build_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Test the model
y_pred = model(x_tensor).detach().numpy()

# Plot results
plt.plot(x, y, label="Target (sin(2πx))", color='blue')
plt.plot(x, y_pred, label="NN Approximation", color='red', linestyle='--')
plt.legend()
plt.title("NN Approximation of f(x)")
plt.show()
