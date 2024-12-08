import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from build_model import build_model

def train_model(model, x, y, epochs=200, lr=0.01):
    """Trains the model and generates frames."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    frames = []

    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot and save frames
        if epoch % 5 == 0 or epoch == epochs - 1:  # Save every 5 epochs
            with torch.no_grad():
                plt.figure(figsize=(6, 4))
                plt.plot(x.cpu(), y.cpu(), label='Target Function', color='blue')
                plt.plot(x.cpu(), y_pred.cpu(), label=f'Approximation (Epoch {epoch + 1})', color='red')
                plt.legend()
                plt.title("Universal Approximation Theorem")
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.grid()
                plt.tight_layout()
                frame_path = f"assets/frame_{epoch + 1}.png"
                plt.savefig(frame_path)
                frames.append(frame_path)
                plt.close()
    return frames

if __name__ == "__main__":
    # Load dataset
    data = np.load("data/dataset.npz")
    x = torch.tensor(data['x'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32)

    # Build and train model
    model = build_model()
    frames = train_model(model, x, y)

    print("Frames saved for visualization.")