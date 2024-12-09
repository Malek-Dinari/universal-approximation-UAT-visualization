import torch.nn as nn

def build_model():
    """Builds a simple neural network for UAT demonstration."""
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )
    return model