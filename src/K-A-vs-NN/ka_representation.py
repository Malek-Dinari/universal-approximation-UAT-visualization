import numpy as np
import matplotlib.pyplot as plt

# Define the target function
def f_target(x, y):
    return np.sin(x + y)

# Initialize parameters for psi and phi
n = 2  # Number of input dimensions
m = 5  # Number of intermediate functions
psi_params = np.random.rand(n, m)  # Coefficients for psi
phi_params = np.random.rand(m)     # Coefficients for phi

# Define psi and phi
def psi(x, params):
    return np.dot(x, params)

def phi(s, params):
    return np.sum(params * np.sin(s))

# Approximate f(x, y) using Kolmogorov-Arnold representation
def f_approx(x, y, psi_params, phi_params):
    inputs = np.array([x, y])
    intermediate = psi(inputs, psi_params)
    return phi(intermediate, phi_params)

# Generate data
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z_target = f_target(X, Y)

# Approximation
Z_approx = np.zeros_like(Z_target)
for i in range(Z_approx.shape[0]):
    for j in range(Z_approx.shape[1]):
        Z_approx[i, j] = f_approx(X[i, j], Y[i, j], psi_params, phi_params)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Target Function")
plt.contourf(X, Y, Z_target, cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("K-A Approximation")
plt.contourf(X, Y, Z_approx, cmap='viridis')
plt.colorbar()

plt.show()
