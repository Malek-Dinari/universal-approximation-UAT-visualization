import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the target function f(x) = sin(2 * pi * x)
def target_function(x):
    return np.sin(2 * np.pi * x)

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

# Set parameters
n_knots = 8    # Number of knots (including boundary knots)
degree = 3     # Degree of the B-spline (cubic splines)
n_control_points = n_knots - degree - 1  # Number of control points
epochs = 200   # Number of epochs for animation
x_vals = np.linspace(0, 1, 100)  # Points where the function is evaluated
target_vals = target_function(x_vals)  # Target function values

# Generate knot vector (uniform spacing)
knots = np.linspace(0, 1, n_knots)

import matplotlib
matplotlib.use('TkAgg')  # Ensure correct backend for Tkinter (interactive)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define any other needed functions, such as b_spline_interpolation() and target_vals

# Initialize the control_points if not already initialized
control_points = np.zeros(5)  # Example initialization
target_vals = np.sin(2 * np.pi * np.linspace(0, 1, 100))  # Example target values
n_control_points = 5
degree = 3
knots = np.linspace(0, 1, n_control_points + degree + 1)  # Example knots

# Create figure and axes
fig, ax = plt.subplots()
line_target, = ax.plot([], [], label=r'Target Function $f(x) = \sin(2\pi x)$', color='black')
line_control, = ax.plot([], [], label='Control Points', color='red')
line_approx, = ax.plot([], [], label='Approximated Function', color='blue')

ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
ax.legend()

# Update function for the animation
def update(frame):
    global control_points  # Use global control_points variable
    control_points += 0.01 * (target_vals[:n_control_points] - b_spline_interpolation(np.linspace(0, 1, n_control_points), control_points, degree, knots))
    
    line_target.set_data(np.linspace(0, 1, 100), target_vals)
    line_control.set_data(np.linspace(0, 1, n_control_points), control_points)
    line_approx.set_data(np.linspace(0, 1, 100), b_spline_interpolation(np.linspace(0, 1, 100), control_points, degree, knots))
    
    return line_target, line_control, line_approx

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

plt.show()

