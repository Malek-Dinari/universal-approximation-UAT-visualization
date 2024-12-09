import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(i, k, t, knots):
    """Computes the B-spline basis function for the i-th interval at the knot vector t."""
    if k == 1:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        den1 = knots[i + k - 1] - knots[i]
        den2 = knots[i + k] - knots[i + 1]
        
        term1 = (t - knots[i]) / den1 * bspline_basis(i, k - 1, t, knots) if den1 != 0 else 0
        term2 = (knots[i + k] - t) / den2 * bspline_basis(i + 1, k - 1, t, knots) if den2 != 0 else 0
        
        return term1 + term2

def b_spline_interpolation(x, control_points, degree=3):
    """Interpolate using B-Splines for given control points."""
    n = len(control_points)
    m = n + degree + 1  # Number of knots
    knots = np.concatenate(([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree))
    
    y = np.zeros_like(x)
    for i in range(n):
        for j in range(len(x)):
            y[j] += control_points[i] * bspline_basis(i, degree, x[j], knots)
    
    return y
