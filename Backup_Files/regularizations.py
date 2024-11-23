import numpy as np

def total_variation(M, weight=0.1):
    """
    Compute the total variation regularization term.
    
    Parameters:
        M (np.ndarray): Mask array.
        weight (float): Regularization weight.
    
    Returns:
        np.ndarray: Gradient of the total variation term.
    """
    grad_M = np.zeros_like(M)
    grad_x = np.roll(M, -1, axis=1) - M
    grad_y = np.roll(M, -1, axis=0) - M
    grad_M += np.roll(grad_x, 1, axis=1) - grad_x + np.roll(grad_y, 1, axis=0) - grad_y
    return weight * grad_M

def shape_bias(M, ellipse_params, weight=0.5):
    """
    Compute the shape bias regularization term.
    
    Parameters:
        M (np.ndarray): Mask array.
        ellipse_params (tuple): Parameters of the desired ellipse (cx, cy, a, b, theta).
        weight (float): Regularization weight.
    
    Returns:
        np.ndarray: Gradient of the shape bias term.
    """
    cx, cy, a, b, theta = ellipse_params
    y, x = np.indices(M.shape)
    x_rot = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
    y_rot = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
    ellipse = ((x_rot / a) ** 2 + (y_rot / b) ** 2) <= 1
    grad = 2 * (M - ellipse.astype(np.float32))
    return weight * grad