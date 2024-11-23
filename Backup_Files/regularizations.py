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