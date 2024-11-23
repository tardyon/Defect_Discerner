import numpy as np

def huber_loss(I_observed, I_estimated, delta=1.0):
    """
    Compute the Huber loss between the observed and estimated intensity images.
    
    Parameters:
        I_observed (np.ndarray): Observed intensity image.
        I_estimated (np.ndarray): Estimated intensity image from the forward model.
        delta (float): Threshold parameter for the Huber loss (default: 1.0).
    
    Returns:
        float: The Huber loss value.
    """
    difference = I_estimated - I_observed
    abs_diff = np.abs(difference)
    mask = abs_diff <= delta
    quadratic_loss = 0.5 * (difference[mask]) ** 2
    linear_loss = delta * (abs_diff[~mask] - 0.5 * delta)
    loss = np.sum(quadratic_loss) + np.sum(linear_loss)
    return loss