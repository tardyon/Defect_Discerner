import numpy as np
from scipy.ndimage import zoom

class MaskMaker:
    """
    Class for generating masks for the simulation.

    Parameters:
        size_x_pixels (int): Size of the mask in pixels along the X-axis.
        size_y_pixels (int): Size of the mask in pixels along the Y-axis.
        size_x_mm (float): Physical size of the mask in millimeters along the X-axis.
        size_y_mm (float): Physical size of the mask in millimeters along the Y-axis.
        prior_mask (np.ndarray, optional): Existing mask array to initialize from.
    """
    def __init__(self, size_x_pixels, size_y_pixels, size_x_mm, size_y_mm, prior_mask=None):
        self.size_x_pixels = size_x_pixels
        self.size_y_pixels = size_y_pixels
        self.size_x_mm = size_x_mm
        self.size_y_mm = size_y_mm
        if prior_mask is not None:
            self.mask = prior_mask.astype(np.float32)
            self.mask = np.clip(self.mask, 0, 1)  # Ensure values between 0 and 1
        else:
            self.mask = np.zeros((self.size_y_pixels, self.size_x_pixels), dtype=np.float32)

    def random_real(self):
        """Initialize mask with random real values between 0 and 1."""
        self.mask = np.random.rand(self.size_y_pixels, self.size_x_pixels).astype(np.float32)
        
    # Remove combination method - no longer needed
    
    def set_pixels(self, values, locations):
        """
        Set specific pixel values at given locations.

        Parameters:
            values (list/array): Values between 0 and 1 to set at each location
            locations (list/array): List of (x,y) integer pixel coordinates
        """
        values = np.asarray(values)
        locations = np.asarray(locations, dtype=np.int32)  # Force integer indices
        
        # Ensure coordinates are within bounds
        mask = ((0 <= locations[:, 0]) & (locations[:, 0] < self.size_x_pixels) & 
                (0 <= locations[:, 1]) & (locations[:, 1] < self.size_y_pixels))
        
        # Only set valid pixel locations
        valid_values = values[mask]
        valid_locations = locations[mask]
        
        for val, (x, y) in zip(valid_values, valid_locations):
            self.mask[y, x] = val

    def add_disk(self, center_x, center_y, diameter, opacity=0.0):
        """
        Add a disk to the mask at specified location.

        Parameters:
            center_x (float): X coordinate of disk center in pixels
            center_y (float): Y coordinate of disk center in pixels
            diameter (float): Diameter of disk in pixels
            opacity (float): Opacity value between 0 and 1 (default 0.0, fully opaque)

        Returns:
            bool: True if disk was fully within bounds, False if partially or fully out of bounds
        """
        radius = diameter / 2
        
        # Check if disk would be fully out of bounds
        if (center_x + radius < 0 or center_x - radius > self.size_x_pixels or
            center_y + radius < 0 or center_y - radius > self.size_y_pixels):
            return False

        # Create coordinate grid for this disk
        y, x = np.ogrid[:self.size_y_pixels, :self.size_x_pixels]
        disk_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        
        # Check if disk is partially out of bounds
        fully_within_bounds = (
            center_x - radius >= 0 and 
            center_x + radius < self.size_x_pixels and
            center_y - radius >= 0 and 
            center_y + radius < self.size_y_pixels
        )
        
        # Apply disk to mask
        self.mask[disk_mask] = opacity
        
        return fully_within_bounds

    def central_disk(self, diameter_fraction=0.2, opacity=0.0):
        """Initialize mask with a central disk.
        
        Parameters:
            diameter_fraction (float): Fraction of the smallest dimension to use as diameter
            opacity (float): Opacity of the disk (0.0 is fully opaque, 1.0 is fully transparent)
        """
        min_dimension = min(self.size_x_pixels, self.size_y_pixels)
        diameter = min_dimension * diameter_fraction
        center_x = self.size_x_pixels // 2
        center_y = self.size_y_pixels // 2
        
        # Initialize with fully transparent background
        self.mask = np.ones((self.size_y_pixels, self.size_x_pixels), dtype=np.float32)
        # Add the disk
        self.add_disk(center_x, center_y, diameter, opacity)

    def resize_mask(self, new_size):
        """
        Resize the mask to the specified new size.

        Parameters:
            new_size (tuple): New size as (height, width).
        """
        zoom_factors = (new_size[0] / self.mask.shape[0], new_size[1] / self.mask.shape[1])
        self.mask = zoom(self.mask, zoom_factors, order=1).astype(np.float32)
