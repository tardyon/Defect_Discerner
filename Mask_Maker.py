import numpy as np

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
            self.mask = prior_mask.astype(np.complex128)
            if not np.iscomplexobj(prior_mask):
                self.mask[self.mask.real < 0] = 0  # Normalize real prior masks to have no negative values
        else:
            self.mask = np.zeros((self.size_y_pixels, self.size_x_pixels), dtype=np.complex128)

    def random_real(self):
        """Initialize mask with random real values between 0 and 1."""
        self.mask = np.random.rand(self.size_y_pixels, self.size_x_pixels).astype(np.complex128)

    def random_complex(self):
        """Initialize mask with random complex numbers."""
        real_part = np.random.rand(self.size_y_pixels, self.size_x_pixels)
        imag_part = np.random.rand(self.size_y_pixels, self.size_x_pixels)
        self.mask = (real_part + 1j * imag_part).astype(np.complex128)

    def combination(self):
        """Initialize mask with a combination of random real and complex values."""
        real_mask = np.random.rand(self.size_y_pixels, self.size_x_pixels)
        imag_mask = np.random.rand(self.size_y_pixels, self.size_x_pixels)
        self.mask = (real_mask + 1j * imag_mask).astype(np.complex128)

    def central_disks(self, sizes, opacities):
        """
        Create a mask with central disks of specified sizes and opacities.

        Parameters:
            sizes (list of float): Diameters of the disks in millimeters.
            opacities (list of float): Opacities for each disk (values between 0 and 1).
        """
        # Create coordinate grids
        x = np.linspace(-self.size_x_mm / 2, self.size_x_mm / 2, self.size_x_pixels)
        y = np.linspace(-self.size_y_mm / 2, self.size_y_mm / 2, self.size_y_pixels)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        self.mask = np.ones((self.size_y_pixels, self.size_x_pixels), dtype=np.complex128)
        for diameter, opacity in zip(sizes, opacities):
            radius = diameter / 2
            disk = np.where(R <= radius, opacity, 1.0)
            self.mask *= disk

    def set_pixels(self, values, locations):
        """
        Set specific pixel values at given locations.

        Parameters:
            values (list of complex): Values to set.
            locations (list of tuple): Corresponding (x, y) pixel indices.
        """
        for value, (x_idx, y_idx) in zip(values, locations):
            if 0 <= x_idx < self.size_x_pixels and 0 <= y_idx < self.size_y_pixels:
                self.mask[y_idx, x_idx] = value
            else:
                raise IndexError(f"Pixel index ({x_idx}, {y_idx}) out of bounds.")
