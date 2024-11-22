# FresnelPropagation.py

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.special import erf
from parameters import FresnelParameters

class FresnelPropagator:
    """
    Class for performing Fresnel propagation on a given mask array using encapsulated parameters.
    """
    def __init__(self, params: FresnelParameters):
        """
        Initialize the FresnelPropagator with specified parameters.

        Parameters:
            params (FresnelParameters): Encapsulated simulation parameters.
        """
        self.params = params
        self.FX = None          # Frequency grid X
        self.FY = None          # Frequency grid Y
        self.F_squared = None   # Frequency squared
        self.H = None           # Transfer function

        # Initialize computational grids and transfer function
        self._initialize_computational_grids()

    def _initialize_computational_grids(self):
        """
        Initialize frequency grids and transfer function based on parameters.
        All calculations are performed in pixel units.
        """
        # Extract parameters in pixel units
        wavelength_pixels = self.params.wavelength_pixels
        z_pixels = self.params.z_pixels  # Changed from 'z_m' to 'z_pixels'
        padding = self.params.padding
        pad_factor = self.params.pad_factor
        canvas_size_pixels = self.params.canvas_size_pixels
        pinhole_radius_inv_pixels = self.params.pinhole_radius_inv_pixels

        # Apply padding if enabled
        if padding:
            ny_padded = canvas_size_pixels * pad_factor
            nx_padded = canvas_size_pixels * pad_factor
        else:
            ny_padded = canvas_size_pixels
            nx_padded = canvas_size_pixels

        # Frequency grids in cycles per pixel
        fx = fftfreq(nx_padded, d=1)
        fy = fftfreq(ny_padded, d=1)
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.F_squared = self.FX**2 + self.FY**2

        # Transfer function based on selected propagation model
        if self.model == 'fresnel':
            self.H = np.exp(-1j * np.pi * wavelength_pixels * z_pixels * self.F_squared)
        elif self.model == 'angular_spectrum':
            k = 2 * np.pi / wavelength_pixels
            kz_squared = k**2 - (2 * np.pi * self.FX)**2 - (2 * np.pi * self.FY)**2
            kz = np.sqrt(kz_squared, where=kz_squared >= 0, out=np.zeros_like(kz_squared))
            self.H = np.exp(1j * kz * z_pixels)
        else:
            raise ValueError("Invalid propagation_model. Choose 'fresnel' or 'angular_spectrum'.")

        # Apply frequency filter (pinhole) if specified
        if pinhole_radius_inv_pixels > 0:
            self.H *= self.F_squared <= (pinhole_radius_inv_pixels)**2

    def _create_edge_rolloff(self) -> np.ndarray:
        """
        Create an edge roll-off function using the error function (erf).

        Returns:
            np.ndarray: Edge roll-off array.
        """
        size_pixels = self.params.canvas_size_pixels
        size_m = self.params.canvas_size_m
        delta_m = self.params.delta_m

        x = np.linspace(-size_m / 2, size_m / 2, size_pixels)
        roll_off = 0.5 * (erf((x + size_m / 2) / delta_m) - erf((x - size_m / 2) / delta_m))
        edge_rolloff = np.outer(roll_off, roll_off)
        return edge_rolloff.astype(np.complex64)

    def _pad_array(self, array: np.ndarray) -> np.ndarray:
        """
        Pad the array to a larger size to optimize FFT.

        Parameters:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: Padded array.
        """
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        pad_ny = ny * pad_factor
        pad_nx = nx * pad_factor
        padded_array = np.zeros((pad_ny, pad_nx), dtype=array.dtype)
        y_start = (pad_ny - ny) // 2
        x_start = (pad_nx - nx) // 2
        padded_array[y_start:y_start+ny, x_start:x_start+nx] = array
        return padded_array

    def _crop_array(self, array: np.ndarray) -> np.ndarray:
        """
        Crop the padded array back to the original size.

        Parameters:
            array (np.ndarray): Input array (padded).

        Returns:
            np.ndarray: Cropped array.
        """
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        orig_ny = ny // pad_factor
        orig_nx = nx // pad_factor
        y_start = (ny - orig_ny) // 2
        x_start = (nx - orig_nx) // 2
        cropped_array = array[y_start:y_start+orig_ny, x_start:x_start+orig_nx]
        return cropped_array

    def propagate(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Perform Fresnel propagation on the input mask array.

        Parameters:
            mask_array (np.ndarray): Input 2D array (can be complex64) representing the initial field.

        Returns:
            np.ndarray: Output array after Fresnel propagation, either intensity or complex field.
        """
        # Apply edge roll-off if enabled
        if self.params.use_edge_rolloff:
            edge_rolloff = self._create_edge_rolloff()
            U0 = mask_array * edge_rolloff
        else:
            U0 = mask_array.copy()

        # Apply padding if enabled
        if self.params.padding:
            U0 = self._pad_array(U0)

        # Perform the propagation using FFT
        U1_fft = fft2(U0)
        U2_fft = U1_fft * self.H
        U2 = ifft2(U2_fft)

        # Remove padding if it was applied
        if self.params.padding:
            U2 = self._crop_array(U2)

        # Select output type
        if self.params.output_type == 'intensity':
            intensity = np.abs(U2)**2
            return intensity.astype(np.float32)
        elif self.params.output_type == 'complex_field':
            return U2.astype(np.complex64)
        else:
            raise ValueError("Invalid output_type. Choose 'intensity' or 'complex_field'.")
