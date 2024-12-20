import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy.special import erf
from parameters import Parameters

class Propagation:
    """
    Class for performing Fresnel and Angular Spectrum propagation on a given mask array using encapsulated parameters.

    Attributes:
        params (Parameters): Encapsulated simulation parameters.
        FX (np.ndarray): Frequency grid in the X-direction.
        FY (np.ndarray): Frequency grid in the Y-direction.
        F_squared (np.ndarray): Squared frequency grid.
        H (np.ndarray): Transfer function based on the selected propagation model.
        model (str): Selected propagation model ('fresnel' or 'angular_spectrum').
    """
    def __init__(self, params: Parameters):
        """
        Initialize the Propagation class with specified parameters.

        Parameters:
            params (Parameters): Encapsulated simulation parameters.
        """
        self.params = params
        self.FX = None
        self.FY = None
        self.F_squared = None
        self.H = None
        self.model = params.propagation_model  # Select propagation model

        # Initialize computational grids and transfer function
        self._initialize_computational_grids()

    def _initialize_computational_grids(self):
        """
        Initialize frequency grids and transfer function based on parameters.

        All calculations are performed in pixel units.

        This method sets up the necessary grids and the transfer function
        required for the propagation based on the selected model.
        """
        # Extract parameters in pixel units
        wavelength_pixels = self.params.wavelength_pixels
        z_pixels = self.params.z_pixels
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
        fx = fftfreq(nx_padded)
        fy = fftfreq(ny_padded)
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.F_squared = (self.FX)**2 + (self.FY)**2

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
        Create an edge roll-off function in pixel units.

        This function generates a smooth transition at the edges of the aperture
        to minimize artifacts in the propagation.

        Returns:
            np.ndarray: Edge roll-off array.
        """
        # Determine the size based on padding
        if self.params.padding:
            size_pixels = self.params.canvas_size_pixels * self.params.pad_factor
        else:
            size_pixels = self.params.canvas_size_pixels

        delta_pixels = self.params.delta_pixels

        x = np.linspace(-size_pixels / 2, size_pixels / 2, size_pixels)
        roll_off = 0.5 * (erf((x + size_pixels / 2) / delta_pixels) - erf((x - size_pixels / 2) / delta_pixels))
        edge_rolloff = np.outer(roll_off, roll_off)
        return edge_rolloff.astype(np.complex64)

    def _pad_array(self, array: np.ndarray) -> np.ndarray:
        """
        Pad the array to a larger size to optimize FFT.

        Parameters:
            array (np.ndarray): Input array.

        Returns:
            np.ndarray: Padded array with ones.
        """
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        pad_ny = ny * pad_factor
        pad_nx = nx * pad_factor
        padded_array = np.ones((pad_ny, pad_nx), dtype=array.dtype)  # Changed from zeros to ones
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
        Perform Fresnel or Angular Spectrum propagation on the input mask array.

        Parameters:
            mask_array (np.ndarray): Input 2D array (can be complex64) representing the initial field.

        Returns:
            np.ndarray: Output array after propagation.
                        - If output_type is 'intensity', returns the intensity as float32.
                        - If output_type is 'complex_field', returns the complex field as complex64.
        """
        # Apply padding if enabled
        if self.params.padding:
            U0 = self._pad_array(mask_array)
        else:
            U0 = mask_array.copy()

        # Apply edge roll-off if enabled
        if self.params.use_edge_rolloff:
            edge_rolloff = self._create_edge_rolloff()
            U0 = U0 * edge_rolloff  # Ensure shapes match

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
            return intensity.astype(np.float32)  # Removed duplicated return
        elif self.params.output_type == 'complex_field':
            return U2.astype(np.complex64)
        else:
            raise ValueError("Invalid output_type. Choose 'intensity' or 'complex_field'.")