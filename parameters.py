# parameters.py

from dataclasses import dataclass

@dataclass
class FresnelParameters:
    """
    Encapsulates all parameters for Fresnel and Angular Spectrum propagation.
    
    Handles unit conversions between pixels and physical dimensions to ensure
    consistency across different propagation models and simulation setups.

    Attributes:
        wavelength_um (float): Wavelength in microns (default: 0.5 µm).
        z_mm (float): Propagation distance in millimeters (default: 50 mm).
        output_type (str): Type of output desired, either 'intensity' or 'complex_field' (default: 'intensity').
        padding (bool): Whether to use padding to optimize FFT computations (default: True).
        pad_factor (int): Factor by which to pad the canvas size (default: 2).
        use_edge_rolloff (bool): Whether to apply an edge roll-off function to minimize artifacts (default: False).
        canvas_size_pixels (int): Original canvas size in pixels (default: 512).
        canvas_size_mm (float): Physical size of the canvas in millimeters (default: 10 mm).
        pinhole_radius_inv_mm (float): Pinhole radius specified in cycles per millimeter (default: 2.0 cycles/mm).
        delta_mm (float): Parameter for the edge roll-off function in millimeters (default: 0.01 mm).
        propagation_model (str): Propagation model to use, either 'fresnel' or 'angular_spectrum' (default: 'fresnel').
    """

    wavelength_um: float = 0.5            # Wavelength in microns (default: 0.5 µm)
    z_mm: float = 50.0                    # Propagation distance in millimeters (default: 50 mm)
    output_type: str = 'intensity'        # 'intensity' or 'complex_field' (default: 'intensity')
    padding: bool = True                  # Whether to use padding to optimize FFT (default: True)
    pad_factor: int = 2                   # Padding factor (default: 2)
    use_edge_rolloff: bool = False        # Whether to apply edge roll-off using erf (default: False)
    canvas_size_pixels: int = 512         # Original canvas size in pixels (default: 512)
    canvas_size_mm: float = 10.0          # Physical size of the canvas in millimeters (default: 10 mm)
    pinhole_radius_inv_mm: float = 2.0    # Pinhole radius as cycles per mm (default: 2.0 cycles/mm)
    delta_mm: float = 0.01                # Parameter for edge roll-off function in millimeters (default: 0.01 mm)
    propagation_model: str = 'fresnel'    # 'fresnel' or 'angular_spectrum' (default: 'fresnel')

    @property
    def scaling_mm_per_pixel(self) -> float:
        """
        Calculate the scaling factor from millimeters to pixels.

        Returns:
            float: Millimeters per pixel.
        """
        return self.canvas_size_mm / self.canvas_size_pixels

    @property
    def wavelength_m(self) -> float:
        """
        Convert wavelength from microns to meters.

        Returns:
            float: Wavelength in meters.
        """
        return self.wavelength_um * 1e-6

    @property
    def scaling_m_per_pixel(self) -> float:
        """
        Calculate the scaling factor from meters to pixels.

        Returns:
            float: Meters per pixel.
        """
        return self.scaling_mm_per_pixel / 1000.0

    @property
    def z_m_converted(self) -> float:
        """
        Convert propagation distance from millimeters to meters.

        Returns:
            float: Propagation distance in meters.
        """
        return self.z_mm / 1000.0

    @property
    def canvas_size_m(self) -> float:
        """
        Convert canvas size from millimeters to meters.

        Returns:
            float: Canvas size in meters.
        """
        return self.canvas_size_mm / 1000.0

    @property
    def delta_m_converted(self) -> float:
        """
        Convert delta parameter from millimeters to meters.

        Returns:
            float: Delta parameter in meters.
        """
        return self.delta_mm / 1000.0

    @property
    def wavelength_pixels(self) -> float:
        """
        Convert wavelength from microns to pixel units.

        Returns:
            float: Wavelength in pixels.
        """
        return self.wavelength_um / (self.scaling_mm_per_pixel * 1000.0)

    @property
    def z_pixels(self) -> float:
        """
        Convert propagation distance from millimeters to pixel units.

        Returns:
            float: Propagation distance in pixels.
        """
        return self.z_mm / self.scaling_mm_per_pixel

    @property
    def pinhole_radius_inv_pixels(self) -> float:
        """
        Convert pinhole radius from cycles per millimeter to pixel units.

        Returns:
            float: Pinhole radius in cycles per pixel.
        """
        return self.pinhole_radius_inv_mm * self.scaling_mm_per_pixel

    @property
    def delta_pixels(self) -> float:
        """
        Convert delta parameter from millimeters to pixel units.

        Returns:
            float: Delta parameter in pixels.
        """
        return self.delta_mm / self.scaling_mm_per_pixel
