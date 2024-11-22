# parameters.py

from dataclasses import dataclass

@dataclass
class FresnelParameters:
    """
    Encapsulates all parameters for Fresnel and Angular Spectrum propagation.
    Unit conversions between pixels and physical dimensions are handled here.
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
        return self.canvas_size_mm / self.canvas_size_pixels

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_um * 1e-6

    @property
    def scaling_m_per_pixel(self) -> float:
        return self.scaling_mm_per_pixel / 1000.0

    @property
    def z_m_converted(self) -> float:
        return self.z_mm / 1000.0

    @property
    def canvas_size_m(self) -> float:
        return self.canvas_size_mm / 1000.0

    @property
    def delta_m_converted(self) -> float:
        return self.delta_mm / 1000.0

    @property
    def wavelength_pixels(self) -> float:
        return self.wavelength_um / (self.scaling_mm_per_pixel * 1000.0)

    @property
    def z_pixels(self) -> float:
        return self.z_mm / self.scaling_mm_per_pixel

    @property
    def pinhole_radius_inv_pixels(self) -> float:
        return self.pinhole_radius_inv_mm * self.scaling_mm_per_pixel

    @property
    def delta_pixels(self) -> float:
        return self.delta_mm / self.scaling_mm_per_pixel
