
from dataclasses import dataclass

@dataclass
class Parameters:
    wavelength_pixels: float
    z_pixels: float
    canvas_size_pixels: int
    padding: bool
    pad_factor: int
    pinhole_radius_inv_pixels: float
    use_edge_rolloff: bool = False
    delta_pixels: float = 10.0  # Example default value