# main.py

import numpy as np
import matplotlib.pyplot as plt
from parameters import Parameters
from Propagation import Propagation  # Updated import

def create_circular_aperture(size_pixels: int, radius_pixels: float) -> np.ndarray:
    """
    Create an occlusive circular aperture mask.

    Parameters:
        size_pixels (int): Size of the mask in pixels (assumed square).
        radius_pixels (float): Radius of the aperture in pixels.

    Returns:
        np.ndarray: Occlusive circular aperture mask.
    """
    Y, X = np.ogrid[:size_pixels, :size_pixels]
    center = size_pixels // 2
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = np.ones((size_pixels, size_pixels), dtype=np.complex64)
    mask[dist_from_center <= radius_pixels] = 0.0  # Occlusive mask
    return mask

def main():
    """
    Main function to perform Fresnel and Angular Spectrum propagation on a circular aperture mask.

    This function sets up the simulation parameters, creates an occlusive circular aperture mask,
    performs propagation using both Fresnel and Angular Spectrum methods, and visualizes the results.
    
    The problem it addresses is simulating and comparing different optical propagation models
    to understand the behavior of light passing through an aperture.
    """
    # Instantiate base parameters with default values
    base_params = Parameters(
        wavelength_um=1.0,                  # 1.0 µm wavelength
        z_mm=1000.0,                        # 1000 mm propagation distance
        output_type='intensity',            # Output intensity
        padding=True,                       # Enable padding
        pad_factor=2,                       # Padding factor
        use_edge_rolloff=False,             # Disable edge roll-off
        canvas_size_pixels=1024,            # 1024x1024 pixels
        canvas_size_mm=100.0,                # 100 mm canvas size
        pinhole_radius_inv_mm=2.0,           # 2 cycles/mm pinhole radius
        delta_mm=1,                         # 1 mm delta for edge roll-off
    )


    # Convert aperture radius from millimeters to pixels using updated scaling
    aperture_radius_mm = 0.1  # Aperture radius in millimeters
    aperture_radius_pixels = aperture_radius_mm / base_params.scaling_mm_per_pixel

    # Create parameters for Fresnel propagation
    params_fresnel = base_params.__dict__.copy()
    params_fresnel['propagation_model'] = 'fresnel'
    params_fresnel = Parameters(**params_fresnel)
    
    # Create parameters for Angular Spectrum propagation
    params_angular = base_params.__dict__.copy()
    params_angular['propagation_model'] = 'angular_spectrum'
    params_angular = Parameters(**params_angular)


    # Create an occlusive circular aperture mask
    mask_array = create_circular_aperture(
        size_pixels=base_params.canvas_size_pixels,
        radius_pixels=aperture_radius_pixels
    )

    # Instantiate the propagators with the parameters
    propagator_fresnel = Propagation(params_fresnel)  # Using 'fresnel' model
    propagator_angular = Propagation(params_angular)  # Using 'angular_spectrum' model

    # Perform Fresnel propagation
    propagated_intensity_fresnel = propagator_fresnel.propagate(mask_array)

    # Perform Angular Spectrum propagation
    propagated_intensity_angular = propagator_angular.propagate(mask_array)

    # Visualize the results
    plt.figure(figsize=(24, 6))

    # Show the mask
    plt.subplot(1, 4, 1)
    extent_mm = params_fresnel.canvas_size_mm  # Already in millimeters
    plt.imshow(
        np.abs(mask_array),
        cmap='gray',
        extent=[-extent_mm/2, extent_mm/2, -extent_mm/2, extent_mm/2]
    )
    plt.title('Occlusive Circular Aperture Mask')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Amplitude')

    # Show the propagated intensity using Fresnel
    plt.subplot(1, 4, 2)
    plt.imshow(
        propagated_intensity_fresnel,
        cmap='gray',
        extent=[-extent_mm/2, extent_mm/2, -extent_mm/2, extent_mm/2]
    )
    plt.title('Fresnel Propagation Intensity')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Intensity')

    # Show the propagated intensity using Angular Spectrum
    plt.subplot(1, 4, 3)
    plt.imshow(
        propagated_intensity_angular,
        cmap='gray',
        extent=[-extent_mm/2, extent_mm/2, -extent_mm/2, extent_mm/2]
    )
    plt.title('Angular Spectrum Propagation Intensity')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar(label='Intensity')

    # Show the cross-sectional lineouts for both methods
    plt.subplot(1, 4, 4)
    center_index = params_fresnel.canvas_size_pixels // 2
    lineout_fresnel = propagated_intensity_fresnel[center_index, :]
    lineout_angular = propagated_intensity_angular[center_index, :]
    plt.plot(
        np.linspace(-extent_mm/2, extent_mm/2, params_fresnel.canvas_size_pixels),
        lineout_fresnel,
        label='Fresnel'
    )
    plt.plot(
        np.linspace(-extent_mm/2, extent_mm/2, params_fresnel.canvas_size_pixels),
        lineout_angular,
        label='Angular Spectrum',
        linestyle='--'
    )
    plt.title('Cross-Sectional Lineout')
    plt.xlabel('X (mm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.ylim(0, 1.1 * max(np.max(lineout_fresnel), np.max(lineout_angular)))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
