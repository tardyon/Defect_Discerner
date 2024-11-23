from inversemain import Propagators, Config  # Re-import Config
import numpy as np
import tifffile as tiff
from tkinter import filedialog, Tk  # Add import
import matplotlib.pyplot as plt  # Add import for matplotlib

# Configuration Parameters
WAVELENGTH = 1000.0  # Wavelength in nanometers
DISTANCE = 0.0       # Propagation distance in millimeters
PIXEL_SIZE = 40.0    # Pixels per millimeter
PROPAGATOR_TYPE = 'Fresnel'  # 'Fresnel' or 'AngularSpectrum'
PINHOLE_RADIUS = 2.0  # Pinhole radius in cycles per millimeter


def test_propagators():
    # Initialize Propagators
    config = Config(
        wavelength=WAVELENGTH,
        distance=DISTANCE,
        pixel_size=PIXEL_SIZE,
        propagator_type=PROPAGATOR_TYPE,
        pinhole_radius=PINHOLE_RADIUS
    )
    propagators = Propagators(config)
    
    # Load mask image using file dialog
    root = Tk()
    root.withdraw()
    mask_path = filedialog.askopenfilename(
        title="Select Mask File",
        filetypes=[("TIFF files", "*.tiff *.tif")]  # Changed semicolon to space
    )
    root.destroy()
    if not mask_path:
        print("No mask file selected.")
        return
    # Normalize the mask image
    mask = tiff.imread(mask_path).astype(np.float32)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    # Perform propagation
    intensity = propagators.propagate(mask)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Propagated Intensity')
    plt.imshow(intensity, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_propagators()