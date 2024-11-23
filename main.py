import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Mask_Maker import MaskMaker
from parameters import Parameters
import tkinter as tk
from tkinter import filedialog   

def load_prior_mask(file_path):
    """Load an image file and convert it to a normalized grayscale mask."""
    try:
        img = Image.open(file_path)
        if img.mode == 'I;16':
            mask = np.array(img, dtype=np.float32) / 65535.0  # Normalize 16-bit to [0,1]
        elif img.mode == 'L':
            mask = np.array(img, dtype=np.float32) / 255.0    # Normalize 8-bit to [0,1]
        else:
            img = img.convert('L')
            mask = np.array(img, dtype=np.float32) / 255.0    # Normalize other modes to [0,1]
        return mask
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def select_prior_images():
    """Open Mac Finder to select prior image files."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select Prior Image Files",
        filetypes=[("Image Files", "*.png *.jpeg *.jpg *.tiff *.tif")]
    )
    return list(file_paths)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test MaskMaker with various priors.")
    # Removed prior_images argument
    # parser.add_argument('--prior_images', nargs='*', help='Paths to prior image files (png, jpeg, tiff)', default=[])
    args = parser.parse_args()

    # Load parameters
    params = Parameters()
    size_x_pixels = params.canvas_size_pixels
    size_y_pixels = params.canvas_size_pixels
    size_x_mm = params.canvas_size_mm
    size_y_mm = params.canvas_size_mm

    # Load prior masks
    # Removed prior masks loading
    # prior_masks = []
    # for img_path in args.prior_images:
    #     mask = load_prior_mask(img_path)
    #     if mask is not None:
    #         prior_masks.append(mask)

    # Option to select prior images via Mac Finder
    prior_images = select_prior_images()
    prior_masks = []
    for img_path in prior_images:
        mask = load_prior_mask(img_path)
        if mask is not None:
            prior_masks.append(mask)

    # Initialize MaskMaker
    # Removed initialization with prior masks
    mask_maker = MaskMaker(size_x_pixels, size_y_pixels, size_x_mm, size_y_mm)
    
    # Generate masks
    mask_types = ['random_real', 'random_complex', 'combination']
    masks = {}
    for mtype in mask_types:
        getattr(mask_maker, mtype)()
        masks[mtype] = mask_maker.mask.copy()

    # Create central disks masks
    sizes = [0.1, 0.5, 1.0]  # diameters in mm
    opacities = [0.8] * 3  # example opacities
    central_disks_masks = []
    for size, opacity in zip(sizes, opacities):
        mask_maker.central_disks([size], [opacity])
        central_disks_masks.append(mask_maker.mask.copy())

    # Plot masks
    num_masks = len(masks) + len(central_disks_masks) + len(prior_masks)
    plt.figure(figsize=(15, 15), constrained_layout=True)  # Adjusted figure size for 3x3 grid

    # Plot random and combination masks
    idx = 1
    for mtype, mask in masks.items():
        ax = plt.subplot(3, 3, idx)
        im = ax.imshow(np.abs(mask), cmap='gray')
        ax.set_title(mtype)
        ax.set_xlabel('Millimeters')
        ax.set_ylabel('Millimeters')
        ax.set_aspect('equal')  # Maintain aspect ratio

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')
        idx += 1

    # Plot central disk masks
    for disk_mask, size in zip(central_disks_masks, sizes):
        ax = plt.subplot(3, 3, idx)
        im = ax.imshow(np.abs(disk_mask), cmap='gray')
        ax.set_title(f'Central Disk {size} mm')
        ax.set_xlabel('Millimeters')
        ax.set_ylabel('Millimeters')
        ax.set_aspect('equal')  # Maintain aspect ratio

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')
        idx += 1

    # Plot prior raw and prior mask
    for prior_mask in prior_masks:
        ax = plt.subplot(3, 3, idx)
        im = ax.imshow(np.abs(prior_mask), cmap='gray')
        ax.set_title('Prior Mask')
        ax.set_xlabel('Millimeters')
        ax.set_ylabel('Millimeters')
        ax.set_aspect('equal')  # Maintain aspect ratio

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')
        idx += 1

    plt.show()

if __name__ == "__main__":
    main()