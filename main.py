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
    root.destroy()  # Properly close the Finder dialog box
    return list(file_paths)

def create_random_disk_mask(mask_maker, num_disks, min_diameter, max_diameter, base_mask=None):
    """
    Create a mask with random disks, optionally overlaying on a base mask.
    
    Returns:
        tuple: (mask, num_valid_disks, num_invalid_disks)
    """
    if base_mask is not None:
        mask_maker.mask = base_mask.copy()
        mask_maker.resize_mask((mask_maker.size_y_pixels, mask_maker.size_x_pixels))  # Ensure correct size
    
    valid_disks = 0
    invalid_disks = 0
    
    for _ in range(num_disks):
        diameter = np.random.uniform(min_diameter, max_diameter)
        center_x = np.random.uniform(diameter/2, mask_maker.size_x_pixels - diameter/2)
        center_y = np.random.uniform(diameter/2, mask_maker.size_y_pixels - diameter/2)
        opacity = np.random.uniform(0, 1)
        
        if mask_maker.add_disk(center_x, center_y, diameter, opacity):
            valid_disks += 1
        else:
            invalid_disks += 1
    
    return mask_maker.mask.copy(), valid_disks, invalid_disks

def create_central_disk_mask(mask_maker, diameter_mm, opacity=0.8):
    """Create a mask with a central disk of specified diameter."""
    # Convert diameter from mm to pixels
    diameter_pixels = diameter_mm * (mask_maker.size_x_pixels / mask_maker.size_x_mm)
    
    # Calculate center coordinates
    center_x = mask_maker.size_x_pixels / 2
    center_y = mask_maker.size_y_pixels / 2
    
    # Create new mask and add disk
    mask_maker.mask = np.ones((mask_maker.size_y_pixels, mask_maker.size_x_pixels), dtype=np.complex128)
    mask_maker.add_disk(center_x, center_y, diameter_pixels, opacity)
    return mask_maker.mask.copy()

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
    opacities = [0.8] * 3    # example opacities
    central_disks_masks = []
    for size in sizes:
        mask_maker = MaskMaker(size_x_pixels, size_y_pixels, size_x_mm, size_y_mm)
        central_disks_masks.append(create_central_disk_mask(mask_maker, size))

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

    plt.show(block=False)  # Show the first figure without blocking

    # Test set_pixels by modifying the user-selected mask
    if prior_masks:
        modified_masks = []
        num_pixels_to_set = [10, 50, 100]
        for num_pixels in num_pixels_to_set:
            # Print array dimensions for debugging
            print(f"Prior mask shape: {prior_masks[0].shape}")
            
            modified_mask_maker = MaskMaker(size_x_pixels, size_y_pixels, size_x_mm, size_y_mm, prior_masks[0].copy())
            values = np.random.rand(num_pixels)
            
            # Explicitly ensure coordinates are within array bounds
            array_height, array_width = prior_masks[0].shape
            x_coords = np.random.randint(0, array_width, num_pixels)
            y_coords = np.random.randint(0, array_height, num_pixels)
            
            # Create locations array and verify bounds
            locations = np.stack((x_coords, y_coords), axis=1)
            print(f"Max x coord: {x_coords.max()}, Max y coord: {y_coords.max()}")
            print(f"Array dimensions: {array_width}x{array_height}")
            
            modified_mask_maker.set_pixels(values, locations)
            modified_masks.append(modified_mask_maker.mask.copy())

        # Plot modified masks
        plt.figure(figsize=(15, 15), constrained_layout=True)  # Adjusted figure size for 3x3 grid
        idx = 1
        for num_pixels, modified_mask in zip(num_pixels_to_set, modified_masks):
            ax = plt.subplot(3, 3, idx)
            im = ax.imshow(np.abs(modified_mask), cmap='gray')
            ax.set_title(f'Modified Mask with {num_pixels} Pixels')
            ax.set_xlabel('Millimeters')
            ax.set_ylabel('Millimeters')
            ax.set_aspect('equal')  # Maintain aspect ratio

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Intensity')
            idx += 1

        plt.show()

    # Create random disk masks
    random_disk_masks = []
    valid_counts = []
    invalid_counts = []
    disk_counts = [10, 50, 100]
    for num_disks in disk_counts:
        disk_maker = MaskMaker(size_x_pixels, size_y_pixels, size_x_mm, size_y_mm)
        mask, valid, invalid = create_random_disk_mask(
            disk_maker, num_disks, min_diameter=3, max_diameter=30
        )
        random_disk_masks.append(mask)
        valid_counts.append(valid)
        invalid_counts.append(invalid)

    # Plot random disk masks
    plt.figure(figsize=(15, 5), constrained_layout=True)
    for idx, (num_disks, disk_mask, valid, invalid) in enumerate(zip(disk_counts, random_disk_masks, valid_counts, invalid_counts), 1):
        ax = plt.subplot(1, 3, idx)
        im = ax.imshow(np.abs(disk_mask), cmap='gray')
        ax.set_title(f'{num_disks} Random Disks\n({valid} valid, {invalid} invalid)')
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Opacity')

    # Create and display all figures
    plt.ion()  # Turn on interactive mode
    
    # First figure (existing masks)
    fig1 = plt.figure(figsize=(15, 15))
    # ... existing plotting code for first figure ...
    
    # Second figure (modified masks)
    if prior_masks:
        fig2 = plt.figure(figsize=(15, 15))
        # ... existing plotting code for second figure ...
    
    # Third figure (random disks)
    fig3 = plt.figure(figsize=(15, 5))
    disk_counts = [10, 50, 100]
    base_mask = prior_masks[0].copy() if prior_masks else None
    
    for idx, num_disks in enumerate(disk_counts, 1):
        disk_maker = MaskMaker(size_x_pixels, size_y_pixels, size_x_mm, size_y_mm)
        mask, valid, invalid = create_random_disk_mask(
            disk_maker, num_disks, min_diameter=3, max_diameter=30, base_mask=base_mask
        )
        
        ax = plt.subplot(1, 3, idx)
        im = ax.imshow(np.abs(mask), cmap='gray')
        ax.set_title(f'{num_disks} Disks\n({valid} valid, {invalid} invalid)')
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Opacity')
    
    # Display all figures
    plt.show()
    input("Press Enter to close all figures...")  # Keep figures open until user input

if __name__ == "__main__":
    main()