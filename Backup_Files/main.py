import numpy as np
from parameters import Parameters
from Mask_Maker import MaskMaker
from Propagation import Propagation
from loss_functions import huber_loss
from regularizations import total_variation
from solvers import ADMMSolver  # Only import ADMMSolver
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from inverseParameters import InverseParameters
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
from tkinter import messagebox
import tifffile as tiff
from tqdm import tqdm  # Add import for progress bars

def select_file(title, filetypes):
    """Simplified file selection dialog."""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    # Don't use topmost attribute
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[(desc, f"*{ext}") for desc, ext in filetypes]
    )
    root.quit()
    root.destroy()
    return file_path

def select_prior_type():
    """GUI dialog for selecting prior type."""
    root = tk.Tk()
    root.title("Select Prior Type")
    root.geometry("300x200")  # Adjusted height

    selected_type = tk.StringVar(value="random")

    def on_select():
        root.quit()

    tk.Label(root, text="Choose prior type:").pack(pady=10)
    tk.Radiobutton(root, text="Random", variable=selected_type, value="random").pack()
    tk.Radiobutton(root, text="Load from file", variable=selected_type, value="load").pack()
    tk.Radiobutton(root, text="Fully Transparent", variable=selected_type, value="transparent").pack()
    tk.Button(root, text="OK", command=on_select).pack(pady=20)

    root.mainloop()
    prior_type = selected_type.get()
    root.destroy()
    return prior_type

def main():
    # Initialize parameters
    params = Parameters()
    inverse_params = InverseParameters()
    
    # Modified file selection for observed image
    observed_file_path = select_file(
        "Select observed image file",
        [
            ("PNG", ".png"),
            ("JPEG", ".jpg"),
            ("JPEG", ".jpeg"),
            ("GIF", ".gif"),
            ("TIFF", ".tiff"),
            ("TIFF", ".tif")
        ]
    )
    
    if not observed_file_path:
        print("No observed image file selected.")
        return

    # Add ground truth mask selection
    ground_truth_path = select_file(
        "Select ground truth mask file (optional)",
        [
            ("PNG", ".png"),
            ("JPEG", ".jpg"),
            ("JPEG", ".jpeg"),
            ("GIF", ".gif"),
            ("TIFF", ".tiff"),
            ("TIFF", ".tif")
        ]
    )

    # Load the observed image
    original_image = tiff.imread(observed_file_path)

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path:
        ground_truth = tiff.imread(ground_truth_path)
        # Normalize to float32 [0,1] range if needed
        if ground_truth.dtype == np.uint16:
            ground_truth = ground_truth.astype(np.float32) / 65535.0
        elif ground_truth.dtype == np.uint8:
            ground_truth = ground_truth.astype(np.float32) / 255.0

    # Keep original_image as is for display (already in 16-bit format)
    # Create a normalized float32 version for processing
    image_array = original_image.astype(np.float32)
    print(f"Original image stats:")
    print(f"  Shape: {original_image.shape}")
    print(f"  Dtype: {original_image.dtype}")
    print(f"  Min: {original_image.min()}")
    print(f"  Max: {original_image.max()}")
    print(f"  Mean: {original_image.mean()}")

    # Normalize the processing array to [0,1] range
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    
    # Simulate observed intensity using the forward model
    propagation = Propagation(params)
    I_observed = np.clip(image_array, 0, 1)  # Ensure values between 0 and 1
    print(f"I_observed stats: min={I_observed.min()}, max={I_observed.max()}, mean={I_observed.mean()}")

    # Update canvas size parameters based on the image size and scaling factor
    params.canvas_size_pixels = I_observed.shape[1]
    params.canvas_size_mm = 10.0  # Adjust as needed based on scaling
    pixel_size_mm = params.canvas_size_mm / params.canvas_size_pixels  # Physical size per pixel

    # Replace command line input with GUI selection
    prior_type = select_prior_type()
    inverse_params.prior_type = prior_type

    # Initialize MaskMaker with prior
    mask_maker = MaskMaker(
        size_x_pixels=params.canvas_size_pixels,
        size_y_pixels=params.canvas_size_pixels,
        size_x_mm=params.canvas_size_mm,
        size_y_mm=params.canvas_size_mm
    )
    if prior_type == 'random':
        mask_maker.random_real()
    elif prior_type == 'transparent':
        mask_maker.fully_transparent()
    elif prior_type == 'load':
        # Ask user to select a prior mask file using the custom function
        prior_file_path = select_file(
            "Select prior mask file",
            [
                ("Numpy", ".npy"),
                ("PNG", ".png"),
                ("JPEG", ".jpg"),
                ("JPEG", ".jpeg"),
                ("GIF", ".gif"),
                ("TIFF", ".tiff"),
                ("TIFF", ".tif")
            ]
        )
        if not prior_file_path:
            print("No prior file selected.")
            return
        if prior_file_path.endswith('.npy'):
            prior_mask = np.load(prior_file_path)
        else:
            prior_image = Image.open(prior_file_path)
            prior_mask = np.array(prior_image, dtype=np.float32)
        mask_maker.mask = prior_mask
    else:
        print("Invalid prior type selected.")
        return
    M_init = mask_maker.mask
    
    # Convert float arrays to 16-bit for visualization
    def float_to_uint16(arr):
        arr = np.clip(arr, 0, 1)
        return (arr * 65535).astype(np.uint16)

    # Save the initial prior mask
    tiff.imwrite('MasksEvolutions/initial_prior.tiff', float_to_uint16(M_init))

    # Ensure the prior matches the observed image size
    if M_init.shape != I_observed.shape:
        M_init = np.resize(M_init, I_observed.shape)

    # Update regularizers to include only 'tv'
    regularizers = {
        'tv': lambda M: inverse_params.tv_weight * total_variation(M)
    }

    constraints = {
        'non_negativity': lambda M: np.clip(M, 0, 1),
        'upper_bound': lambda M: np.clip(M, 0, 1)
    }
    
    # Define the forward operator
    def forward_operator(M):
        return propagation.propagate(M)
    
    # Create directory for mask evolution
    os.makedirs("MasksEvolutions", exist_ok=True)

    # Track loss history
    loss_history = []
    # Modify iteration_callback to record loss at every iteration
    def iteration_callback(M_current, iteration):
        # Save masks at intervals
        if iteration % inverse_params.save_interval == 0:
            tiff.imwrite(f"MasksEvolutions/mask_iter_{iteration:03d}.tiff",
                         float_to_uint16(M_current))
        # Compute current intensity and loss
        I_current = forward_operator(M_current)
        loss = huber_loss(I_observed, I_current)
        loss_history.append((iteration, loss))

    # Ensure ADMMSolver calls the callback at each iteration
    solver = ADMMSolver(forward_operator, huber_loss, regularizers, constraints,
                        callback=iteration_callback)

    print(f"\nRunning ADMM solver...")
    M_reconstructed = solver.solve(
        I_observed,
        M_init.copy(),
        max_iter=inverse_params.max_iter,
        rho=inverse_params.admm_rho
    )
    
    # Save the final reconstructed mask
    tiff.imwrite('MasksEvolutions/reconstructed_mask.tiff', float_to_uint16(M_reconstructed))

    # Simplify results storage - only for ADMM
    results = {'ADMM': M_reconstructed}
    loss_histories = {'ADMM': loss_history}

    # Plotting section for ADMM solver
    solver_name = 'ADMM'
    M_reconstructed = results[solver_name]
    # Create a new figure for the solver with 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    plt.suptitle(f'{solver_name} Solver Results', fontsize=16, y=0.95)

    # 1. Original Image (Top left)
    ax = axs[0, 0]
    ax.set_title('Original Input Image')
    extent = [-params.canvas_size_mm / 2, params.canvas_size_mm / 2,
              -params.canvas_size_mm / 2, params.canvas_size_mm / 2]
    im = ax.imshow(original_image, cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 2. Prior Mask (Top middle)
    ax = axs[0, 1]
    ax.set_title('Prior Mask')
    im = ax.imshow(float_to_uint16(M_init), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 3. Reconstructed Mask (Top right)
    ax = axs[0, 2]
    mse = np.mean((M_reconstructed - I_observed) ** 2)
    ssim_index = ssim(I_observed, M_reconstructed, data_range=1.0)
    ax.set_title(f'Reconstructed Mask\nMSE: {mse:.6f}\nSSIM: {ssim_index:.6f}')
    im = ax.imshow(float_to_uint16(M_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 4. Forward Propagation (Bottom left)
    ax = axs[1, 0]
    I_reconstructed = propagation.propagate(M_reconstructed)
    ax.set_title('Reconstructed Propagation')
    im = ax.imshow(float_to_uint16(I_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 5. Loss History (Bottom middle)
    ax = axs[1, 1]
    if loss_histories[solver_name]:
        iterations, losses = zip(*loss_histories[solver_name])
        ax.semilogy(iterations, losses, 'b-', linewidth=2)
    else:
        ax.plot([], [])
    ax.set_title('Loss History')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.minorticks_on()

    # 6. Ground Truth Mask (Bottom right)
    if ground_truth is not None:
        ax = axs[1, 2]
        ax.set_title('Ground Truth Mask')
        im = ax.imshow(float_to_uint16(ground_truth), cmap='gray', extent=extent, aspect='equal')
        fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Position (mm)')

        # Save ground truth
        tiff.imwrite('MasksEvolutions/ground_truth.tiff', float_to_uint16(ground_truth))
    else:
        axs[1, 2].axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plt.savefig(f'MasksEvolutions/{solver_name}_results.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Save the original image too
    tiff.imwrite('MasksEvolutions/original_input.tiff', original_image)

    # Save the results in 16-bit TIFF format
    tiff.imwrite('MasksEvolutions/debug_I_observed.tiff', float_to_uint16(I_observed))
    tiff.imwrite('MasksEvolutions/debug_M_reconstructed.tiff', float_to_uint16(M_reconstructed))

if __name__ == "__main__":
    main()