import numpy as np
from parameters import Parameters
from Mask_Maker import MaskMaker
from Propagation import Propagation
from loss_functions import huber_loss
from regularizations import total_variation, shape_bias
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
    root.geometry("300x200")
    
    selected_type = tk.StringVar(value="random")
    
    def on_select():
        root.quit()
    
    tk.Label(root, text="Choose prior type:").pack(pady=10)
    tk.Radiobutton(root, text="Random", variable=selected_type, value="random").pack()
    tk.Radiobutton(root, text="Load from file", variable=selected_type, value="load").pack()
    tk.Radiobutton(root, text="Central disk", variable=selected_type, value="disk").pack()
    tk.Button(root, text="OK", command=on_select).pack(pady=20)
    
    root.mainloop()
    prior_type = selected_type.get()
    root.destroy()
    return prior_type

def main():
    # Initialize parameters
    params = Parameters()
    inverse_params = InverseParameters()
    
    # Modified file selection
    file_path = select_file(
        "Select an image file",
        [
            ("PNG", ".png"),
            ("JPEG", ".jpg"),
            ("JPEG", ".jpeg"),
            ("GIF", ".gif"),
            ("TIFF", ".tiff"),
            ("TIFF", ".tif")
        ]
    )
    
    if not file_path:
        print("No file selected.")
        return

    # Load the image directly using tifffile for proper 16-bit handling
    original_image = tiff.imread(file_path)
    
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

    # Update canvas size parameters based on the image size
    params.canvas_size_pixels = I_observed.shape[1]
    params.canvas_size_mm = 10.0  # Adjust as needed based on scaling

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
    elif prior_type == 'disk':
        mask_maker.central_disk(diameter_fraction=0.2, opacity=0.0)
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

    # Ensure the prior matches the observed image size
    if M_init.shape != I_observed.shape:
        M_init = np.resize(M_init, I_observed.shape)

    # Replace the hardcoded regularizers with parameters from inverse_params
    ellipse_params = inverse_params.get_ellipse_params(params.canvas_size_pixels)
    regularizers = {
        'tv': lambda M: inverse_params.tv_weight * total_variation(M),
        'shape': lambda M: inverse_params.shape_weight * shape_bias(M, ellipse_params)
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

    # Convert float arrays to 16-bit for visualization
    def float_to_uint16(arr):
        arr = np.clip(arr, 0, 1)
        return (arr * 65535).astype(np.uint16)

    # Track loss history
    loss_history = []
    # Use save_interval from inverse_params
    def iteration_callback(M_current, iteration):
        if iteration % inverse_params.save_interval == 0:
            tiff.imwrite(f"MasksEvolutions/mask_iter_{iteration:03d}.tiff", 
                        float_to_uint16(M_current))
            I_current = forward_operator(M_current)
            loss = huber_loss(I_observed, I_current)
            loss_history.append((iteration, loss))

    # Only use the ADMM solver
    solver = ADMMSolver(forward_operator, huber_loss, regularizers, constraints, callback=iteration_callback)

    print(f"\nRunning ADMM solver...")
    M_reconstructed = solver.solve(
        I_observed,
        M_init.copy(),
        max_iter=inverse_params.max_iter,
        rho=inverse_params.admm_rho
    )

    # Simplify results storage - only for ADMM
    results = {'ADMM': M_reconstructed}
    loss_histories = {'ADMM': loss_history}

    # Plotting section for ADMM solver
    solver_name = 'ADMM'
    M_reconstructed = results[solver_name]
    # Create a new figure for the solver with 2x3 grid
    fig = plt.figure(figsize=(24, 12))
    plt.suptitle(f'{solver_name} Solver Results', fontsize=16, y=0.95)
    
    # Define subplot positions
    subplot_positions = {
        'original': 1,    # Top left
        'prior': 2,       # Top middle
        'reconstructed': 3, # Top right
        'propagation': 4,  # Bottom left 
        'loss': 5         # Bottom middle
    }
    
    # 1. Original Image (Top left)
    plt.subplot(2, 3, subplot_positions['original'])
    plt.title('Original Input Image')
    plt.imshow(original_image, cmap='gray')
    plt.colorbar(label='Intensity (16-bit)')
    plt.axis('off')
    
    # 2. Prior Mask (Top middle)
    plt.subplot(2, 3, subplot_positions['prior'])
    plt.title('Prior Mask')
    plt.imshow(float_to_uint16(M_init), cmap='gray')
    plt.colorbar(label='Intensity (16-bit)')
    plt.axis('off')
    
    # 3. Reconstructed Mask (Top right)
    plt.subplot(2, 3, subplot_positions['reconstructed'])
    mse = np.mean((M_reconstructed - I_observed)**2)
    ssim_index = ssim(I_observed, M_reconstructed, data_range=1.0)
    plt.title(f'Reconstructed Mask\nMSE: {mse:.6f}\nSSIM: {ssim_index:.6f}')
    plt.imshow(float_to_uint16(M_reconstructed), cmap='gray')
    plt.colorbar(label='Intensity (16-bit)')
    plt.axis('off')
    
    # 4. Forward Propagation (Bottom left)
    plt.subplot(2, 3, subplot_positions['propagation'])
    I_reconstructed = propagation.propagate(M_reconstructed)
    plt.title('Reconstructed Propagation')
    plt.imshow(float_to_uint16(I_reconstructed), cmap='gray')
    plt.colorbar(label='Intensity (16-bit)')
    plt.axis('off')
    
    # 5. Loss History (Bottom middle)
    if loss_histories[solver_name]:
        plt.subplot(2, 3, subplot_positions['loss'])
        iterations, losses = zip(*loss_histories[solver_name])
        plt.semilogy(iterations, losses, 'b-', linewidth=2)
        plt.title('Loss History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.minorticks_on()
    
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