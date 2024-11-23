import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import tifffile as tiff
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
import os

# ========================================================================
# Configuration Parameters
# ========================================================================

@dataclass
class Config:
    # Propagation Parameters
    wavelength: float = 1000.0         # Wavelength in nanometers
    distance: float = 0.0               # Propagation distance in millimeters
    pixel_size: float = 40.0            # Pixels per millimeter
    propagator_type: str = 'Fresnel'    # 'Fresnel' or 'AngularSpectrum'

    # ADMM Solver Parameters
    max_iter: int = 50                # Maximum number of iterations
    rho: float = 10.0                   # ADMM penalty parameter
    save_interval: int = 5            # Interval for saving mask evolution
    learning_rate: float = 1e-4       # Learning rate for updates
    convergence_threshold: float = 1e-7 # Convergence threshold for early stopping

    # Regularization Parameters
    tv_weight: float = 0.1             # Weight for total variation regularization

    # MaskMaker Parameters
    prior_type: str = 'random'         # 'random', 'load', 'ones'

    pinhole_radius: float = 2.0        # Pinhole radius in cycles per millimeter

    # Miscellaneous
    output_dir: str = 'MasksEvolutions' # Directory to save outputs

    def __post_init__(self):
        self.wavelength = self.wavelength * 1e-9  # Convert nanometers to meters
        self.distance = self.distance * 1e-3      # Convert millimeters to meters
        self.pixel_size = 1 / self.pixel_size     # Convert pixels per millimeter to meters per pixel

# ========================================================================
# Utility Functions
# ========================================================================

def select_file(title, filetypes):
    """Simplified file selection dialog."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[(desc, f"*{ext}") for desc, ext in filetypes]
    )
    root.quit()
    root.destroy()
    return file_path

def select_prior_type_gui():
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
    tk.Radiobutton(root, text="Ones Mask", variable=selected_type, value="ones").pack()  # New option
    tk.Button(root, text="OK", command=on_select).pack(pady=20)

    root.mainloop()
    prior_type = selected_type.get()
    root.destroy()
    return prior_type

# ========================================================================
# MaskMaker Class
# ========================================================================

class MaskMaker:
    """Generates and updates mask arrays."""
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.mask = np.zeros((size_y, size_x), dtype=np.float32)

    def random_real(self):
        self.mask = np.random.rand(self.size_y, self.size_x).astype(np.float32)

    def ones_mask(self):
        self.mask = np.ones((self.size_y, self.size_x), dtype=np.float32)

    def load_from_file(self, filepath):
        if filepath.endswith('.npy'):
            self.mask = np.load(filepath)
        else:
            prior_image = Image.open(filepath).convert('L')
            self.mask = np.array(prior_image, dtype=np.float32) / 255.0

    def set_pixels(self, values, locations):
        """
        Set specific pixel values at given locations.

        Parameters:
            values (list/array): Values between 0 and 1 to set at each location
            locations (list/array): List of (x,y) integer pixel coordinates
        """
        values = np.asarray(values)
        locations = np.asarray(locations, dtype=np.int32)  # Force integer indices
        
        # Ensure coordinates are within bounds
        mask = ((0 <= locations[:, 0]) & (locations[:, 0] < self.size_x) & 
                (0 <= locations[:, 1]) & (locations[:, 1] < self.size_y))
        
        # Only set valid pixel locations
        valid_values = values[mask]
        valid_locations = locations[mask]
        
        for val, (x, y) in zip(valid_values, valid_locations):
            self.mask[y, x] = val

# ========================================================================
# Propagators Class
# ========================================================================

class Propagators:
    """Handles different types of wave propagation."""
    def __init__(self, config: Config):
        self.config = config
        self.propagator_type = config.propagator_type.lower()
        self.wavelength = config.wavelength  # Use converted value
        self.distance = config.distance      # Use converted value
        self.pixel_size = config.pixel_size  # Use converted value
        self.pinhole_radius = config.pinhole_radius
        self.pad_factor = 2  # Padding factor

        # Initialize attributes
        self.FX = None
        self.FY = None
        self.F_squared = None
        self.H = None

    def _initialize_computational_grids(self, ny, nx):
        """
        Initialize frequency grids and transfer function based on parameters.
        """
        # Calculate padded size
        pad_ny = ny * self.pad_factor
        pad_nx = nx * self.pad_factor

        # Frequency grids in cycles per meter
        fy = np.fft.fftfreq(pad_ny, d=self.pixel_size)
        fx = np.fft.fftfreq(pad_nx, d=self.pixel_size)
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.F_squared = self.FX**2 + self.FY**2

        # Transfer function based on selected propagation model
        if self.propagator_type == 'fresnel':
            self.H = np.exp(-1j * np.pi * self.wavelength * self.distance * self.F_squared)
        elif self.propagator_type == 'angularspectrum':
            k = 2 * np.pi / self.wavelength
            kz_squared = k**2 - (2 * np.pi * self.FX)**2 - (2 * np.pi * self.FY)**2
            kz = np.sqrt(np.maximum(kz_squared, 0.0))
            self.H = np.exp(1j * kz * self.distance)
        else:
            raise ValueError(f"Unknown propagator type: {self.propagator_type}")

        # Apply pinhole filter in frequency domain if specified
        if self.pinhole_radius > 0:
            # Pinhole radius in cycles per meter
            pinhole_radius_inv = self.pinhole_radius
            self.H *= self.F_squared <= (pinhole_radius_inv)**2

    def _pad_array(self, array):
        """Pad the array with ones to reduce ringing artifacts."""
        ny, nx = array.shape
        pad_ny = ny * self.pad_factor
        pad_nx = nx * self.pad_factor

        # Create array of ones (transmissive)
        padded_array = np.ones((pad_ny, pad_nx), dtype=array.dtype)

        # Calculate padding dimensions
        pad_y = (pad_ny - ny) // 2
        pad_x = (pad_nx - nx) // 2

        # Insert the array in the center
        padded_array[pad_y:pad_y+ny, pad_x:pad_x+nx] = array
        return padded_array

    def _crop_array(self, array, ny, nx):
        """
        Crop the padded array back to the original size.
        """
        pad_ny, pad_nx = array.shape
        y_start = (pad_ny - ny) // 2
        x_start = (pad_nx - nx) // 2
        cropped_array = array[y_start:y_start+ny, x_start:x_start+nx]
        return cropped_array

    def propagate(self, mask):
        ny, nx = mask.shape
        # Initialize computational grids and transfer function
        self._initialize_computational_grids(ny, nx)
        U0 = self._pad_array(mask)

        # Perform the propagation using FFT
        U1_fft = np.fft.fft2(U0)
        U2_fft = U1_fft * self.H
        U2 = np.fft.ifft2(U2_fft)

        # Remove padding
        U2 = self._crop_array(U2, ny, nx)

        # Return real-valued intensity
        intensity = np.abs(U2)**2
        return intensity.astype(np.float32)
    
# ========================================================================
# ADMM Solver Class
# ========================================================================

class ADMMSolver:
    """ADMM solver for inverse problems."""
    def __init__(self, forward_operator, regularizer, constraint, config: Config, callback=None):
        self.forward_operator = forward_operator
        self.regularizer = regularizer
        self.constraint = constraint
        self.config = config
        self.callback = callback

    def solve(self, I_observed, M_init):
        M = M_init.copy()
        Z = M_init.copy()
        U = np.zeros_like(M_init)
        rho = self.config.rho
        max_iter = self.config.max_iter
        learning_rate = self.config.learning_rate
        convergence_threshold = self.config.convergence_threshold

        for iteration in tqdm(range(1, max_iter + 1), desc="ADMM Solver"):
            # Update M: Minimize the augmented Lagrangian
            I_pred = self.forward_operator(M)
            gradient = 2 * (I_pred - I_observed)
            M = M - learning_rate * (gradient + rho * (M - Z + U))

            # Apply regularization (e.g., total variation)
            M = self.regularizer(M)

            # Update Z with constraints
            Z_old = Z.copy()
            Z = self.constraint(M + U)

            # Update dual variable U
            U += M - Z

            # Check for convergence
            if np.linalg.norm(Z - Z_old) < convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break

            # Callback for monitoring
            if self.callback is not None:
                self.callback(Z, iteration)

        return Z

# ========================================================================
# Regularization and Constraints
# ========================================================================

def total_variation(M):
    """Total variation regularization using Rudin-Osher-Fatemi (ROF) model."""
    # Simple gradient descent step for TV minimization
    tv_weight = config.tv_weight
    grad_x = np.diff(M, axis=1)
    grad_y = np.diff(M, axis=0)
    grad_x = np.pad(grad_x, ((0,0),(0,1)), 'constant')
    grad_y = np.pad(grad_y, ((0,1),(0,0)), 'constant')
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    div_x = grad_x / grad_mag
    div_y = grad_y / grad_mag
    divergence = (np.diff(div_x, axis=1, append=0) +
                  np.diff(div_y, axis=0, append=0))
    M -= tv_weight * divergence
    return M

def non_negativity(M):
    """Non-negativity constraint."""
    return np.clip(M, 0, 1)

def huber_loss(I_observed, I_estimated, delta=1.0):
    """
    Compute the Huber loss between the observed and estimated intensity images.
    
    Parameters:
        I_observed (np.ndarray): Observed intensity image.
        I_estimated (np.ndarray): Estimated intensity image from the forward model.
        delta (float): Threshold parameter for the Huber loss (default: 1.0).
    
    Returns:
        float: The Huber loss value.
    """
    difference = I_estimated - I_observed
    abs_diff = np.abs(difference)
    mask = abs_diff <= delta
    quadratic_loss = 0.5 * (difference[mask]) ** 2
    linear_loss = delta * (abs_diff[~mask] - 0.5 * delta)
    loss = np.sum(quadratic_loss) + np.sum(linear_loss)
    return loss

# ========================================================================
# Main Function
# ========================================================================

def main():
    global config
    config = Config()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # User selects observed image
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

    # User selects ground truth mask (optional)
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

    # Load observed image
    original_image = tiff.imread(observed_file_path)
    image_array = original_image.astype(np.float32)
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    I_observed = np.clip(image_array, 0, 1)

    # Load ground truth if provided
    ground_truth = None
    if (ground_truth_path):
        ground_truth = tiff.imread(ground_truth_path)
        ground_truth = ground_truth.astype(np.float32)
        ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())

    # MaskMaker setup
    prior_type = select_prior_type_gui()
    config.prior_type = prior_type
    size_y, size_x = I_observed.shape
    mask_maker = MaskMaker(size_x, size_y)

    if config.prior_type == 'random':
        mask_maker.random_real()
    elif config.prior_type == 'ones':
        mask_maker.ones_mask()
    elif config.prior_type == 'load':
        prior_file_path = select_file(
            "Select prior mask file",
            [
                ("Numpy", ".npy"),
                ("PNG", ".png"),
                ("JPEG", ".jpg"),
                ("GIF", ".gif"),
                ("TIFF", ".tiff"),
                ("TIFF", ".tif")
            ]
        )
        if not prior_file_path:
            print("No prior file selected.")
            return
        mask_maker.load_from_file(prior_file_path)
    else:
        print("Invalid prior type selected.")
        return
    M_init = mask_maker.mask

    # Initialize Propagators
    propagators = Propagators(config)

    # Regularizer and Constraint
    regularizer = total_variation
    constraint = non_negativity

    # Track loss history
    loss_history = []

    # Callback function to monitor progress
    def iteration_callback(M_current, iteration):
        # Save masks at intervals
        if iteration % config.save_interval == 0:
            tiff.imwrite(os.path.join(config.output_dir, f'mask_iter_{iteration:03d}.tiff'), 
                        (M_current * 65535).astype(np.uint16))
        # Compute current intensity and loss
        I_current = propagators.propagate(M_current)
        loss = np.mean((I_current - I_observed)**2)
        loss_history.append((iteration, loss))
        # Removed print statement to only show the progress bar

    # Initialize ADMM Solver
    solver = ADMMSolver(
        forward_operator=propagators.propagate,
        regularizer=regularizer,
        constraint=constraint,
        config=config,
        callback=iteration_callback
    )

    print("\nRunning ADMM solver...")
    M_reconstructed = solver.solve(I_observed, M_init)

    # Save the final reconstructed mask
    tiff.imwrite(os.path.join(config.output_dir, 'reconstructed_mask.tiff'), 
                (M_reconstructed * 65535).astype(np.uint16))

    # Visualization
    plt.figure(figsize=(18, 12))

    # Observed Image
    plt.subplot(2, 3, 1)
    plt.title('Observed Image')
    plt.imshow(I_observed, cmap='gray')
    plt.axis('off')

    # Prior Mask
    plt.subplot(2, 3, 2)
    plt.title('Prior Mask')
    plt.imshow(M_init, cmap='gray')
    plt.axis('off')

    # Reconstructed Mask
    plt.subplot(2, 3, 3)
    plt.title('Reconstructed Mask')
    plt.imshow(M_reconstructed, cmap='gray')
    plt.axis('off')

    # Reconstructed Intensity
    I_reconstructed = propagators.propagate(M_reconstructed)
    plt.subplot(2, 3, 4)
    plt.title('Reconstructed Intensity')
    plt.imshow(I_reconstructed, cmap='gray')
    plt.axis('off')

    # Loss History
    plt.subplot(2, 3, 5)
    iterations, losses = zip(*loss_history) if loss_history else ([], [])
    plt.plot(iterations, losses, 'b-', linewidth=2)
    plt.title('Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # Ground Truth Mask
    if ground_truth is not None:
        plt.subplot(2, 3, 6)
        plt.title('Ground Truth Mask')
        plt.imshow(ground_truth, cmap='gray')
        plt.axis('off')

        # Save ground truth
        tiff.imwrite(os.path.join(config.output_dir, 'ground_truth.tiff'), 
                    (ground_truth * 65535).astype(np.uint16))
    else:
        plt.subplot(2, 3, 6).axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'results.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Save additional outputs
    tiff.imwrite(os.path.join(config.output_dir, 'original_input.tiff'), original_image)
    tiff.imwrite(os.path.join(config.output_dir, 'I_observed.tiff'), (I_observed * 65535).astype(np.uint16))
    tiff.imwrite(os.path.join(config.output_dir, 'M_reconstructed.tiff'), (M_reconstructed * 65535).astype(np.uint16))

if __name__ == "__main__":
    main()