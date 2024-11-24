# masterMain.py
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from scipy.fft import fft2, ifft2, fftfreq
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parameters class
@dataclass
class Parameters:
    wavelength_nm: float = 1000                    # Wavelength in nanometers
    z_mm: float = 0.0                             # Propagation distance in millimeters
    padding: bool = True                         # Flag to control padding of input mask
    pad_factor: int = 2
    pixel_scale_factor: float = 400/10.0              # Pixels per millimeter (user-set)
    pinhole_radius_inv_cyc_mm: float = 2.0        # Inverse pinhole radius in cycles/mm
    propagation_model: str = 'fresnel'
    prior_type: str = 'random'                    # 'random', 'load', 'transparent'
    prior_filepath: str = None                    # Filepath to load prior mask
    max_iter: int = 100                           # Maximum number of iterations
    convergence_threshold: float = 1e-7           # Convergence threshold
    save_interval: int = 5                       # Interval for saving mask evolution
    tv_weight: float = 0.1                      # TV regularization weight
    admm_rho: float = 10.0                         # ADMM penalty parameter
    clip_propagation: bool = True                 # Flag to control clipping of propagated intensity

    def __post_init__(self):
        pass

    def validate_inverse_parameters(self):
        if self.prior_type not in ['random', 'load', 'transparent']:
            raise ValueError(f"Invalid prior_type '{self.prior_type}'. Valid options are ['random', 'load', 'transparent'].")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive.")

# Propagation class
class Propagation:
    def __init__(self, params: Parameters, image_shape):
        self.params = params
        self.image_shape = image_shape
        self.H = None
        self.model = params.propagation_model
        self._initialize_computational_grids()

    def _initialize_computational_grids(self):
        ny, nx = self.image_shape
        padding = self.params.padding
        pad_factor = self.params.pad_factor
        pixel_scale = self.params.pixel_scale_factor  # Pixels per mm
        wavelength_pixels = self.params.wavelength_nm / 1e6 * pixel_scale  # Convert nm to mm, then to pixels
        z_pixels = self.params.z_mm * pixel_scale
        pinhole_radius_inv_pixels = self.params.pinhole_radius_inv_cyc_mm / pixel_scale

        if padding:
            ny_padded = ny * pad_factor
            nx_padded = nx * pad_factor
        else:
            ny_padded = ny
            nx_padded = nx

        fx = fftfreq(nx_padded)
        fy = fftfreq(ny_padded)
        FX, FY = np.meshgrid(fx, fy)
        F_squared = (FX)**2 + (FY)**2

        if self.model == 'fresnel':
            self.H = np.exp(-1j * np.pi * wavelength_pixels * z_pixels * F_squared)
        elif self.model == 'angular_spectrum':
            k = 2 * np.pi / wavelength_pixels
            kz_squared = k**2 - (2 * np.pi * FX)**2 - (2 * np.pi * FY)**2
            kz = np.sqrt(np.maximum(kz_squared, 0.0))
            self.H = np.exp(1j * kz * z_pixels)
        else:
            raise ValueError("Invalid propagation_model. Choose 'fresnel' or 'angular_spectrum'.")

        if pinhole_radius_inv_pixels > 0:
            self.H *= F_squared <= (pinhole_radius_inv_pixels)**2

    def _pad_array(self, array: np.ndarray) -> np.ndarray:
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        pad_ny = ny * pad_factor
        pad_nx = nx * pad_factor

        padded_array = np.ones((pad_ny, pad_nx), dtype=array.dtype)  # Padding with ones

        pad_y = (pad_ny - ny) // 2
        pad_x = (pad_nx - nx) // 2

        padded_array[pad_y:pad_y+ny, pad_x:pad_x+nx] = array
        return padded_array

    def _crop_array(self, array: np.ndarray) -> np.ndarray:
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        orig_ny = ny // pad_factor
        orig_nx = nx // pad_factor
        y_start = (ny - orig_ny) // 2
        x_start = (nx - orig_nx) // 2
        cropped_array = array[y_start:y_start+orig_ny, x_start:x_start+orig_nx]
        return cropped_array

    def propagate(self, mask_array: np.ndarray) -> np.ndarray:
        if self.params.padding:
            U0 = self._pad_array(mask_array)
        else:
            U0 = mask_array.copy()

        U1_fft = fft2(U0)
        U2_fft = U1_fft * self.H
        U2 = ifft2(U2_fft)

        if self.params.padding:
            U2 = self._crop_array(U2)

        intensity = np.abs(U2)**2
        return intensity.astype(np.float32)

# MaskMaker class
class MaskMaker:
    def __init__(self, size_x_pixels, size_y_pixels, size_x_mm=None, size_y_mm=None, prior_mask=None):
        self.size_x_pixels = size_x_pixels
        self.size_y_pixels = size_y_pixels
        self.size_x_mm = size_x_mm
        self.size_y_mm = size_y_mm
        if prior_mask is not None:
            self.mask = prior_mask.astype(np.float32)
            self.mask = np.clip(self.mask, 0, 1)
        else:
            self.mask = np.zeros((self.size_y_pixels, self.size_x_pixels), dtype=np.float32)

    def random_real(self):
        self.mask = np.random.rand(self.size_y_pixels, self.size_x_pixels).astype(np.float32)

    def set_pixels(self, values, locations):
        values = np.asarray(values)
        locations = np.asarray(locations, dtype=np.int32)

        mask = ((0 <= locations[:, 0]) & (locations[:, 0] < self.size_x_pixels) &
                (0 <= locations[:, 1]) & (locations[:, 1] < self.size_y_pixels))

        valid_values = values[mask]
        valid_locations = locations[mask]

        for val, (x, y) in zip(valid_values, valid_locations):
            self.mask[y, x] = val

    def resize_mask(self, new_size):
        zoom_factors = (new_size[0] / self.mask.shape[0], new_size[1] / self.mask.shape[1])
        self.mask = zoom(self.mask, zoom_factors, order=1).astype(np.float32)

    def fully_transparent(self):
        self.mask = np.ones((self.size_y_pixels, self.size_x_pixels), dtype=np.float32)

# Solver classes
class SolverBase:
    def __init__(self, forward_operator, loss_function, regularizers, constraints, propagation, callback=None):
        self.forward_operator = forward_operator
        self.loss_function = loss_function
        self.regularizers = regularizers
        self.constraints = constraints
        self.propagation = propagation
        self.callback = callback

class ADMMSolver(SolverBase):
    def _compute_gradient(self, I_observed, M):
        """Compute gradient of the loss function with respect to M."""
        if self.propagation.params.padding:
            M_padded = self.propagation._pad_array(M)
        else:
            M_padded = M.copy()

        # Forward propagation
        U_fft = fft2(M_padded)
        U_prop_fft = U_fft * self.propagation.H
        U_prop = ifft2(U_prop_fft)

        if self.propagation.params.padding:
            U_prop_cropped = self.propagation._crop_array(U_prop)
        else:
            U_prop_cropped = U_prop

        I_estimated = np.abs(U_prop_cropped)**2

        # Compute difference
        difference = I_estimated - I_observed

        # Backpropagate the difference
        if self.propagation.params.padding:
            difference_padded = self.propagation._pad_array(difference)
        else:
            difference_padded = difference

        # Compute gradient in the Fourier domain
        grad_fft = fft2(2 * U_prop * difference_padded)
        grad_back_fft = grad_fft * np.conj(self.propagation.H)
        grad_back = ifft2(grad_back_fft)

        if self.propagation.params.padding:
            grad_back = self.propagation._crop_array(grad_back)

        grad_M = np.real(grad_back)
        return grad_M

    def _proximal_operator(self, Z, alpha):
        """Proximal operator for TV regularization using Chambolle's method."""
        return denoise_tv_chambolle(Z, weight=alpha, channel_axis=None)

    def solve(self, I_observed, M_init, max_iter=100, rho=1.0, convergence_threshold=1e-7, tv_weight=0.001):
        """Solve the inverse problem using ADMM.

        Parameters:
        - I_observed: Observed intensity image.
        - M_init: Initial estimate of the mask.
        - max_iter: Maximum number of iterations.
        - rho: ADMM penalty parameter.
        - convergence_threshold: Threshold for convergence.
        - tv_weight: Weight for TV regularization.

        Returns:
        - M: Reconstructed mask.
        """
        M = M_init.copy()
        Z = M.copy()
        U = np.zeros_like(M)
        prev_loss = float('inf')
        eps_abs = convergence_threshold
        eps_rel = convergence_threshold

        for i in tqdm(range(max_iter), desc="ADMMSolver"):
            M_prev = M.copy()

            # Compute gradient
            grad = self._compute_gradient(I_observed, M)
            total_grad = grad  # TV regularization handled by proximal operator

            # Update M
            M = Z - U - (1.0 / rho) * total_grad

            # Apply constraints
            M = np.clip(M, 0, 1)

            # Update Z using proximal operator
            Z_prev = Z.copy()
            Z = self._proximal_operator(M + U, tv_weight / rho)

            # Update U
            U = U + (M - Z)

            # Compute residuals for convergence
            r_norm = np.linalg.norm(M - Z)
            s_norm = np.linalg.norm(-rho * (Z - Z_prev))
            eps_pri = np.sqrt(M.size) * eps_abs + eps_rel * max(np.linalg.norm(M), np.linalg.norm(-Z))
            eps_dual = np.sqrt(M.size) * eps_abs + eps_rel * np.linalg.norm(rho * U)

            # Check convergence
            if r_norm < eps_pri and s_norm < eps_dual:
                logging.info(f"Converged at iteration {i}")
                break

            if self.callback:
                self.callback(M, i)

        return M

# Loss function
def huber_loss(I_observed, I_estimated, delta=1.0):
    """Compute the Huber loss between observed and estimated intensities."""
    difference = I_estimated - I_observed
    abs_diff = np.abs(difference)
    mask = abs_diff <= delta
    quadratic_loss = 0.5 * (difference[mask]) ** 2
    linear_loss = delta * (abs_diff[~mask] - 0.5 * delta)
    loss = np.sum(quadratic_loss) + np.sum(linear_loss)
    return loss

# File selection helper
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[(desc, f"*{ext}") for desc, ext in filetypes]
    )
    root.quit()
    root.destroy()
    return file_path

# Prior type selection
def select_prior_type():
    root = tk.Tk()
    root.title("Select Prior Type")
    root.geometry("300x200")

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
    params.validate_inverse_parameters()

    # File selection for observed image
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
        logging.error("No observed image file selected.")
        return

    # Ground truth mask selection
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
    image_shape = original_image.shape

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path:
        ground_truth = tiff.imread(ground_truth_path)
        if ground_truth.dtype == np.uint16:
            ground_truth = ground_truth.astype(np.float32) / 65535.0
        elif ground_truth.dtype == np.uint8:
            ground_truth = ground_truth.astype(np.float32) / 255.0
        else:
            ground_truth = ground_truth.astype(np.float32)
            ground_truth /= ground_truth.max()

    # Prepare image for processing
    if original_image.dtype == np.uint16:
        I_observed = original_image.astype(np.float32) / 65535.0
    elif original_image.dtype == np.uint8:
        I_observed = original_image.astype(np.float32) / 255.0
    else:
        I_observed = original_image.astype(np.float32)
        I_observed /= I_observed.max()

    logging.info(f"Original image stats:")
    logging.info(f"  Shape: {original_image.shape}")
    logging.info(f"  Dtype: {original_image.dtype}")
    logging.info(f"  Min: {original_image.min()}")
    logging.info(f"  Max: {original_image.max()}")
    logging.info(f"  Mean: {original_image.mean()}")
    logging.info(f"I_observed stats: min={I_observed.min()}, max={I_observed.max()}, mean={I_observed.mean()}")

    # Apply clipping if needed
    I_observed = np.clip(I_observed, 0, 1)

    # Update pixel scale factor if needed
    pixel_scale_factor = params.pixel_scale_factor  # Pixels per mm

    # Prior type selection
    prior_type = select_prior_type()
    params.prior_type = prior_type

    # Initialize MaskMaker with prior
    mask_maker = MaskMaker(
        size_x_pixels=image_shape[1],
        size_y_pixels=image_shape[0]
    )
    if prior_type == 'random':
        mask_maker.random_real()
    elif prior_type == 'transparent':
        mask_maker.fully_transparent()
    elif prior_type == 'load':
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
            logging.error("No prior file selected.")
            return
        if prior_file_path.endswith('.npy'):
            prior_mask = np.load(prior_file_path)
        else:
            prior_image = tiff.imread(prior_file_path)
            if prior_image.dtype == np.uint16:
                prior_mask = prior_image.astype(np.float32) / 65535.0
            elif prior_image.dtype == np.uint8:
                prior_mask = prior_image.astype(np.float32) / 255.0
            else:
                prior_mask = prior_image.astype(np.float32)
                prior_mask /= prior_mask.max()
        mask_maker.mask = prior_mask
    else:
        logging.error("Invalid prior type selected.")
        return
    M_init = mask_maker.mask

    # Ensure the prior matches the observed image size
    if M_init.shape != I_observed.shape:
        M_init = resize(M_init, I_observed.shape, mode='reflect', anti_aliasing=True)

    # Convert float arrays to 16-bit for visualization
    def float_to_uint16(arr):
        arr = np.clip(arr, 0, 1)
        return (arr * 65535).astype(np.uint16)

    # Save the initial prior mask
    os.makedirs("MasksEvolutions", exist_ok=True)
    tiff.imwrite('MasksEvolutions/initial_prior.tiff', float_to_uint16(M_init))

    # Forward operator
    propagation = Propagation(params, image_shape)
    def forward_operator(M):
        return propagation.propagate(M)

    # Track loss history
    loss_history = []
    def iteration_callback(M_current, iteration):
        if iteration % params.save_interval == 0:
            tiff.imwrite(f"MasksEvolutions/mask_iter_{iteration:03d}.tiff",
                         float_to_uint16(M_current))
        I_current = forward_operator(M_current)
        loss = huber_loss(I_observed, I_current)
        loss_history.append((iteration, loss))

    # ADMMSolver initialization
    solver = ADMMSolver(
        forward_operator=forward_operator,
        loss_function=huber_loss,
        regularizers=None,  # Not used
        constraints=None,   # Not used
        propagation=propagation,
        callback=iteration_callback
    )

    logging.info(f"\nRunning ADMM solver...")
    M_reconstructed = solver.solve(
        I_observed,
        M_init.copy(),
        max_iter=params.max_iter,
        rho=params.admm_rho,
        convergence_threshold=params.convergence_threshold,
        tv_weight=params.tv_weight
    )

    # Save the final reconstructed mask
    tiff.imwrite('MasksEvolutions/reconstructed_mask.tiff', float_to_uint16(M_reconstructed))

    # Results storage
    results = {'ADMM': M_reconstructed}
    loss_histories = {'ADMM': loss_history}

    # Plotting
    solver_name = 'ADMM'
    M_reconstructed = results[solver_name]
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    plt.suptitle(f'{solver_name} Solver Results', fontsize=16, y=0.95)

    # 1. Original Image
    ax = axs[0, 0]
    ax.set_title('Original Input Image')
    extent = [
        -image_shape[1] / (2 * pixel_scale_factor),
        image_shape[1] / (2 * pixel_scale_factor),
        -image_shape[0] / (2 * pixel_scale_factor),
        image_shape[0] / (2 * pixel_scale_factor)
    ]
    im = ax.imshow(original_image, cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 2. Prior Mask
    ax = axs[0, 1]
    ax.set_title('Prior Mask')
    im = ax.imshow(float_to_uint16(M_init), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 3. Reconstructed Mask
    ax = axs[0, 2]
    if ground_truth is not None:
        mse = np.mean((M_reconstructed - ground_truth) ** 2)
        ssim_index = ssim(ground_truth, M_reconstructed, data_range=1.0)
        ax.set_title(f'Reconstructed Mask\nMSE: {mse:.6f}\nSSIM: {ssim_index:.6f}')
    else:
        ax.set_title(f'Reconstructed Mask')
    im = ax.imshow(float_to_uint16(M_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 4. Forward Propagation
    ax = axs[1, 0]
    I_reconstructed = propagation.propagate(M_reconstructed)
    ax.set_title('Reconstructed Propagation')
    im = ax.imshow(float_to_uint16(I_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 5. Loss History
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

    # 6. Ground Truth Mask
    if ground_truth is not None:
        ax = axs[1, 2]
        ax.set_title('Ground Truth Mask')
        im = ax.imshow(float_to_uint16(ground_truth), cmap='gray', extent=extent, aspect='equal')
        fig.colorbar(im, ax=ax, label='Intensity')
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Position (mm)')
        tiff.imwrite('MasksEvolutions/ground_truth.tiff', float_to_uint16(ground_truth))
    else:
        axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'MasksEvolutions/{solver_name}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save additional outputs
    tiff.imwrite('MasksEvolutions/original_input.tiff', original_image)
    tiff.imwrite('MasksEvolutions/debug_I_observed.tiff', float_to_uint16(I_observed))
    tiff.imwrite('MasksEvolutions/debug_M_reconstructed.tiff', float_to_uint16(M_reconstructed))

if __name__ == "__main__":
    main()
