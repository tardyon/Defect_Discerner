# masterMain.py

import numpy as np
import yaml
import os
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass, asdict
from scipy.fft import fft2, ifft2, fftfreq
from scipy.special import erf
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from tqdm import tqdm

# Parameters class
@dataclass
class Parameters:
    wavelength_um: float = 1
    z_mm: float = 0.0
    output_type: str = 'intensity'
    padding: bool = True
    pad_factor: int = 2
    use_edge_rolloff: bool = False
    canvas_size_pixels: int = 400
    canvas_size_mm: float = 10.0
    pinhole_radius_inv_mm: float = 2.0
    delta_mm: float = 0.01
    propagation_model: str = 'fresnel'

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
        return self.wavelength_um * 1e-3 / self.scaling_mm_per_pixel  # Convert µm to mm

    @property
    def z_pixels(self) -> float:
        return self.z_mm / self.scaling_mm_per_pixel

    @property
    def pinhole_radius_inv_pixels(self) -> float:
        return self.pinhole_radius_inv_mm * self.scaling_mm_per_pixel

    @property
    def delta_pixels(self) -> float:
        return self.delta_mm / self.scaling_mm_per_pixel

# InverseParameters class
@dataclass
class InverseParameters:
    prior_type: str = 'random'               # 'random', 'load', 'transparent'
    prior_filepath: str = None               # Filepath to load prior mask
    max_iter: int = 500                      # Maximum number of iterations
    convergence_threshold: float = 1e-7      # Convergence threshold
    save_interval: int = 10                  # Interval for saving mask evolution
    tv_weight: float = 0.1                   # Weight for total variation regularization
    admm_rho: float = 12.0                   # ADMM penalty parameter

    def validate(self):
        valid_prior_types = ['random', 'load', 'transparent']
        if self.prior_type not in valid_prior_types:
            raise ValueError(f"Invalid prior_type '{self.prior_type}'. Valid options are {valid_prior_types}.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive.")

# Propagation class
class Propagation:
    def __init__(self, params: Parameters):
        self.params = params
        self.FX = None
        self.FY = None
        self.F_squared = None
        self.H = None
        self.model = params.propagation_model
        self._initialize_computational_grids()

    def _initialize_computational_grids(self):
        wavelength_pixels = self.params.wavelength_pixels
        z_pixels = self.params.z_pixels
        canvas_size_pixels = self.params.canvas_size_pixels
        padding = self.params.padding
        pad_factor = self.params.pad_factor
        pinhole_radius_inv_pixels = self.params.pinhole_radius_inv_pixels

        if padding:
            ny_padded = canvas_size_pixels * pad_factor
            nx_padded = canvas_size_pixels * pad_factor
        else:
            ny_padded = canvas_size_pixels
            nx_padded = canvas_size_pixels

        fx = fftfreq(nx_padded)
        fy = fftfreq(ny_padded)
        self.FX, self.FY = np.meshgrid(fx, fy)
        self.F_squared = (self.FX)**2 + (self.FY)**2

        if self.model == 'fresnel':
            self.H = np.exp(-1j * np.pi * wavelength_pixels * z_pixels * self.F_squared)
        elif self.model == 'angular_spectrum':
            k = 2 * np.pi / wavelength_pixels
            kz_squared = k**2 - (2 * np.pi * self.FX)**2 - (2 * np.pi * self.FY)**2
            kz = np.sqrt(kz_squared, where=kz_squared >= 0, out=np.zeros_like(kz_squared))
            self.H = np.exp(1j * kz * z_pixels)
        else:
            raise ValueError("Invalid propagation_model. Choose 'fresnel' or 'angular_spectrum'.")

        if pinhole_radius_inv_pixels > 0:
            self.H *= self.F_squared <= (pinhole_radius_inv_pixels)**2

    def _create_edge_rolloff(self) -> np.ndarray:
        if self.params.padding:
            size_pixels = self.params.canvas_size_pixels * self.params.pad_factor
        else:
            size_pixels = self.params.canvas_size_pixels

        delta_pixels = self.params.delta_pixels

        x = np.linspace(-size_pixels/2, size_pixels/2, size_pixels)
        y = np.linspace(-size_pixels/2, size_pixels/2, size_pixels)
        X, Y = np.meshgrid(x, y)

        rolloff_x = 0.5 * (erf((x + size_pixels/2)/delta_pixels) - erf((x - size_pixels/2)/delta_pixels))
        rolloff_y = 0.5 * (erf((y + size_pixels/2)/delta_pixels) - erf((y - size_pixels/2)/delta_pixels))

        edge_rolloff = np.outer(rolloff_y, rolloff_x)
        return edge_rolloff.astype(np.complex64)

    def _pad_array(self, array: np.ndarray) -> np.ndarray:
        pad_factor = self.params.pad_factor
        ny, nx = array.shape
        pad_ny = ny * pad_factor
        pad_nx = nx * pad_factor

        padded_array = np.ones((pad_ny, pad_nx), dtype=array.dtype)

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

        if self.params.use_edge_rolloff:
            edge_rolloff = self._create_edge_rolloff()
            if U0.shape != edge_rolloff.shape:
                raise ValueError(
                    f"Shape mismatch: U0 shape {U0.shape} and edge_rolloff shape {edge_rolloff.shape}"
                )
            U0 = U0 * edge_rolloff

        U1_fft = fft2(U0)
        U2_fft = U1_fft * self.H
        U2 = ifft2(U2_fft)

        if self.params.padding:
            U2 = self._crop_array(U2)

        intensity = np.abs(U2)**2
        return intensity.astype(np.float32)

# MaskMaker class
class MaskMaker:
    def __init__(self, size_x_pixels, size_y_pixels, size_x_mm, size_y_mm, prior_mask=None):
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
    def __init__(self, forward_operator, loss_function, regularizers, constraints, callback=None):
        self.forward_operator = forward_operator
        self.loss_function = loss_function
        self.regularizers = regularizers
        self.constraints = constraints
        self.callback = callback

    def solve(self, I_observed, M_init, **kwargs):
        raise NotImplementedError("Solve method must be implemented by subclasses.")

class ADMMSolver(SolverBase):
    def solve(self, I_observed, M_init, max_iter=100, rho=1.0):
        M = M_init.copy()
        Z = M.copy()
        U = np.zeros_like(M)

        for i in tqdm(range(max_iter), desc="ADMMSolver"):
            grad = self._compute_gradient(I_observed, M)
            M = Z - U - grad / rho
            for constraint in self.constraints.values():
                M = constraint(M)
            Z = M + U
            Z = self._proximal_operator(Z, 1 / rho)
            U += M - Z

            if self.callback:
                self.callback(M, i)

        return M

    def _compute_gradient(self, I_observed, M):
        I_estimated = self.forward_operator(M)
        difference = I_estimated - I_observed
        grad_M = 2 * self._backpropagate(difference, M)
        return grad_M

    def _backpropagate(self, diff, M):
        return diff

    def _proximal_operator(self, Z, alpha):
        return Z  # Placeholder implementation

# Loss function
def huber_loss(I_observed, I_estimated, delta=1.0):
    difference = I_estimated - I_observed
    abs_diff = np.abs(difference)
    mask = abs_diff <= delta
    quadratic_loss = 0.5 * (difference[mask]) ** 2
    linear_loss = delta * (abs_diff[~mask] - 0.5 * delta)
    loss = np.sum(quadratic_loss) + np.sum(linear_loss)
    return loss

# Regularization function
def total_variation(M, weight=0.1):
    grad_M = np.zeros_like(M)
    grad_x = np.roll(M, -1, axis=1) - M
    grad_y = np.roll(M, -1, axis=0) - M
    grad_M += np.roll(grad_x, 1, axis=1) - grad_x + np.roll(grad_y, 1, axis=0) - grad_y
    return weight * grad_M

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
    inverse_params = InverseParameters()

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
        print("No observed image file selected.")
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

    # Load ground truth if provided
    ground_truth = None
    if ground_truth_path:
        ground_truth = tiff.imread(ground_truth_path)
        if ground_truth.dtype == np.uint16:
            ground_truth = ground_truth.astype(np.float32) / 65535.0
        elif ground_truth.dtype == np.uint8:
            ground_truth = ground_truth.astype(np.float32) / 255.0

    # Prepare image for processing
    image_array = original_image.astype(np.float32)
    print(f"Original image stats:")
    print(f"  Shape: {original_image.shape}")
    print(f"  Dtype: {original_image.dtype}")
    print(f"  Min: {original_image.min()}")
    print(f"  Max: {original_image.max()}")
    print(f"  Mean: {original_image.mean()}")

    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())

    # Simulate observed intensity
    propagation = Propagation(params)
    I_observed = np.clip(image_array, 0, 1)
    print(f"I_observed stats: min={I_observed.min()}, max={I_observed.max()}, mean={I_observed.mean()}")

    # Update canvas size parameters
    params.canvas_size_pixels = I_observed.shape[1]
    params.canvas_size_mm = 10.0
    pixel_size_mm = params.canvas_size_mm / params.canvas_size_pixels

    # Prior type selection
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
    os.makedirs("MasksEvolutions", exist_ok=True)
    tiff.imwrite('MasksEvolutions/initial_prior.tiff', float_to_uint16(M_init))

    # Ensure the prior matches the observed image size
    if M_init.shape != I_observed.shape:
        M_init = np.resize(M_init, I_observed.shape)

    # Regularizers
    regularizers = {
        'tv': lambda M: inverse_params.tv_weight * total_variation(M)
    }

    constraints = {
        'non_negativity': lambda M: np.clip(M, 0, 1),
        'upper_bound': lambda M: np.clip(M, 0, 1)
    }

    # Forward operator
    def forward_operator(M):
        return propagation.propagate(M)

    # Track loss history
    loss_history = []
    def iteration_callback(M_current, iteration):
        if iteration % inverse_params.save_interval == 0:
            tiff.imwrite(f"MasksEvolutions/mask_iter_{iteration:03d}.tiff",
                         float_to_uint16(M_current))
        I_current = forward_operator(M_current)
        loss = huber_loss(I_observed, I_current)
        loss_history.append((iteration, loss))

    # ADMMSolver
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
    extent = [-params.canvas_size_mm / 2, params.canvas_size_mm / 2,
              -params.canvas_size_mm / 2, params.canvas_size_mm / 2]
    im = ax.imshow(original_image, cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 2. Prior Mask
    ax = axs[0, 1]
    ax.set_title('Prior Mask')
    im = ax.imshow(float_to_uint16(M_init), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 3. Reconstructed Mask
    ax = axs[0, 2]
    mse = np.mean((M_reconstructed - I_observed) ** 2)
    ssim_index = ssim(I_observed, M_reconstructed, data_range=1.0)
    ax.set_title(f'Reconstructed Mask\nMSE: {mse:.6f}\nSSIM: {ssim_index:.6f}')
    im = ax.imshow(float_to_uint16(M_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Position (mm)')

    # 4. Forward Propagation
    ax = axs[1, 0]
    I_reconstructed = propagation.propagate(M_reconstructed)
    ax.set_title('Reconstructed Propagation')
    im = ax.imshow(float_to_uint16(I_reconstructed), cmap='gray', extent=extent, aspect='equal')
    fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
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
        fig.colorbar(im, ax=ax, label='Intensity (16-bit)')
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