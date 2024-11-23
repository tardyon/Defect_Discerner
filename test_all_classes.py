import matplotlib.pyplot as plt
from inversemain import (
    Config, MaskMaker, Propagators,
    ADMMSolver, total_variation, non_negativity, select_file
)

# Configuration Parameters
WAVELENGTH = 1000.0  # Wavelength in nanometers
DISTANCE = 0.0       # Propagation distance in millimeters
PIXEL_SIZE = 40.0    # Pixels per millimeter
PROPAGATOR_TYPE = 'Fresnel'  # 'Fresnel' or 'AngularSpectrum'
PINHOLE_RADIUS = 2.0  # Pinhole radius in cycles per millimeter
MAX_ITER = 50         # Maximum number of iterations
RHO = 5.0             # ADMM penalty parameter
SAVE_INTERVAL = 5     # Interval for saving mask evolution
LEARNING_RATE = 1e-3  # Learning rate for updates
CONVERGENCE_THRESHOLD = 1e-7  # Convergence threshold for early stopping
TV_WEIGHT = 0.1       # Weight for total variation regularization

def test_maskmaker():
    print("Testing MaskMaker...")
    mask_maker = MaskMaker(size_x=256, size_y=256)
    mask_maker.random_real()
    mask_random = mask_maker.mask
    
    mask_maker.ones_mask()
    mask_ones = mask_maker.mask
    
    # Save or visualize masks as needed
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Random Mask')
    plt.imshow(mask_random, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Ones Mask')
    plt.imshow(mask_ones, cmap='gray')
    plt.axis('off')
    
    plt.show()

def test_propagators():
    print("Testing Propagators...")
    config = Config(
        wavelength=WAVELENGTH,
        distance=DISTANCE,
        pixel_size=PIXEL_SIZE,
        propagator_type=PROPAGATOR_TYPE,
        pinhole_radius=PINHOLE_RADIUS
    )
    propagators = Propagators(config)
    
    # Create a sample mask
    mask_maker = MaskMaker(size_x=256, size_y=256)
    mask_maker.ones_mask()
    mask = mask_maker.mask
    
    intensity = propagators.propagate(mask)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Sample Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Propagated Intensity')
    plt.imshow(intensity, cmap='gray')
    plt.axis('off')
    
    plt.show()

def test_admmsolver():
    print("Testing ADMMSolver...")
    config = Config(
        wavelength=WAVELENGTH,
        distance=DISTANCE,
        pixel_size=PIXEL_SIZE,
        propagator_type=PROPAGATOR_TYPE,
        pinhole_radius=PINHOLE_RADIUS,
        max_iter=MAX_ITER,
        rho=RHO,
        save_interval=SAVE_INTERVAL,
        learning_rate=LEARNING_RATE,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        tv_weight=TV_WEIGHT
    )
    propagators = Propagators(config)
    regularizer = total_variation
    constraint = non_negativity
    
    # Create sample observed image and initial mask
    mask_maker = MaskMaker(size_x=256, size_y=256)
    mask_maker.random_real()
    I_observed = mask_maker.mask
    mask_maker.ones_mask()
    M_init = mask_maker.mask
    
    def callback(M_current, iteration):
        if iteration % config.save_interval == 0:
            print(f"Iteration {iteration} completed.")
    
    solver = ADMMSolver(
        forward_operator=propagators.propagate,
        regularizer=regularizer,
        constraint=constraint,
        config=config,
        callback=callback
    )
    
    M_reconstructed = solver.solve(I_observed, M_init)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Initial Mask')
    plt.imshow(M_init, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Mask')
    plt.imshow(M_reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.show()

def main():
    test_maskmaker()
    test_propagators()
    test_admmsolver()

if __name__ == "__main__":
    main()