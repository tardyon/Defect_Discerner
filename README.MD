# Defect Discerner

**Author**: Michael C.M. Varney  
**Version**: 1.0.0

Defect Discerner is a Python-based simulation tool for studying and visualizing optical propagation models using Fresnel and Angular Spectrum methods. It is designed to compare the behavior of light propagation through an aperture under different models, helping researchers analyze the nuances of light behavior in optical systems.

---

## Features

- **Fresnel and Angular Spectrum Propagation**: Simulates optical propagation using two models.
- **Circular Aperture Mask Creation**: Generates occlusive circular aperture masks with adjustable parameters.
- **Visualization**: Displays the aperture mask, propagated intensities, and cross-sectional lineouts for comparison.
- **Configurable Parameters**: Adjust wavelength, propagation distance, canvas size, and more through the `Parameters`.

---

## New Features

- **User-Provided Masks**: You can now select an image file (PNG, GIF, JPG, TIFF) to use as the true mask for simulations.
- **Normalization**: The selected image is normalized to have a mean around 1 and positive values ≥ 0.
- **Inverse Parameters Configuration**: A new `InverseParameters` class allows you to control solver settings, select priors, and update methods.

---

## Installation

### Requirements
- Python 3.8+
- Required Python packages:
  - `numpy`
  - `matplotlib`
  - `streamlit-drawable-canvas`
  - `Pillow>=8.0.0`: For image loading and processing.

Install the dependencies using:

```bash
pip install -r requirements.txt

File Structure

	•	main.py: Main script to run the simulation and visualize results.
	•	parameters.py: Contains the Parameters class for configuring simulation parameters.
	•	Propagation.py: Module for propagating the aperture mask using specified models.

Usage

Run the Simulation

Run the main.py script to execute the simulation:

python main.py

Adjustable Parameters

The propagation is governed by parameters defined in the Parameters class. These include:
	•	Wavelength: wavelength_um (in micrometers)
	•	Propagation Distance: z_mm (in millimeters)
	•	Canvas Size: canvas_size_pixels (in pixels) and canvas_size_mm (in millimeters)
	•	Pinhole Radius: pinhole_radius_inv_mm (in cycles/mm)

You can modify these values in the main() function to tailor the simulation to your needs.

Output

The simulation generates:
	1.	Aperture Mask Visualization: Displays the occlusive circular aperture.
	2.	Propagated Intensities: Shows the intensity distributions for Fresnel and Angular Spectrum propagation.
	3.	Cross-Sectional Lineouts: Compares intensity profiles of the two propagation models.

Example Visualization

Mask	Fresnel Propagation	Angular Spectrum Propagation	Cross-Sectional Lineout
			

(Replace # with example images of your results)

Customization

Circular Aperture Mask

Customize the mask using the create_circular_aperture function in main.py:
	•	size_pixels: Define the size of the mask.
	•	radius_pixels: Specify the aperture radius.

Propagation Models

Add new propagation models by extending the Propagation class in Propagation.py.

Contributing

Contributions are welcome! Please submit issues or pull requests for bug fixes, enhancements, or new features.

License

This project is licensed under the MIT License.

Contact

For questions or support, contact Michael C.M. Varney
