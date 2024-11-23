"""
Defect Discerner - Mask Doodle Module

Author: Michael C.M. Varney
Version: 1.0.0

This module provides the Streamlit interface for drawing masks and performing optical propagation simulations.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Failed to import streamlit_drawable_canvas. Please ensure it's installed correctly.")
    st.stop()
from io import BytesIO
import tifffile
from Propagation import Propagation
from parameters import Parameters
import os
import yaml

# Constants
# Removed fixed CANVAS_SIZE
# CANVAS_SIZE = 400  # Fixed canvas size
CONFIG_FILE = "last_session.yaml"

# Initialize session state with saved parameters if available
if 'params' not in st.session_state:
    if os.path.exists(CONFIG_FILE):
        try:
            st.session_state.params = Parameters.from_yaml(CONFIG_FILE)
            # Removed setting canvas_size_pixels to fixed CANVAS_SIZE
            # st.session_state.params.canvas_size_pixels = CANVAS_SIZE  # Ensure canvas size is fixed
        except Exception as e:
            st.warning(f"Could not load saved parameters: {e}")
            # Use default parameters
            st.session_state.params = Parameters(
                wavelength_um=1.0,
                z_mm=50,
                # canvas_size_pixels=CANVAS_SIZE,  # Removed fixed canvas size
                canvas_size_mm=10.0,
                padding=True,
                pad_factor=2,
                pinhole_radius_inv_mm=2.0,
                delta_mm=0.1,
                use_edge_rolloff=True,
                output_type='intensity',
                propagation_model='fresnel'
            )
    else:
        # Use default parameters
        st.session_state.params = Parameters(
            wavelength_um=1.0,
            z_mm=50,
            # canvas_size_pixels=CANVAS_SIZE,  # Removed fixed canvas size
            canvas_size_mm=10.0,
            padding=True,
            pad_factor=2,
            pinhole_radius_inv_mm=2.0,
            delta_mm=0.1,
            use_edge_rolloff=True,
            output_type='intensity',
            propagation_model='fresnel'
        )
    st.session_state.propagation_system = Propagation(st.session_state.params)

if 'canvas_data' not in st.session_state:
    st.session_state.canvas_data = None
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None
if 'stroke_width_mm' not in st.session_state:
    st.session_state.stroke_width_mm = 0.5

# Function to update stroke width
def calculate_stroke_width(stroke_mm, params):
    pixels_per_mm = params.canvas_size_pixels / params.canvas_size_mm
    return stroke_mm * pixels_per_mm

def update_params(wavelength, z_distance, canvas_physical_size, canvas_size_pixels, pinhole_radius, use_pinhole, use_rolloff, prop_model):
    """Update parameters without triggering a page rerun"""
    st.session_state.params = Parameters(
        wavelength_um=wavelength,
        z_mm=z_distance,
        canvas_size_pixels=canvas_size_pixels,
        canvas_size_mm=canvas_physical_size,
        padding=True,
        pad_factor=2,
        pinhole_radius_inv_mm=pinhole_radius if use_pinhole else 0.0,
        delta_mm=0.1,
        use_edge_rolloff=use_rolloff,
        output_type='intensity',
        propagation_model=prop_model
    )
    st.session_state.propagation_system = Propagation(st.session_state.params)

# Define save_params at module level
def save_params():
    try:
        st.session_state.params.to_yaml(CONFIG_FILE)
        st.success("Parameters saved for next session!")
    except Exception as e:
        st.warning(f"Could not save parameters: {e}")

# Streamlit app UI
st.title("Interactive Mask Drawing and Propagation")

# Initialize session state if needed
if 'canvas_result' not in st.session_state:
    st.session_state.canvas_result = None
if 'stroke_width_mm' not in st.session_state:
    st.session_state.stroke_width_mm = 0.5

# Split into sidebar and main content
with st.sidebar:
    # Drawing Controls section first
    st.header("Drawing Controls")
    
    # Update drawing modes to exclude 'ellipse'
    drawing_mode = st.selectbox(
        "Drawing Tool", 
        ("freedraw", "line", "rect", "circle", "transform"),  # Removed 'ellipse'
        index=0
    )
    
    # Stroke width in mm with slider and text input
    stroke_width_mm = st.slider("Stroke width (mm)", 0.0, 3.0, st.session_state.stroke_width_mm, 0.01)
    stroke_width_input = st.text_input("Exact Stroke width (mm)", value=str(stroke_width_mm))
    try:
        stroke_width_mm = float(stroke_width_input)
        stroke_width_mm = max(0.0, min(stroke_width_mm, 3.0))
    except ValueError:
        pass
    st.session_state.stroke_width_mm = stroke_width_mm
    
    # Calculate stroke width in pixels
    stroke_width = calculate_stroke_width(stroke_width_mm, st.session_state.params)
    
    stroke_opacity = st.slider("Stroke opacity", 0.0, 1.0, 1.0, 0.1)
    if drawing_mode == 'freedraw':
        point_display_radius = st.slider("Point display radius", 1, 25, 3)
    else:
        point_display_radius = 0
    
    # Explanation of point_display_radius
    # The 'point_display_radius' determines the radius of points displayed during drawing.
    # It affects the size of the cursor or the plotted points when using the 'freedraw' tool.
    
    realtime_update = st.checkbox("Update in realtime", True)
    
    # Drawing canvas after controls are defined
    st.subheader("Draw Mask")
    try:
        # Replaced fixed CANVAS_SIZE with dynamic value from params
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=int(stroke_width),  # Convert to integer for canvas
            stroke_color=f"rgba(0, 0, 0, {stroke_opacity})",
            background_color="#FFFFFF",
            background_image=st.session_state.uploaded_mask if 'uploaded_mask' in st.session_state else None,
            update_streamlit=realtime_update,
            height=st.session_state.params.canvas_size_pixels,  # Dynamic canvas size
            width=st.session_state.params.canvas_size_pixels,   # Dynamic canvas size
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius,
            key=f"canvas_{st.session_state.params.canvas_size_mm}",  # Force redraw on size change
        )
        if canvas_result.json_data is not None:
            st.session_state.canvas_data = canvas_result.json_data
        st.session_state.canvas_result = canvas_result
    except Exception as e:
        st.error(f"Error creating canvas: {str(e)}")
        st.stop()

    # Propagation parameters in collapsible section
    with st.expander("Propagation Parameters"):
        # Wavelength control with slider and text input
        col1_wave, col2_wave = st.columns([3, 1])
        with col1_wave:
            wavelength = st.slider("Wavelength (µm)", 0.1, 10.0, 1.0, 0.01)  # Increased max to 100.0 µm and default to 20.0 µm
        with col2_wave:
            wavelength = st.number_input("µm", value=wavelength, min_value=0.1, max_value=100.0, step=0.01, format="%.1f")
        
        # Propagation distance control
        col1_z, col2_z = st.columns([3, 1])
        with col1_z:
            z_distance = st.slider("Propagation Distance", 0.0, 1000.0, 50.0, 0.1)
        with col2_z:
            z_distance = st.number_input("mm", value=z_distance, min_value=0.0, max_value=1000.0, step=0.1, format="%.1f")
        
        # Canvas physical size control with warning
        st.warning("⚠️ Changing Canvas Size will reset your drawing!")
        col1_size, col2_size = st.columns([3, 1])
        with col1_size:
            canvas_physical_size = st.slider("Canvas Size (mm)", 1.0, 100.0, 10.0, 0.1)
        with col2_size:
            canvas_physical_size = st.number_input("mm", value=canvas_physical_size, min_value=1.0, max_value=100.0, step=0.1, format="%.1f")
        
        # Canvas resolution control
        col1_res, col2_res = st.columns([3, 1])
        with col1_res:
            canvas_size_pixels = st.slider("Canvas Resolution (pixels)", 100, 2000, st.session_state.params.canvas_size_pixels, 50)
        with col2_res:
            canvas_size_pixels = st.number_input("Pixels", value=canvas_size_pixels, min_value=100, max_value=2000, step=50)
        
        # Pinhole parameters with default True
        use_pinhole = st.checkbox("Use Pinhole Filter", True)
        if use_pinhole:
            col1_pin, col2_pin = st.columns([3, 1])
            with col1_pin:
                pinhole_radius = st.slider("Pinhole Radius", 0.0, 10.0, 2.0, 0.1)
            with col2_pin:
                pinhole_radius = st.number_input("cycles/mm", value=pinhole_radius, min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
        else:
            pinhole_radius = 0.0
        
        # Other parameters
        prop_model = st.selectbox("Propagation Model", ["fresnel", "angular_spectrum"])
        use_rolloff = st.checkbox("Use Edge Rolloff", True)
        
        # Update parameters immediately when any value changes
        if any([
            wavelength != st.session_state.params.wavelength_um,
            z_distance != st.session_state.params.z_mm,
            canvas_physical_size != st.session_state.params.canvas_size_mm,
            canvas_size_pixels != st.session_state.params.canvas_size_pixels,  # Added canvas_size_pixels
            pinhole_radius != (st.session_state.params.pinhole_radius_inv_mm if use_pinhole else 0.0),
            use_pinhole != (st.session_state.params.pinhole_radius_inv_mm > 0.0),  # Added check for use_pinhole change
            use_rolloff != st.session_state.params.use_edge_rolloff,
            prop_model != st.session_state.params.propagation_model
        ]):
            update_params(
                wavelength, z_distance, canvas_physical_size,
                canvas_size_pixels, pinhole_radius, use_pinhole, use_rolloff, prop_model
            )
            save_params()  # Auto-save when parameters change
    
    # New collapsible section for Load and Save functionalities
    with st.expander("Session Controls"):
        # Load Mask
        st.header("Load Mask")
        uploaded_file = st.file_uploader("Upload Mask (TIFF)", type=['tiff', 'tif'])
        
        if uploaded_file is not None:
            try:
                # Load the mask
                mask_data = tifffile.imread(uploaded_file)
                
                # Normalize to [0, 1] range if 16-bit
                if mask_data.dtype == np.uint16:
                    mask_data = mask_data.astype(np.float32) / 65535.0
                elif mask_data.dtype == np.uint8:
                    mask_data = mask_data.astype(np.float32) / 255.0
                    
                # Determine expected shape based on padding
                if st.session_state.params.padding:
                    expected_size = st.session_state.params.canvas_size_pixels * st.session_state.params.pad_factor
                else:
                    expected_size = st.session_state.params.canvas_size_pixels

                expected_shape = (expected_size, expected_size)
                
                # Resize to match canvas if needed
                if mask_data.shape != expected_shape:
                    from scipy.ndimage import zoom
                    zoom_factor = st.session_state.params.canvas_size_pixels * st.session_state.params.pad_factor / mask_data.shape[0] if st.session_state.params.padding else st.session_state.params.canvas_size_pixels / mask_data.shape[0]
                    mask_data = zoom(mask_data, zoom_factor)
                
                # Store in session state for drawing
                st.session_state.uploaded_mask = mask_data
                st.success("Mask loaded successfully!")
            except Exception as e:
                st.error(f"Error loading mask: {e}")
        
        # Save Parameters
        st.button("Save Current Parameters", on_click=save_params)
        
        # Save Canvas and Intensity
        if st.button("Save Canvas and Intensity"):
            if st.session_state.canvas_result and st.session_state.canvas_result.image_data is not None:
                # Create mask TIFF
                mask = (np.mean(st.session_state.canvas_result.image_data[:, :, :3], axis=2) * 65535 / 255).astype(np.uint16)
                mask_bytes = BytesIO()
                tifffile.imwrite(mask_bytes, mask)
                mask_bytes.seek(0)
                
                # Create intensity TIFF
                propagated_intensity = st.session_state.propagation_system.propagate(mask / 65535.0)
                intensity_16bit = (propagated_intensity / propagated_intensity.max() * 65535).astype(np.uint16)
                intensity_bytes = BytesIO()
                tifffile.imwrite(intensity_bytes, intensity_16bit)
                intensity_bytes.seek(0)
                
                # Create download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Mask",
                        data=mask_bytes,
                        file_name="mask.tiff",
                        mime="image/tiff"
                    )
                with col2:
                    st.download_button(
                        label="Download Intensity",
                        data=intensity_bytes,
                        file_name="intensity.tiff",
                        mime="image/tiff"
                    )
            else:
                st.error("No canvas data to save.")

# Main content area - only show propagated intensity (removed split columns)
st.subheader("Propagated Intensity")
if st.session_state.canvas_result is not None and st.session_state.canvas_result.image_data is not None:
    # Ensure mask size matches parameters
    mask = np.mean(st.session_state.canvas_result.image_data[:, :, :3], axis=2).astype(np.float32) / 255.0
    if mask.shape[0] != st.session_state.params.canvas_size_pixels:
        st.session_state.params.canvas_size_pixels = mask.shape[0]
        st.session_state.propagation_system = Propagation(st.session_state.params)
    propagated_intensity = st.session_state.propagation_system.propagate(mask)
    
    # Create larger figure with physical units
    fig, ax = plt.subplots(figsize=(12, 12))
    extent = [-st.session_state.params.canvas_size_mm/2, st.session_state.params.canvas_size_mm/2, 
             -st.session_state.params.canvas_size_mm/2, st.session_state.params.canvas_size_mm/2]
    im = ax.imshow(propagated_intensity, cmap="gray", extent=extent)
    plt.colorbar(im, ax=ax, label="Intensity")
    ax.set_xlabel("Position (mm)")
    ax.set_ylabel("Position (mm)")
    st.pyplot(fig)