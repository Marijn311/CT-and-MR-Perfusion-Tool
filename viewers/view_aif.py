from utils.determine_aif import gamma_variate
from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
import numpy as np


def view_aif(time_index, ctc_volumes, aif_properties, aif_mask, mask, vmin=0, vmax=100):
    """
    Interactive visualization tool for Arterial Input Function (AIF) selection.
    Creates a three-panel display showing:
    1. Contrast time curves with AIF segmentation overlay (red)
    2. Plain contrast time curves 
    3. Mean signal in the selected AIF region and the corresponding fitted gamma variate function

    Parameters: 
        - time_index (list): Time points corresponding to the CTC volumes
        - ctc_volumes (list): List of 3D arrays (z,y,x) containing the contrast time curves (CTC)
        - aif_properties (nd.array): 1D array with 4 elements representing parameters for fitted gamma variate function (t0, alpha, beta, amplitude)
        - aif_mask (nd.array):  3D binary mask (z,y,x) indicating the selected AIF voxels
        - mask (nd.array): 3D binary brain mask
        - vmin (float, default=0): Lower percentile for contrast scaling
        - vmax (float, default=100): Upper percentile for contrast scaling
        
    Controls:
        - Mouse wheel: Navigate through time points
        - Slider: Navigate through slices
        - Arrow keys: Navigate through time points
    """

    # Convert from a list of 3D arrays to one 4D array
    ctc_volumes = np.stack(ctc_volumes, axis=0)

    def majority(array, ignore=None):
        """Return the most common element in a 1D array, ignoring specified values."""
        labels, counts = np.unique(array.ravel(), return_counts=True)
        if ignore is not None:
            mask = labels.isin(ignore)
            counts = counts[~mask]
            labels = labels[~mask]
            return labels[np.argmax(counts)]
        else:
            return labels[np.argmax(counts)]
        
    majority_slice = majority(np.where(aif_mask)[0])
    
    # Create figure with 3 subplots and space for slider
    fig = plt.figure(figsize=(18, 7))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    # Initialize current indices
    current_time_idx = 0
    current_slice_idx = int(majority_slice)
    
    # Pre-calculate global vmin and vmax for consistent scaling across all slices
    all_masked_values = []
    for s in range(ctc_volumes.shape[1]):  
        mask_slice = mask[s,:,:]
        for t in range(ctc_volumes.shape[0]): 
            masked_image = ctc_volumes[t, s, :, :].copy()
            masked_image[mask_slice == 0] = 0
            if np.any(mask_slice > 0):  # Only if there are brain voxels in this slice
                all_masked_values.extend(masked_image[mask_slice > 0].flatten())

    # Set vmin and vmax based on the percentiles of all masked values
    ctc_vmin, ctc_vmax = np.percentile(all_masked_values, [vmin, vmax])
    
    def update_ctc_display(time_idx=None, slice_idx=None):
        """ Update the displayed images based on new time or slice indices."""
        nonlocal current_time_idx, current_slice_idx
        
        if time_idx is not None:
            current_time_idx = max(0, min(time_idx, ctc_volumes.shape[0] - 1))
        if slice_idx is not None:
            current_slice_idx = max(0, min(slice_idx, ctc_volumes.shape[1] - 1))
        
        # Get the current slice data
        ctc_slice_data = ctc_volumes[:, current_slice_idx, :, :]
        mask_slice = mask[current_slice_idx]
        
        # Apply brain mask to the CTC image for both subplots
        ctc_image = ctc_slice_data[current_time_idx].copy()
        ctc_image[mask_slice == 0] = 0  # Set pixels outside brain mask to 0
        
        # Update left subplot (CTC with AIF overlay)
        # Create RGB version for overlay
        ctc_image_rgb = np.stack([ctc_image] * 3, axis=-1)
        # Normalize to 0-255 range for RGB display
        ctc_image_rgb = ((ctc_image_rgb - ctc_vmin) / (ctc_vmax - ctc_vmin) * 255).clip(0, 255)
        # Apply red color to AIF candidate voxels
        aif_mask_slice = aif_mask[current_slice_idx] > 0
        ctc_image_rgb[aif_mask_slice] = [255, 0, 0]  # Set red color (RGB)
        
        ctc_img_overlay.set_data(ctc_image_rgb.astype('uint8'))
        ax1.set_title(f'Contrast Signal with AIF Segmentation\n(Red: AIF Region)\nSlice {current_slice_idx+1}/{ctc_volumes.shape[1]}, Timepoint {current_time_idx+1}/{ctc_volumes.shape[0]}\nTime: {time_index[current_time_idx]:.1f}s')
        
        # Update middle subplot (regular CTC)
        ctc_img.set_data(ctc_image)
        ax2.set_title(f'Contrast Signal\n(Scroll Through Time with Mouse)\nSlice {current_slice_idx+1}/{ctc_volumes.shape[1]}, Timepoint {current_time_idx+1}/{ctc_volumes.shape[0]}\nTime: {time_index[current_time_idx]:.1f}s')

        # Update the vertical line on the signal plot to show current time
        if hasattr(update_ctc_display, 'time_line'):
            update_ctc_display.time_line.remove()
        update_ctc_display.time_line = ax3.axvline(time_index[current_time_idx], color='red', linestyle='--', alpha=0.7, label='Current Time')
        ax3.legend()
        fig.canvas.draw_idle()
    
    def on_scroll(event):
        if event.inaxes in [ax1, ax2]:  # Allow scrolling on both left and middle subplots
            if event.button == 'up':
                update_ctc_display(time_idx=current_time_idx + 1)
            elif event.button == 'down':
                update_ctc_display(time_idx=current_time_idx - 1)
    
    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_ctc_display(time_idx=current_time_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_ctc_display(time_idx=current_time_idx - 1)
    
    # Initialize both CTC displays
    initial_ctc_slice_data = ctc_volumes[:, current_slice_idx, :, :]
    initial_mask_slice = mask[current_slice_idx]
    initial_ctc_image = initial_ctc_slice_data[current_time_idx].copy()
    initial_ctc_image[initial_mask_slice == 0] = 0  # Apply brain mask
    
    # Left subplot: CTC with AIF overlay
    initial_ctc_image_rgb = np.stack([initial_ctc_image] * 3, axis=-1)
    initial_ctc_image_rgb = ((initial_ctc_image_rgb - ctc_vmin) / (ctc_vmax - ctc_vmin) * 255).clip(0, 255)
    initial_aif_mask_slice = aif_mask[current_slice_idx] > 0
    initial_ctc_image_rgb[initial_aif_mask_slice] = [255, 0, 0]  # Set red color
    ctc_img_overlay = ax1.imshow(initial_ctc_image_rgb.astype('uint8'))
    ax1.set_axis_off()
    
    # Middle subplot: Regular CTC
    ctc_img = ax2.imshow(initial_ctc_image, cmap=plt.cm.gray, vmin=ctc_vmin, vmax=ctc_vmax)
    ax2.set_axis_off()
    
    # Create slider for slice navigation
    slider_ax = plt.axes([0.375, 0.02, 0.25, 0.03])  # Position: [left, bottom, width, height]
    slice_slider = Slider(slider_ax, 'Slice', 0, ctc_volumes.shape[1]-1, valinit=current_slice_idx, valfmt='%d')
    
    def on_slice_change(val):
        """ Update display when the slice slider is changed."""
        slice_idx = int(slice_slider.val)
        update_ctc_display(slice_idx=slice_idx)
    
    slice_slider.on_changed(on_slice_change)
    
    # Third subplot: Signal plot
    # Calculate mean AIF signal
    mean_aif_signal = ctc_volumes[:, aif_mask].mean(axis=1)
    
    ax3.plot(time_index, mean_aif_signal, label='Mean signal in AIF segmentation', linewidth=2)
    # Plot fitted gamma variate using the same time points as the signal for proper comparison

    fitted_signal = gamma_variate(time_index, *aif_properties)
    ax3.plot(time_index, fitted_signal, label='Fitted Gamma Variate', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Signal Intensity')
    ax3.set_title(f'Signal vs Fitted Model')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initialize the display
    update_ctc_display()
    plt.subplots_adjust(bottom=0.15)  # Make room for the slider
    plt.show()


