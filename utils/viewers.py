import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.widgets import Slider
from utils.aif import gv


def view_4d_img(img_list, title, image_type=None, vmin=0, vmax=100):
    """Visualize a 4D volume in img_list as a 2D slice taken from the middle of the 3D volume.
    You can scroll through the time in this middle slice. 
    Besides the middle slice we show the maximum intensity projection (MIP) of the middle third slices of the volume for 'ctp' images,
    or minimum intensity projection (MinIP) for 'mrp' images."""

    current_idx = 0
    
    all_mip_values = []
    all_original_values = []
    processed_mip_volumes = []
    processed_original_volumes = []
    
    for img in img_list:
        
        # Maximum or minimum intensity projection from the middle third of the volume
        img_array = np.array(img)
        depth = img_array.shape[0]
        middle_third = img_array[depth//3:depth*2//3, :, :]
        
        # Use minimum intensity projection for 'mrp', maximum for 'ctp' or default
        if image_type == 'mrp':
            middle_slice_projection = np.min(middle_third, axis=0)
        else:
            middle_slice_projection = np.max(middle_third, axis=0)


        # Original image - take middle slice
        middle_slice_idx = img_array.shape[0] // 2
        middle_slice_original = img[middle_slice_idx, :, :,]

        processed_mip_volumes.append(middle_slice_projection)
        processed_original_volumes.append(middle_slice_original)
        all_mip_values.extend(middle_slice_projection.flatten())
        all_original_values.extend(middle_slice_original.flatten())


    mip_vmin, mip_vmax = np.percentile(all_mip_values, [vmin, vmax])
    original_vmin, original_vmax = np.percentile(all_original_values, [vmin, vmax])



    def update_volume(new_idx):
        nonlocal current_idx
        current_idx = max(0, min(new_idx, len(img_list) - 1))
        
        # Update original image
        img_original.set_data(processed_original_volumes[current_idx])
        ax_original.set_title(f"Middle Slice of 4D Volume\n(Scroll Through Time)\nTimepoint {current_idx+1}/{len(img_list)}")

        # Update MIP/MinIP image
        img_mip.set_data(processed_mip_volumes[current_idx])
        projection_type = "Minimum Intensity Projection" if image_type == 'mrp' else "Maximum Intensity Projection"
        ax_mip.set_title(f"{projection_type} of Middle Third of Slices\n(Scroll Through Time)\nTimepoint {current_idx+1}/{len(img_list)}")

        fig.suptitle(f"{title}", fontsize=14)
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes in [ax_original, ax_mip]:
            if event.button == 'up':
                update_volume(current_idx + 1)
            elif event.button == 'down':
                update_volume(current_idx - 1)

    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_volume(current_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_volume(current_idx - 1)

    # Create figure with two subplots side by side
    fig, (ax_original, ax_mip) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize displays
    img_original = ax_original.imshow(processed_original_volumes[0], cmap=plt.cm.Greys_r, vmin=original_vmin, vmax=original_vmax)
    ax_original.set_axis_off()
    
    img_mip = ax_mip.imshow(processed_mip_volumes[0], cmap=plt.cm.Greys_r, vmin=mip_vmin, vmax=mip_vmax)
    ax_mip.set_axis_off()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_volume(0)  # Initialize with the first volume
    plt.tight_layout()
    plt.show()




def view_brain_mask(brain_mask, volume_list):
    """
    Visualize the brain mask, the corresponding smoothed volume, and the masked volume side by side.
    """

    # Select the first volume
    volume=volume_list[0]
    
    # Apply the mask to the images and set background to -1000
    masked_volume_array = np.where(brain_mask > 0, volume, -1000)

    # Pre-calculate vmin and vmax for all volumes to ensure consistent scaling
    mask_vmin, mask_vmax = np.percentile(brain_mask, [2, 98])
    volume_vmin, volume_vmax = np.percentile(volume, [2, 98])
    masked_vmin, masked_vmax = np.min(masked_volume_array[masked_volume_array != -1000]), np.max(masked_volume_array[masked_volume_array != -1000])

    # Create a figure with three subplots
    fig, (ax_mask, ax_volume, ax_masked) = plt.subplots(1, 3, figsize=(15, 5))

    current_slice = 0

    def update_slice(new_slice):
        nonlocal current_slice
        current_slice = max(0, min(new_slice, brain_mask.shape[0] - 1))
        mask_display.set_data(brain_mask[current_slice])
        volume_display.set_data(volume[current_slice])
        masked_display.set_data(masked_volume_array[current_slice])
        ax_mask.set_title(f"3D Brain Mask\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask.shape[0]}")
        ax_volume.set_title(f"First Timepoint of 4D Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask.shape[0]}")
        ax_masked.set_title(f"First Timepoint of 4D Masked Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask.shape[0]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes in [ax_mask, ax_volume, ax_masked]:
            if event.button == 'up':
                update_slice(current_slice + 1)
            elif event.button == 'down':
                update_slice(current_slice - 1)

    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_slice(current_slice + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(current_slice - 1)

    # Display the first slice with consistent scaling across all slices
    mask_display = ax_mask.imshow(brain_mask[current_slice], cmap=plt.cm.gray, vmin=mask_vmin, vmax=mask_vmax)
    volume_display = ax_volume.imshow(volume[current_slice], cmap=plt.cm.gray, vmin=volume_vmin, vmax=volume_vmax)
    masked_display = ax_masked.imshow(masked_volume_array[current_slice], cmap=plt.cm.gray, vmin=masked_vmin, vmax=masked_vmax)

    # Remove axis labels
    ax_mask.set_axis_off()
    ax_volume.set_axis_off()
    ax_masked.set_axis_off()

    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize display
    update_slice(0)

    plt.show()


def show_perfusion_map(volume, title):
    """Visualize the perfusion map that were generaetd by the toolbox"""

    vmin, vmax = np.percentile(volume, [1, 99])

    fig, ax = plt.subplots()
    slice_idx = 0
    img = ax.imshow(volume[slice_idx], cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    ax.set_title(f"3D {title} Map\n(Scroll Through Slices)\nSlice {slice_idx + 1}/{volume.shape[0]}")
    plt.colorbar(img, ax=ax)

    def update_slice(new_idx):
        nonlocal slice_idx
        slice_idx = max(0, min(new_idx, volume.shape[0] - 1))
        img.set_data(volume[slice_idx])
        ax.set_title(f"3D {title} Map\n(Scroll Through Slices)\nSlice {slice_idx + 1}/{volume.shape[0]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes == ax:
            if event.button == 'up':
                update_slice(slice_idx + 1)
            elif event.button == 'down':
                update_slice(slice_idx - 1)

    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_slice(slice_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(slice_idx - 1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()



def view_aif_selection(time_index, img_ctc, aif_propperties, aif_candidate_segmentations, brain_mask, mean_fitting_error, aif_smoothness):
    """
    Visualize the region from which the AIF is extracted.
    Visualize the CTC images in this region.
    Visualize the contrast concentration curve in the AIF region and the fitted gamma variate function.
    """

    # Convert from a list of 3d volumes to one 4d volumes
    img_ctc = np.stack(img_ctc, axis=0)

    majority_slice = majority(np.where(aif_candidate_segmentations)[0])
    
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
    for s in range(img_ctc.shape[1]):  # Loop through slices
        brain_mask_slice = brain_mask[s]
        for t in range(img_ctc.shape[0]):  # Loop through time
            masked_image = img_ctc[t, s, :, :].copy()
            masked_image[brain_mask_slice == 0] = 0
            if np.any(brain_mask_slice == 1):  # Only if there are brain voxels in this slice
                all_masked_values.extend(masked_image[brain_mask_slice == 1].flatten())
    
    # Set vmin and vmax based on the 2nd and 98th percentiles of all masked values
    ctc_vmin, ctc_vmax = np.percentile(all_masked_values, [2, 98])
    
    def update_ctc_display(time_idx=None, slice_idx=None):
        nonlocal current_time_idx, current_slice_idx
        
        if time_idx is not None:
            current_time_idx = max(0, min(time_idx, img_ctc.shape[0] - 1))
        if slice_idx is not None:
            current_slice_idx = max(0, min(slice_idx, img_ctc.shape[1] - 1))
        
        # Get the current slice data
        ctc_slice_data = img_ctc[:, current_slice_idx, :, :]
        brain_mask_slice = brain_mask[current_slice_idx]
        
        # Apply brain mask to the CTC image for both subplots
        ctc_image = ctc_slice_data[current_time_idx].copy()
        ctc_image[brain_mask_slice == 0] = 0  # Set pixels outside brain mask to 0
        
        # Update left subplot (CTC with AIF overlay)
        # Create RGB version for overlay
        ctc_image_rgb = np.stack([ctc_image] * 3, axis=-1)
        # Normalize to 0-255 range for RGB display
        ctc_image_rgb = ((ctc_image_rgb - ctc_vmin) / (ctc_vmax - ctc_vmin) * 255).clip(0, 255)
        # Apply red color to AIF candidate voxels
        aif_mask_slice = aif_candidate_segmentations[current_slice_idx] > 0
        ctc_image_rgb[aif_mask_slice] = [255, 0, 0]  # Set red color (RGB)
        
        ctc_img_overlay.set_data(ctc_image_rgb.astype('uint8'))
        ax1.set_title(f'CTC With AIF Candidate Segmentation\n(Red: AIF Region)\nSlice {current_slice_idx+1}/{img_ctc.shape[1]}, Timepoint {current_time_idx+1}/{img_ctc.shape[0]}\nTime: {time_index[current_time_idx]:.1f}s')
        
        # Update middle subplot (regular CTC)
        ctc_img.set_data(ctc_image)
        ax2.set_title(f'Contrast Signal in Different Slices\n(Scroll Through Time with Mouse, Slices with Slider)\nSlice {current_slice_idx+1}/{img_ctc.shape[1]}, Timepoint {current_time_idx+1}/{img_ctc.shape[0]}\nTime: {time_index[current_time_idx]:.1f}s')

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
    initial_ctc_slice_data = img_ctc[:, current_slice_idx, :, :]
    initial_brain_mask_slice = brain_mask[current_slice_idx]
    initial_ctc_image = initial_ctc_slice_data[current_time_idx].copy()
    initial_ctc_image[initial_brain_mask_slice == 0] = 0  # Apply brain mask
    
    # Left subplot: CTC with AIF overlay
    initial_ctc_image_rgb = np.stack([initial_ctc_image] * 3, axis=-1)
    initial_ctc_image_rgb = ((initial_ctc_image_rgb - ctc_vmin) / (ctc_vmax - ctc_vmin) * 255).clip(0, 255)
    initial_aif_mask_slice = aif_candidate_segmentations[current_slice_idx] > 0
    initial_ctc_image_rgb[initial_aif_mask_slice] = [255, 0, 0]  # Set red color
    ctc_img_overlay = ax1.imshow(initial_ctc_image_rgb.astype('uint8'))
    ax1.set_axis_off()
    
    # Middle subplot: Regular CTC
    ctc_img = ax2.imshow(initial_ctc_image, cmap=plt.cm.gray, vmin=ctc_vmin, vmax=ctc_vmax)
    ax2.set_axis_off()
    
    # Create slider for slice navigation
    slider_ax = plt.axes([0.375, 0.02, 0.25, 0.03])  # Position: [left, bottom, width, height]
    slice_slider = Slider(slider_ax, 'Slice', 0, img_ctc.shape[1]-1, valinit=current_slice_idx, valfmt='%d')
    
    def on_slice_change(val):
        slice_idx = int(slice_slider.val)
        update_ctc_display(slice_idx=slice_idx)
    
    slice_slider.on_changed(on_slice_change)
    
    # Third subplot: Signal plot
    # Calculate mean AIF signal
    mean_aif_signal = img_ctc[:, aif_candidate_segmentations].mean(axis=1)
    
    ax3.plot(time_index, mean_aif_signal, label='Mean AIF Candidate Signal', linewidth=2)
    # Plot fitted gamma variate using the same time points as the signal for proper comparison
    fitted_signal = gv(time_index, *aif_propperties)
    ax3.plot(time_index, fitted_signal, label='Fitted Gamma Variate', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Signal Intensity')
    ax3.set_title(f'AIF Diagnostic: Signal vs Fitted Model\nMean Fitting Error: {mean_fitting_error:.2f}, Smoothness: {aif_smoothness:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initialize the display
    update_ctc_display()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the slider
    plt.show()


def majority(array, ignore=None):
    labels, counts = np.unique(array.ravel(), return_counts=True)
    if ignore is not None:
        mask = labels.isin(ignore)
        counts = counts[~mask]
        labels = labels[~mask]
        return labels[np.argmax(counts)]
    else:
        return labels[np.argmax(counts)]

def show_comparison_maps(generated, commercial, title, mask, apply_mask):
    """
    Display side-by-side comparison of generated and commercial perfusion maps.
    """
    
    # Find the mode and set it to NaN. This mode value is the background value.
    mode_commercial = scipy.stats.mode(commercial.flatten()).mode
    commercial_masked = np.where(commercial == mode_commercial, np.nan, commercial)
    mode_generated = scipy.stats.mode(generated.flatten()).mode
    generated_masked = np.where(generated == mode_generated, np.nan, generated)

    # The commercial and generated maps have different brain masks.
    # We only use the pixels which are not nan in both images.
    if apply_mask==True:
        mask = np.where(np.isfinite(commercial_masked) & np.isfinite(generated_masked), 1, 0)
        generated_masked = np.where(mask > 0, generated, np.nan)
        commercial_masked = np.where(mask > 0, commercial, np.nan)
    
    # Calculate value ranges for display only using foreground voxels (Take the 2nd and 98th percentiles to avoid outliers which can skew the color mapping)
    gen_foreground = generated_masked[mask > 0]
    com_foreground = commercial_masked[mask > 0]
    
    gen_vmin, gen_vmax = np.percentile(gen_foreground[np.isfinite(gen_foreground)], [2, 98])
    com_vmin, com_vmax = np.percentile(com_foreground[np.isfinite(com_foreground)], [2, 98])

    # Use the wider range for consistent visualization
    vmin = min(gen_vmin, com_vmin)
    vmax = max(gen_vmax, com_vmax)
    
    # Create colormap with black background for NaN values
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')
        
    # Plot the generated and commercial volumes side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    slice_idx = 0
    
    # Ensure both volumes have the same number of slices
    max_slices = min(generated_masked.shape[0], commercial_masked.shape[0])
    
    # Initial display
    title = title.upper()
    img1 = ax1.imshow(generated_masked[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Generated {title}")
    ax1.set_axis_off()
    
    img2 = ax2.imshow(commercial_masked[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Commercial {title}")
    ax2.set_axis_off()
    
    # Add colorbars
    plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add main title with slice info
    fig.suptitle(f"{title} Comparison\nSlice {slice_idx + 1}/{max_slices}", fontsize=14)
    
    def update_slice(new_idx):
        nonlocal slice_idx
        slice_idx = max(0, min(new_idx, max_slices - 1))
        
        img1.set_data(generated_masked[slice_idx])
        img2.set_data(commercial_masked[slice_idx])

        fig.suptitle(f"{title} Comparison\nSlice {slice_idx + 1}/{max_slices}", fontsize=14)
        fig.canvas.draw_idle()
    
    def on_scroll(event):
        if event.inaxes in [ax1, ax2]:
            if event.button == 'up':
                update_slice(slice_idx + 1)
            elif event.button == 'down':
                update_slice(slice_idx - 1)
    
    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_slice(slice_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(slice_idx - 1)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()


def shows_contrast_curve(total_contrast_value, s0_index):
    """
    Display a line graph of total contrast values over time with highlighted s0_index.
    
    Parameters
    ----------
    total_contrast_value : list
        List containing the normalized sum of contrast for every timepoint.
    s0_index : int
        Index of the baseline timepoint (s0) to be highlighted.
    """
    
    # Create timepoint indices for x-axis
    timepoints = list(range(len(total_contrast_value)))
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the contrast curve
    ax.plot(timepoints, total_contrast_value, 'b-', linewidth=2, label='Total Contrast Signal')
    
    # Highlight the s0_index point
    if 0 <= s0_index < len(total_contrast_value):
        ax.plot(s0_index, total_contrast_value[s0_index], 'ro', markersize=8, 
                label=f'S0 Index (Baseline): {s0_index}')
        # Add a vertical line at s0_index for better visibility
        ax.axvline(s0_index, color='red', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Timepoint Index')
    ax.set_ylabel('Total Contrast Value (Normalized)')
    ax.set_title('Total Contrast Signal Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some padding to the plot
    ax.margins(x=0.02, y=0.1)
    
    plt.tight_layout()
    plt.show()