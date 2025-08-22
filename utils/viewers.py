import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy
from utils.aif import gv


def view_4d_img(img_list, title, window=(200, 500)):
    """Visualize a 4D volume in img_list as a 2D slice taken from the middle of the 3D volume.
    You can scroll through the time in this middle slice. 
    Besides the middle slice we show the maximum intensity projection (MIP) of the middle third slices of the volume."""

    current_idx = 0
    
    # Pre-calculate vmin and vmax for all volumes to ensure consistent scaling
    all_mip_values = []
    all_original_values = []
    processed_mip_volumes = []
    processed_original_volumes = []
    
    for im in img_list:
        # Maximum intensity projection from the middle third of the volume
        imSlice_mip = sitk.IntensityWindowing(
            sitk.MaximumProjection(im[:, :, im.GetDepth()//3:im.GetDepth()*2//3], 2)[:, :, 0],
            window[0]-window[1]/2, window[0]+window[1]/2, 0, 255
        )
        
        # Original image - take middle slice
        middle_slice = im.GetDepth() // 2
        imSlice_original = sitk.IntensityWindowing(
            im[:, :, middle_slice],
            window[0]-window[1]/2, window[0]+window[1]/2, 0, 255
        )
        
        imSlice_mip_array = sitk.GetArrayFromImage(imSlice_mip).astype('uint8')
        imSlice_original_array = sitk.GetArrayFromImage(imSlice_original).astype('uint8')
        
        processed_mip_volumes.append(imSlice_mip_array)
        processed_original_volumes.append(imSlice_original_array)
        all_mip_values.extend(imSlice_mip_array.flatten())
        all_original_values.extend(imSlice_original_array.flatten())
    
    mip_vmin, mip_vmax = np.percentile(all_mip_values, [2, 98])
    original_vmin, original_vmax = np.percentile(all_original_values, [2, 98])

    def update_volume(new_idx):
        nonlocal current_idx
        current_idx = max(0, min(new_idx, len(img_list) - 1))
        
        # Update original image
        img_original.set_data(processed_original_volumes[current_idx])
        ax_original.set_title(f"Middle Slice of 4D Volume\n(Scroll Through Time)\nTimepoint {current_idx+1}/{len(img_list)}")

        # Update MIP image
        img_mip.set_data(processed_mip_volumes[current_idx])
        ax_mip.set_title(f"Maximum Intensity Projection of Middle Third of Slices\n(Scroll Through Time)\nTimepoint {current_idx+1}/{len(img_list)}")

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




def view_brain_mask(brain_mask, img):
    """
    Visualize the brain mask, the corresponding smoothed images, and the masked images side by side.
    """

    # Select the first timepoint
    img=img[0]

    # Convert the brain mask and images to numpy arrays
    brain_mask_array = sitk.GetArrayFromImage(brain_mask)
    img_array = sitk.GetArrayFromImage(img)
    
    # Apply the mask to the images and set background to -1000
    masked_img_array = np.where(brain_mask_array > 0, img_array, -1000)

    # Pre-calculate vmin and vmax for all volumes to ensure consistent scaling
    mask_vmin, mask_vmax = np.percentile(brain_mask_array, [2, 98])
    img_vmin, img_vmax = np.percentile(img_array, [2, 98])
    masked_vmin, masked_vmax = np.min(masked_img_array[masked_img_array != -1000]), np.max(masked_img_array[masked_img_array != -1000])

    # Create a figure with three subplots
    fig, (ax_mask, ax_img, ax_masked) = plt.subplots(1, 3, figsize=(15, 5))

    current_slice = 0

    def update_slice(new_slice):
        nonlocal current_slice
        current_slice = max(0, min(new_slice, brain_mask_array.shape[0] - 1))
        mask_display.set_data(brain_mask_array[current_slice])
        img_display.set_data(img_array[current_slice])
        masked_display.set_data(masked_img_array[current_slice])
        ax_mask.set_title(f"3D Brain Mask\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask_array.shape[0]}")
        ax_img.set_title(f"First Timepoint of 4D Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask_array.shape[0]}")
        ax_masked.set_title(f"First Timepoint of 4D Masked Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{brain_mask_array.shape[0]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes in [ax_mask, ax_img, ax_masked]:
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
    mask_display = ax_mask.imshow(brain_mask_array[current_slice], cmap=plt.cm.gray, vmin=mask_vmin, vmax=mask_vmax)
    img_display = ax_img.imshow(img_array[current_slice], cmap=plt.cm.gray, vmin=img_vmin, vmax=img_vmax)
    masked_display = ax_masked.imshow(masked_img_array[current_slice], cmap=plt.cm.gray, vmin=masked_vmin, vmax=masked_vmax)

    # Remove axis labels
    ax_mask.set_axis_off()
    ax_img.set_axis_off()
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



def view_aif_selection(img, time_index, img_ctc, s0_index, aif_propperties, aif_candidate_segmentations, brain_mask, window=(100, 1200)):
    """
    Visualize the region from which the AIF is extracted.
    Visualize the CTC images in this region.
    Visualize the contrast concentration curve in the AIF region and the fitted gamma variate function.
    """

    majority_slice = majority(np.where(aif_candidate_segmentations)[0])
    majority_img = img[(img_ctc.shape[0] - s0_index)//2 + s0_index][:, :, int(majority_slice)]
    majority_img = sitk.IntensityWindowing(majority_img, window[0]-window[1]/2, window[0]+window[1]/2, 0, 255)
    majority_labels = sitk.GetImageFromArray(aif_candidate_segmentations[majority_slice].astype('uint8'))
    majority_labels.CopyInformation(majority_img)
    majority_img = sitk.GetArrayFromImage(sitk.LabelOverlay(majority_img, majority_labels))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # First subplot: AIF Candidate Segmentation Overlay
    ax1.imshow(majority_img.astype('uint8'))
    ax1.set_title('Slice With AIF Candidate Segmentation\nIn purple: Region From Which AIF is Extracted')
    ax1.set_axis_off()
    
    # Second subplot: Scrollable CTC image
    current_time_idx = 0
    ctc_slice_data = img_ctc[:, int(majority_slice), :, :]  # Get the same slice as the overlay
    
    # Get brain mask for the same slice
    brain_mask_array = sitk.GetArrayFromImage(brain_mask)
    brain_maskSlice = brain_mask_array[int(majority_slice)]  # Extract the same slice from brain mask
    
    # Pre-calculate global vmin and vmax for consistent scaling
    all_masked_values = []
    for t in range(ctc_slice_data.shape[0]):
        masked_image = ctc_slice_data[t].copy()
        masked_image[brain_maskSlice == 0] = 0
        all_masked_values.extend(masked_image[brain_maskSlice == 1].flatten())  # Only include brain voxels
    
    # Set vmin and vmax based on the 2nd and 98th percentiles of all masked values
    ctc_vmin, ctc_vmax = np.percentile(all_masked_values, [2, 98])
    
    def update_ctc_display(time_idx):
        nonlocal current_time_idx
        current_time_idx = max(0, min(time_idx, ctc_slice_data.shape[0] - 1))
        
        # Apply brain mask to the CTC image
        ctc_image = ctc_slice_data[current_time_idx].copy()
        ctc_image[brain_maskSlice == 0] = 0  # Set pixels outside brain mask to 0
        
        ctc_img.set_data(ctc_image)
        ax2.set_title(f'Contrast Signal in Slice With AIF Selection\n(Scroll Through Time)\nTimepoint {current_time_idx+1}/{ctc_slice_data.shape[0]}\nTime: {time_index[current_time_idx]:.1f}s')

        # Update the vertical line on the signal plot to show current time
        if hasattr(update_ctc_display, 'time_line'):
            update_ctc_display.time_line.remove()
        update_ctc_display.time_line = ax3.axvline(time_index[current_time_idx], color='red', linestyle='--', alpha=0.7, label='Current Time')
        ax3.legend()
        fig.canvas.draw_idle()
    
    def on_scroll(event):
        if event.inaxes == ax2:
            if event.button == 'up':
                update_ctc_display(current_time_idx + 1)
            elif event.button == 'down':
                update_ctc_display(current_time_idx - 1)
    
    def on_key(event):
        if event.key == 'up' or event.key == 'right':
            update_ctc_display(current_time_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_ctc_display(current_time_idx - 1)
    
    # Initialize CTC display with brain mask applied and global scaling
    initial_ctc_image = ctc_slice_data[current_time_idx].copy()
    initial_ctc_image[brain_maskSlice == 0] = 0  # Apply brain mask
    ctc_img = ax2.imshow(initial_ctc_image, cmap=plt.cm.gray, vmin=ctc_vmin, vmax=ctc_vmax)
    ax2.set_axis_off()
    
    # Third subplot: Signal plot
    ax3.plot(time_index, img_ctc[:, aif_candidate_segmentations].mean(axis=1), label='Mean AIF Candidate Signal')
    ax3.plot(np.linspace(0, max(time_index)*1.5, 1000), gv(np.linspace(0, max(time_index)*1.5, 1000), *aif_propperties), label='Fitted Gamma Variate')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Signal Intensity')
    ax3.set_title('AIF Diagnostic: Signal vs Fitted Model')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initialize the display
    update_ctc_display(0)
    
    plt.tight_layout()
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

def show_comparison_maps(generated_volume, commercial_volume, title, mask, apply_mask_to_commercial):
    """
    Display side-by-side comparison of generated and commercial perfusion maps.
    """
    
    # The commercial maps and the maps generated by this toolbox have different brain masks.
    # To make the comparison more fair, we can mask the commercial maps with the generated maps.
    if apply_mask_to_commercial==True:
        commercial_volume = commercial_volume * mask

    # Set background to NaN for both volumes to ensure it appears black
    generated_volume_masked = np.where(mask > 0, generated_volume, np.nan)
    if apply_mask_to_commercial==True:
        commercial_volume_masked = np.where(mask > 0, commercial_volume, np.nan)
    else:
        # Find the mode and set it to NaN. This mode value is the background value.
        mode_value = scipy.stats.mode(commercial_volume.flatten()).mode
        commercial_volume_masked = np.where(commercial_volume == mode_value, np.nan, commercial_volume)

    # Calculate value ranges for display only using foreground voxels (Take the 2nd and 98th percentiles to avoid outliers which can skew the color mapping)
    gen_foreground = generated_volume_masked[mask > 0]
    com_foreground = commercial_volume_masked[mask > 0]
    
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
    max_slices = min(generated_volume_masked.shape[0], commercial_volume_masked.shape[0])
    
    # Initial display
    title = title.upper()
    img1 = ax1.imshow(generated_volume_masked[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Generated {title}")
    ax1.set_axis_off()
    
    img2 = ax2.imshow(commercial_volume_masked[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
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
        
        img1.set_data(generated_volume_masked[slice_idx])
        img2.set_data(commercial_volume_masked[slice_idx])

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