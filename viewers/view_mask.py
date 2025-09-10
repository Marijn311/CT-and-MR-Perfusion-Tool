import numpy as np
import matplotlib.pyplot as plt

def view_mask(mask, volume_list, vmin=0, vmax=100):
    """
    Display three side-by-side viewers showing brain mask, perfusion image, and masked perfusion image.
    Creates an interactive visualization with three subplots that can be navigated through slices
    using scroll wheel or arrow keys.
    
    Parameters:
        - mask: 3D numpy array representing the brain mask
        - volume_list: List of 3D nd.array perfusion volumes (z,y,x)
        - vmin (int, optional): Lower percentile for intensity scaling. Defaults to 0.
        - vmax (int, optional): Upper percentile for intensity scaling. Defaults to 100.
    Controls:
        - Mouse wheel: Navigate through time points
        - Slider: Navigate through slices
        - Arrow keys: Navigate through time points
    """
    

    # Select the first volume
    volume=volume_list[0]
    
    # Apply the mask to the images and set background to -1000
    masked_volume_array = np.where(mask > 0, volume, -1000)

    # Determine the global vmin and vmax for all three image types to ensure consistent scaling
    mask_vmin, mask_vmax = np.percentile(mask, [vmin, vmax])
    volume_vmin, volume_vmax = np.percentile(volume, [vmin, vmax])
    masked_vmin, masked_vmax = np.min(masked_volume_array[masked_volume_array != -1000]), np.max(masked_volume_array[masked_volume_array != -1000])

    # Create a figure with three subplots
    fig, (ax_mask, ax_volume, ax_masked) = plt.subplots(1, 3, figsize=(15, 5))

    # Initialize the current slice index
    current_slice = 0

    def update_slice(new_slice):
        """Update the displayed slice in all three subplots"""
        nonlocal current_slice
        current_slice = max(0, min(new_slice, mask.shape[0] - 1))
        mask_display.set_data(mask[current_slice])
        volume_display.set_data(volume[current_slice])
        masked_display.set_data(masked_volume_array[current_slice])
        ax_mask.set_title(f"3D Brain Mask\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        ax_volume.set_title(f"First Timepoint of 4D Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        ax_masked.set_title(f"First Timepoint of 4D Masked Smoothed CTP\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        """Handle scroll events to navigate through slices"""
        if event.inaxes in [ax_mask, ax_volume, ax_masked]:
            if event.button == 'up':
                update_slice(current_slice + 1)
            elif event.button == 'down':
                update_slice(current_slice - 1)

    def on_key(event):
        """Handle key press events to navigate through slices"""
        if event.key == 'up' or event.key == 'right':
            update_slice(current_slice + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(current_slice - 1)

    # Display the first slice with consistent scaling across all slices
    mask_display = ax_mask.imshow(mask[current_slice], cmap=plt.cm.gray, vmin=mask_vmin, vmax=mask_vmax)
    volume_display = ax_volume.imshow(volume[current_slice], cmap=plt.cm.gray, vmin=volume_vmin, vmax=volume_vmax)
    masked_display = ax_masked.imshow(masked_volume_array[current_slice], cmap=plt.cm.gray, vmin=masked_vmin, vmax=masked_vmax)

    # Remove axis labels
    ax_mask.set_axis_off()
    ax_volume.set_axis_off()
    ax_masked.set_axis_off()

    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_slice(0)
    plt.show()

