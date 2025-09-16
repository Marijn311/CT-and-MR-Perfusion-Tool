import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view_mask(mask, volume_list, vmin=0, vmax=100):
    """
    Display three side-by-side viewers showing brain mask, perfusion image, and masked perfusion image.
    Creates an interactive visualization with three subplots that can be navigated through slices
    using scroll wheel or arrow keys, and through time points using the bottom slider.
    
    Parameters:
        - mask: 3D numpy array representing the brain mask
        - volume_list: List of 3D nd.array perfusion volumes (z,y,x)
        - vmin (int, optional): Lower percentile for intensity scaling.
        - vmax (int, optional): Upper percentile for intensity scaling. 
    Controls:
        - Mouse wheel: Navigate through slices
        - Time slider: Navigate through time points
        - Arrow keys: Navigate through slices
    """
    

    # Initialize the current slice and time indices
    current_slice = 0
    current_time = 0
    
    # Select the initial volume
    volume = volume_list[current_time]
    
    # Apply the mask to the images and set background to -1000
    masked_volume_array = np.where(mask > 0, volume, -1000)

    # Determine the global vmin and vmax for all volumes to ensure consistent scaling
    mask_vmin, mask_vmax = np.percentile(mask, [vmin, vmax])
    
    # Calculate volume scaling across all time points
    all_volume_values = []
    all_masked_values = []
    for vol in volume_list:
        all_volume_values.extend(vol.flatten())
        masked_vol = np.where(mask > 0, vol, -1000)
        all_masked_values.extend(masked_vol[masked_vol != -1000].flatten())
    
    volume_vmin, volume_vmax = np.percentile(all_volume_values, [vmin, vmax])
    masked_vmin, masked_vmax = np.min(all_masked_values), np.max(all_masked_values)

    # Create a figure with three subplots and space for slider
    fig, (ax_mask, ax_volume, ax_masked) = plt.subplots(1, 3, figsize=(15, 5))

    def update_display(new_slice=None, new_time=None):
        """Update the displayed slice and/or time point in all three subplots"""
        nonlocal current_slice, current_time, volume, masked_volume_array
        
        if new_slice is not None:
            current_slice = max(0, min(new_slice, mask.shape[0] - 1))
        if new_time is not None:
            current_time = max(0, min(new_time, len(volume_list) - 1))
            volume = volume_list[current_time]
            masked_volume_array = np.where(mask > 0, volume, -1000)
        
        # Update displays
        mask_display.set_data(mask[current_slice])
        volume_display.set_data(volume[current_slice])
        masked_display.set_data(masked_volume_array[current_slice])
        
        # Update titles
        ax_mask.set_title(f"3D Brain Mask\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        ax_volume.set_title(f"Timepoint {current_time+1}/{len(volume_list)} of 4D image\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        ax_masked.set_title(f"Timepoint {current_time+1}/{len(volume_list)} of 4D Masked image\n(Scroll Through Slices)\nSlice {current_slice+1}/{mask.shape[0]}")
        
        fig.canvas.draw_idle()

    def on_scroll(event):
        """Handle scroll events to navigate through slices"""
        if event.inaxes in [ax_mask, ax_volume, ax_masked]:
            if event.button == 'up':
                update_display(new_slice=current_slice + 1)
            elif event.button == 'down':
                update_display(new_slice=current_slice - 1)

    def on_key(event):
        """Handle key press events to navigate through slices"""
        if event.key == 'up' or event.key == 'right':
            update_display(new_slice=current_slice + 1)
        elif event.key == 'down' or event.key == 'left':
            update_display(new_slice=current_slice - 1)

    # Display the first slice with consistent scaling across all slices
    mask_display = ax_mask.imshow(mask[current_slice], cmap=plt.cm.gray, vmin=mask_vmin, vmax=mask_vmax)
    volume_display = ax_volume.imshow(volume[current_slice], cmap=plt.cm.gray, vmin=volume_vmin, vmax=volume_vmax)
    masked_display = ax_masked.imshow(masked_volume_array[current_slice], cmap=plt.cm.gray, vmin=masked_vmin, vmax=masked_vmax)

    # Remove axis labels
    ax_mask.set_axis_off()
    ax_volume.set_axis_off()
    ax_masked.set_axis_off()

    # Create slider for time navigation (only show if there are multiple time points)
    if len(volume_list) > 1:
        slider_ax = plt.axes([0.375, 0.02, 0.25, 0.03])  # Position: [left, bottom, width, height]
        time_slider = Slider(slider_ax, 'Time', 0, len(volume_list)-1, valinit=current_time, valfmt='%d')
        
        def on_time_change(val):
            """Update display when the time slider is changed"""
            time_idx = int(time_slider.val)
            update_display(new_time=time_idx)
        
        time_slider.on_changed(on_time_change)

    # Connect event handlers
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize the display
    update_display()
    
    # Adjust layout to make room for the slider if present
    if len(volume_list) > 1:
        plt.subplots_adjust(bottom=0.15)
    
    plt.show()

