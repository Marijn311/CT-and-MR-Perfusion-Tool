import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def view_4d_img(img_list, title, projection, vmin=0, vmax=100):
    """
    Visualize a 4D volume.
    Displays two side-by-side views: individual slices (left) and a minimum or maximum intensity projection 
    of middle third slices (right). Navigation via mouse wheel (time) and slider (slices).
    
    Parameters:
        - img_list (list): List of 3D image volumes in nd.array (z,y,x) representing time series
        - title (str): Title for the visualization window
        - projection (str):          'min' for minimum intensity projection, 
                                     'max' for maximum intensity projection.
        - vmin (int, optional): Lower percentile for colormap scaling. Defaults to 0.
        - vmax (int, optional): Upper percentile for colormap scaling. Defaults to 100.
    Controls:
        - Mouse wheel: Navigate through time points
        - Slider: Navigate through slices
        - Arrow keys: Navigate through time points
    """

    # Initial indices
    current_time_idx = 0
    current_slice_idx = img_list[0].shape[0] // 2  
    
    # Get dimensions from first image
    img_array = np.array(img_list[0])
    num_slices = img_array.shape[0]
    
    # Store data to determine the global vmin, vmax, and the intensity projections 
    all_mip_values = []
    all_original_values = []
    processed_mip_volumes = []
    
    for img in img_list:
        
        # Define the middle third of slices for the intensity projection.
        # We use the middle third of slices to avoid the top slices (which are progressively smaller 
        # so have the skull more central) and the bottom slices (which often contain artifacts) from distorting the projection.
        img_array = np.array(img)
        depth = img_array.shape[0]
        middle_third = img_array[depth//3:depth*2//3, :, :]

        # We use minimum intensity projection for mrp and maximum intensity projection for ctp.
        # Since the contrast agent causes a drop in intensity in mrp and a rise in intensity in ctp.
        if projection == 'min':
            middle_slice_projection = np.min(middle_third, axis=0)
        if projection == 'max':
            middle_slice_projection = np.max(middle_third, axis=0)

        processed_mip_volumes.append(middle_slice_projection)
        all_mip_values.extend(middle_slice_projection.flatten())
        
        # Get all slice values for global scaling
        for slice_idx in range(img_array.shape[0]):
            all_original_values.extend(img[slice_idx, :, :].flatten())

    # Determine global vmin and vmax for consistent scaling across time points
    mip_vmin, mip_vmax = np.percentile(all_mip_values, [vmin, vmax])
    original_vmin, original_vmax = np.percentile(all_original_values, [vmin, vmax])


    def update_display(new_time_idx=None, new_slice_idx=None):
        """ Update the displayed images based on new time or slice indices."""
        nonlocal current_time_idx, current_slice_idx
        
        if new_time_idx is not None:
            current_time_idx = max(0, min(new_time_idx, len(img_list) - 1))
        if new_slice_idx is not None:
            current_slice_idx = max(0, min(new_slice_idx, num_slices - 1))
        
        # Update original image
        current_slice_data = img_list[current_time_idx][current_slice_idx, :, :]
        img_original.set_data(current_slice_data)
        ax_original.set_title(f"Slice {current_slice_idx+1}/{num_slices} of 4D Volume\n(Mouse Wheel: Time, Slider: Slice)\nTimepoint {current_time_idx+1}/{len(img_list)}")

        # Update MIP/MinIP image
        img_mip.set_data(processed_mip_volumes[current_time_idx])
        projection_name = "Minimum Intensity Projection" if projection == 'min' else "Maximum Intensity Projection"
        ax_mip.set_title(f"{projection_name} of Middle Third of Slices\n(Mouse Wheel: Time)\nTimepoint {current_time_idx+1}/{len(img_list)}")

        fig.suptitle(f"{title}", fontsize=14)
        fig.canvas.draw_idle()

    def on_scroll(event):
        """ Handle mouse wheel events to navigate through time points."""
        if event.inaxes in [ax_original, ax_mip]:
            if event.button == 'up':
                update_display(new_time_idx=current_time_idx + 1)
            elif event.button == 'down':
                update_display(new_time_idx=current_time_idx - 1)

    def on_key(event):
        """ Handle key press events to navigate through time points."""
        if event.key == 'up' or event.key == 'right':
            update_display(new_time_idx=current_time_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_display(new_time_idx=current_time_idx - 1)

    # Create figure with two subplots side by side and space for slider
    fig, (ax_original, ax_mip) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initialize displays
    initial_slice_data = img_list[0][current_slice_idx, :, :]
    img_original = ax_original.imshow(initial_slice_data, cmap=plt.cm.Greys_r, vmin=original_vmin, vmax=original_vmax)
    ax_original.set_axis_off()
    img_mip = ax_mip.imshow(processed_mip_volumes[0], cmap=plt.cm.Greys_r, vmin=mip_vmin, vmax=mip_vmax)
    ax_mip.set_axis_off()

    # Create slider for slice navigation
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])  # Position: [left, bottom, width, height]
    slice_slider = Slider(slider_ax, 'Slice', 0, num_slices-1, valinit=current_slice_idx, valfmt='%d')
    
    def on_slice_change(val):
        """ Update display when the slice slider is changed."""
        slice_idx = int(slice_slider.val)
        update_display(new_slice_idx=slice_idx)
    
    slice_slider.on_changed(on_slice_change)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_display()
    plt.subplots_adjust(bottom=0.15)  # Make room for the slider
    plt.show()

