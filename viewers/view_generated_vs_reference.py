import scipy
import numpy as np
import matplotlib.pyplot as plt


def view_generated_vs_reference(generated, reference, title, apply_mask, vmin=0, vmax=100):
    """
    Display side-by-side comparison of generated and reference perfusion maps.

    Parameters: 
        - generated (nd.array): 3D array (z,y,x) containing the generated perfusion map data
        - reference (nd.array): 3D array (z,y,x) containing the reference perfusion map data
        - title (str): Title for the comparison display
        - apply_mask (bool): If True, apply a mask to only compare voxels that are not background/nan/inf in both maps.
        - vmin (float, default=0): Lower percentile for contrast scaling
        - vmax (float, default=100): Upper percentile for contrast scaling

    Controls:
        - Mouse wheel: Navigate through time points
        - Slider: Navigate through slices
        - Arrow keys: Navigate through time points

    """
    
    # Find the mode and set it to NaN. This mode value is the background value.
    mode_reference = scipy.stats.mode(reference.flatten()).mode
    reference_masked = np.where(reference == mode_reference, np.nan, reference)
    mode_generated = scipy.stats.mode(generated.flatten()).mode
    generated_masked = np.where(generated == mode_generated, np.nan, generated)

    # The reference and generated maps have different brain masks.
    # If requested we only use the pixels which are not nan nor inf in both images.
    if apply_mask==True:
        mask = np.where(np.isfinite(reference_masked) & np.isfinite(generated_masked), 1, 0)
        generated_masked = np.where(mask > 0, generated, np.nan)
        reference_masked = np.where(mask > 0, reference, np.nan)
    
    # Calculate value ranges for display only using foreground voxels 
    gen_foreground = generated_masked[np.isfinite(generated_masked)]
    ref_foreground = reference_masked[np.isfinite(reference_masked)]
    gen_vmin, gen_vmax = np.percentile(gen_foreground[np.isfinite(gen_foreground)], [vmin, vmax])
    ref_vmin, ref_vmax = np.percentile(ref_foreground[np.isfinite(ref_foreground)], [vmin, vmax])

    # Create colormap with black background for NaN values
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')
        
    # Plot the generated and reference volumes side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    slice_idx = 0
    
    # Ensure both volumes have the same number of slices
    max_slices = min(generated_masked.shape[0], reference_masked.shape[0])
    
    # Initial display
    title = title.upper()
    img1 = ax1.imshow(generated_masked[slice_idx], cmap=cmap, vmin=gen_vmin, vmax=gen_vmax)
    ax1.set_title(f"Generated {title}")
    ax1.set_axis_off()
    
    img2 = ax2.imshow(reference_masked[slice_idx], cmap=cmap, vmin=ref_vmin, vmax=ref_vmax)
    ax2.set_title(f"reference {title}")
    ax2.set_axis_off()
    
    # Add colorbars
    plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add main title with slice info
    fig.suptitle(f"{title} Comparison\nSlice {slice_idx + 1}/{max_slices}", fontsize=14)
    
    def update_slice(new_idx):
        """Update the displayed slice based on user input."""
        nonlocal slice_idx
        slice_idx = max(0, min(new_idx, max_slices - 1))
        
        img1.set_data(generated_masked[slice_idx])
        img2.set_data(reference_masked[slice_idx])

        fig.suptitle(f"{title} Comparison\nSlice {slice_idx + 1}/{max_slices}", fontsize=14)
        fig.canvas.draw_idle()
    
    def on_scroll(event):
        """Handle scroll events for navigation."""
        if event.inaxes in [ax1, ax2]:
            if event.button == 'up':
                update_slice(slice_idx + 1)
            elif event.button == 'down':
                update_slice(slice_idx - 1)
    
    def on_key(event):
        """Handle key press events for navigation."""
        if event.key == 'up' or event.key == 'right':
            update_slice(slice_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(slice_idx - 1)
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    plt.show()

