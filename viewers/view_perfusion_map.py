import numpy as np
from matplotlib import pyplot as plt

def view_perfusion_map(volume, title, vmin=0, vmax=100):
    """
    Display an interactive 3D perfusion map viewer.
    
    Parameters:
        - volume (nd.array): 3D array containing the perfusion data.
        - title (str): Title to display on the plot.
        - vmin (float, optional): Lower percentile for color scaling.
        - vmax (float, optional): Upper percentile for color scaling.
    Controls:
        - Mouse wheel: Navigate through time points
        - Arrow keys: Navigate through time points

    """

    vmin_global, vmax_global = np.percentile(volume, [vmin, vmax])

    fig, ax = plt.subplots()
    slice_idx = 0
    img = ax.imshow(volume[slice_idx], cmap=plt.cm.jet, vmin=vmin_global, vmax=vmax_global)
    ax.set_title(f"3D {title} Map\n(Scroll Through Slices)\nSlice {slice_idx + 1}/{volume.shape[0]}")
    plt.colorbar(img, ax=ax)

    def update_slice(new_idx):
        """Update the displayed slice based on user input."""
        nonlocal slice_idx
        slice_idx = max(0, min(new_idx, volume.shape[0] - 1))
        img.set_data(volume[slice_idx])
        ax.set_title(f"3D {title} Map\n(Scroll Through Slices)\nSlice {slice_idx + 1}/{volume.shape[0]}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        """Handle scroll events to navigate through slices."""
        if event.inaxes == ax:
            if event.button == 'up':
                update_slice(slice_idx + 1)
            elif event.button == 'down':
                update_slice(slice_idx - 1)

    def on_key(event):
        """Handle key press events to navigate through slices."""
        if event.key == 'up' or event.key == 'right':
            update_slice(slice_idx + 1)
        elif event.key == 'down' or event.key == 'left':
            update_slice(slice_idx - 1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.axis('off')
    plt.show()


