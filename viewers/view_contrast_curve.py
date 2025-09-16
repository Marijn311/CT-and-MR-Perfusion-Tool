from matplotlib import pyplot as plt


def view_contrast_curve(bolus_data, time_index, s0_index):
    """
    Display a line graph of normalized mean signal values in the 3D perfusion volume over time with highlighted s0_index.

    Parameters: 
        - bolus_data (nd.array): 1D Array containing the normalized sum of contrast for every timepoint in the 4D perfusion volume.
        - time_index (list): List of time indices in seconds corresponding to each volume.
        - s0_index (int): Index showing the start of the bolus. Everything before this index is averaged to compute the baseline S0.
    """
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the contrast curve
    ax.plot(time_index, bolus_data, 'b-', linewidth=2, label='Normalized Mean (Whole Brain) Signal over Time')
    
    # Convert S0 index to corresponding time in seconds
    s0_time = time_index[s0_index] 

    # Highlight the s0_index point
    ax.plot(s0_time, bolus_data[s0_index], 'ro', markersize=8, label=f'Start of bolus: {s0_time}')
    # Add a vertical line at s0_index for better visibility
    ax.axvline(s0_time, color='red', linestyle='--', alpha=0.7)

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Normalized Mean (Whole Brain) Signal')
    ax.set_title('Normalized Mean (Whole Brain) Signal Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some padding to the plot
    ax.margins(x=0.02, y=0.1)
    
    plt.tight_layout()
    plt.show()

