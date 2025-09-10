from matplotlib import pyplot as plt


def view_contrast_curve(bolus_data, s0_index):
    """
    Display a line graph of normalized mean signal values in the 3D perfusion volume over time with highlighted s0_index.

    Parameters: 
        - bolus_data (nd.array): 1D Array containing the normalized sum of contrast for every timepoint in the 4D perfusion volume.
        - s0_index (int): Index showing the start of the bolus. Everything before this index is averaged to compute the baseline S0.
    """
    
    # Create timepoint indices for x-axis
    timepoints = list(range(len(bolus_data)))
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the contrast curve
    ax.plot(timepoints, bolus_data, 'b-', linewidth=2, label='Total Contrast Signal')
    
    # Highlight the s0_index point
    if 0 <= s0_index < len(bolus_data):
        ax.plot(s0_index, bolus_data[s0_index], 'ro', markersize=8, 
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

