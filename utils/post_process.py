import numpy as np
from scipy.ndimage import binary_erosion

def post_process_perfusion_map(perfusion_map, mask, perf_map_type):
    """
    Post-process perfusion maps.I chose to normalize the maps using whole brain mean. 
    For my purposes I dont need absolute quantification of perfusion values, but relative comparison between groups.

    Parameters:
        - perfusion_map (nd.array): 3D array (z,y,x) of a perfusion map to be post-processed.
        - mask (nd.array): 3D array (z,y,x) of the brain mask.
        - perf_map_type (str): Type of perfusion map ('CBF', 'CBV', etc) in case different maps require specific processing.
    Returns:
        - perfusion_map (nd.array): 3D array (z,y,x) the post-processed perfusion map.
    
    """

    # Due to slight patient motion, the skull can still fall inside the mask at the edges.
    # Therefore we erode the brain mask with a flat kernel to get a more accurate mean value.  
    kernel_size = mask.shape[2] // 15
    kernel = np.ones((1, kernel_size, kernel_size), dtype=bool)  # x,x kernel in x-y plane, no erosion in z, since the number of slices is very low compared to in-plane resolution
    mask = binary_erosion(mask, kernel)
    gen = perfusion_map[mask == 1]
    gen_mean = np.mean(gen)
    perfusion_map = (perfusion_map - gen_mean) / gen_mean

    return perfusion_map