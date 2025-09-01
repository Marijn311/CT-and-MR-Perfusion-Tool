import numpy as np
from scipy.ndimage import binary_erosion

def post_process_perfusion_map(generated_volume, brain_mask, img_name):
    """
    Post-process perfusion maps by normalizing and applying filters.
    This function normalizes perfusion maps using whole brain mean normalization
    to enable relative comparison between groups rather than absolute values.
    """

    # In the commercial toolbox, the perfusion maps are multiplied with some constant values.
    # This makes the perfusion values in the commercial maps closer to reality when you look at the units of these maps.
    # However, for my purpose I am comparing perfusion maps between groups. I do not care about the absolute values, but rather the relative differences.
    # Hence, I normalize both volumes with their own whole brain mean.
    # This also helps to get the commercial and generated maps in the same range for visual comparison.

    # Erode the brain mask with a flat kernel
    # Due to the patient motion, skull can fall inside the mask at the edges.
    # to get a more stable result we shrink the mask a bit before extracting the mean.
    kernel = np.ones((1, 10, 10), dtype=bool)  # x,x kernel in x-y plane, no erosion in z, since the number of slices is very low compared to in-plane resolution
    brain_mask = binary_erosion(brain_mask, kernel)

    gen_mean = generated_volume[brain_mask == 1].mean()
    generated_volume = (generated_volume - gen_mean) / gen_mean

    return generated_volume