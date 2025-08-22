from scipy.ndimage import gaussian_filter
import SimpleITK as sitk


def post_process_perfusion_map(generated_volume, mask, img_name):
    """
    Post-process perfusion maps by normalizing and applying filters.
    This function normalizes perfusion maps using whole brain mean normalization
    to enable relative comparison between groups rather than absolute values.
    """

    mask = sitk.GetArrayFromImage(mask)

    # In the commercial toolbox, the perfusion maps are multiplied with some constant values.
    # This makes the perfusion values in the commercial maps closer to reality when you look at the units of these maps.
    # However, for my purpose I am comparing perfusion maps between groups. I do not care about the absolute values, but rather the relative differences.
    # Hence, I normalize both volumes with their own whole brain mean.
    # This also helps to get the commercial and generated maps in the same range for visual comparison.
    gen_mean = generated_volume[mask == 1].mean()
    generated_volume = (generated_volume - gen_mean) / gen_mean

    if img_name == "TMAX":
        generated_volume = gaussian_filter(generated_volume, sigma=1.0)

    return generated_volume