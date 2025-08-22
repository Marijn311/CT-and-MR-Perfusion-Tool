import numpy as np
import nibabel as nib
import scipy.stats
import SimpleITK as sitk
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from utils.viewers import show_comparison_maps

def compare_perfusion_maps(cbf, cbv, mtt, tmax, com_cbf_path, com_cbv_path, com_mtt_path, com_tmax_path, mask, plot):
    """
    Compare generated perfusion maps with commercially generated maps.
    
    This function loads commercial perfusion maps, reorients them to match the generated maps,
    normalizes them using their own whole brain mean values, and calculates similarity metrics
    between the generated and commercial maps for each perfusion parameter.
    
    Parameters:
        cbf (numpy.ndarray): Generated cerebral blood flow map.
        cbv (numpy.ndarray): Generated cerebral blood volume map.
        mtt (numpy.ndarray): Generated mean transit time map.
        tmax (numpy.ndarray): Generated time to maximum map.
        com_cbf_path (str): Path to commercially generated CBF map.
        com_cbv_path (str): Path to commercially generated CBV map.
        com_mtt_path (str): Path to commercially generated MTT map.
        com_tmax_path (str): Path to commercially generated Tmax map.
        mask (SimpleITK.Image): Brain mask used for comparison and normalization.
        plot (bool): If True, displays comparison plots for each perfusion map.

    Returns:
        dict: Dictionary containing similarity metrics (NCC, SSIM, MAE, MSE, RMSE) 
              for each perfusion map type (cbf, cbv, mtt, tmax).
    """

    mask = sitk.GetArrayFromImage(mask)

    # Load commercially generated maps
    com_cbf_img = nib.load(com_cbf_path)
    com_cbv_img = nib.load(com_cbv_path)
    com_mtt_img = nib.load(com_mtt_path)
    com_tmax_img = nib.load(com_tmax_path)

    # Reorient to ras+ (to match generated maps)
    com_cbf_img = nib.as_closest_canonical(com_cbf_img)
    com_cbv_img = nib.as_closest_canonical(com_cbv_img)
    com_mtt_img = nib.as_closest_canonical(com_mtt_img)
    com_tmax_img = nib.as_closest_canonical(com_tmax_img)
    
    # Get the data arrays
    com_cbf = com_cbf_img.get_fdata()
    com_cbv = com_cbv_img.get_fdata()
    com_mtt = com_mtt_img.get_fdata()
    com_tmax = com_tmax_img.get_fdata()
    
    # Transpose commercial maps to match the orientation of the generated maps (slices, height, width)
    if com_cbf.ndim == 3:
        com_cbf = np.transpose(com_cbf, (2, 1, 0))
        com_cbv = np.transpose(com_cbv, (2, 1, 0))
        com_mtt = np.transpose(com_mtt, (2, 1, 0))
        com_tmax = np.transpose(com_tmax, (2, 1, 0))

    # Normalize the commercial maps with their own whole brain mean, to match the normalization of generated maps
    com_cbf_mean = com_cbf[mask == 1].mean()
    com_cbv_mean = com_cbv[mask == 1].mean()
    com_mtt_mean = com_mtt[mask == 1].mean()
    com_tmax_mean = com_tmax[mask == 1].mean()
    com_cbf = (com_cbf - com_cbf_mean) / com_cbf_mean
    com_cbv = (com_cbv - com_cbv_mean) / com_cbv_mean
    com_mtt = (com_mtt - com_mtt_mean) / com_mtt_mean
    com_tmax = (com_tmax - com_tmax_mean) / com_tmax_mean

    # Define perfusion maps for comparison
    perfusion_maps = [
        ('cbf', cbf, com_cbf),
        ('cbv', cbv, com_cbv),
        ('mtt', mtt, com_mtt),
        ('tmax', tmax, com_tmax)
    ]
    
    # Calculate and display similarity metrics for each map type
    all_metrics = {}
    for map_name, generated, commercial in perfusion_maps:
        print(f"\n{map_name} Comparison:")
        print("-" * 30)
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(generated, commercial, mask, apply_mask_to_commercial=True)
        all_metrics[map_name] = metrics
        
        # Display metrics
        print(f"Normalized Cross-Correlation: {metrics['ncc']:.3f}")
        print(f"Structural Similarity Index: {metrics['ssim']:.3f}")
        print(f"Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"Mean Squared Error: {metrics['mse']:.3f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.3f}")
        
        # Display side-by-side comparison
        if plot:
            show_comparison_maps(generated, commercial, map_name, mask, apply_mask_to_commercial=False)

    return all_metrics


def calculate_similarity_metrics(generated, commercial, mask, apply_mask_to_commercial):
    """Calculate comprehensive similarity metrics between generated and commercial perfusion maps.
    This function computes multiple similarity metrics (NCC, SSIM, MSE, MAE, RMSE) between 
    two perfusion maps while handling different brain masks and background values appropriately.
    The comparison can be made more fair by optionally masking the commercial maps with the 
    generated maps' mask.
    Parameters
    ----------
    generated : numpy.ndarray
        The generated perfusion map (2D or 3D array).
    commercial : numpy.ndarray
        The commercial perfusion map to compare against (same shape as generated).
    mask : numpy.ndarray
        Binary brain mask (same shape as input maps) where 1 indicates brain tissue 
        and 0 indicates background.
    apply_mask_to_commercial : bool
        If True, applies the mask to the commercial map for fair comparison.
        If False, automatically detects and masks the background using mode value.
    Returns
    -------
    dict
        Dictionary containing similarity metrics:
        - 'ncc' : float
            Normalized Cross-Correlation coefficient [-1, 1]
        - 'ssim' : float
            Structural Similarity Index [0, 1]
        - 'mse' : float
            Mean Squared Error
        - 'mae' : float
            Mean Absolute Error
        - 'rmse' : float
            Root Mean Squared Error    
    """

    # The commercial maps and the maps generated by this toolbox have different brain masks.
    # To make the comparison more fair, we mask the commercial maps with the generated maps.
    if apply_mask_to_commercial==True:
        commercial = commercial * mask

    # Set background to NaN for both volumes to ensure calculation are not done on the background
    generated_masked = np.where(mask > 0, generated, np.nan)
    if apply_mask_to_commercial==True:
        commercial_masked = np.where(mask > 0, commercial, np.nan)
    else:
        # Find the mode and set it to NaN. This mode value is the background value.
        mode_value = scipy.stats.mode(commercial.flatten()).mode
        commercial_masked = np.where(commercial == mode_value, np.nan, commercial)

    # Flatten arrays
    gen_flat = generated_masked.flatten()
    com_flat = commercial_masked.flatten()

    # Create mask for valid values (only keep pixels where both arrays are not NaN and not inf)
    valid_mask = np.isfinite(gen_flat) & np.isfinite(com_flat)
    gen_valid = gen_flat[valid_mask]
    com_valid = com_flat[valid_mask]
    
    # Calculate NCC (Normalized Cross-Correlation)
    gen_norm = (gen_valid - np.mean(gen_valid)) / np.std(gen_valid)
    com_norm = (com_valid - np.mean(com_valid)) / np.std(com_valid)
    ncc = np.mean(gen_norm * com_norm)
    
    # Calculate SSIM
    # Replace NaN and infinite values with 0 for SSIM calculation
    # SSIM requires finite values, so we convert NaN/inf to 0 which represents background
    # Note that the SSIM willl be quite high because we are comparing the same background values.
    generated_masked = np.where(np.isfinite(generated_masked), generated_masked, 0)
    commercial_masked = np.where(np.isfinite(commercial_masked), commercial_masked, 0)
    # Calculate data range using 98th percentile to avoid outliers influencing the ssim normalization
    gen_p1, gen_p99 = np.percentile(generated_masked[np.isfinite(generated_masked)], [1, 99])
    com_p1, com_p99 = np.percentile(commercial_masked[np.isfinite(commercial_masked)], [1, 99])
    data_range = max(gen_p99, com_p99) - min(gen_p1, com_p1)
    ssim_value = ssim(generated_masked, commercial_masked, data_range=data_range)

    # Calculate MSE and MAE 
    mse = mean_squared_error(com_valid, gen_valid)
    mae = mean_absolute_error(com_valid, gen_valid)
    rmse = np.sqrt(mse)
    
    return {
        'ncc': ncc,
        'ssim': ssim_value,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }


def average_metrics_on_dataset(all_metrics):
    """
    all_metrics is a list of dictionaries containing similarity metrics for each comparison that was made.
    This function averages the metrics across all subjects and returns the result.

    for each metric it gives the mean, median, and standard deviation across all subjects.

    """ 
    
    # Get all perfusion map types from the first subject
    map_types = list(all_metrics[0].keys())
    
    # Get all similarity metrics from the first subject's first map type
    similarity_metrics = list(all_metrics[0][map_types[0]].keys())
    
    averaged_metrics = {}
    
    for map_type in map_types:
        averaged_metrics[map_type] = {}
        
        for sim_metric in similarity_metrics:
            # Extract values for this specific metric across all subjects
            values = [subject[map_type][sim_metric] for subject in all_metrics]
            
            # Calculate statistics
            averaged_metrics[map_type][sim_metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values)
            }
    
    # Print results to screen
    print("\n" + "="*80)
    print("AVERAGED METRICS ACROSS ALL SUBJECTS")
    print("="*80)
    
    for map_type in map_types:
        print(f"\n{map_type.upper()}:")
        print("-" * 50)
        
        for sim_metric in similarity_metrics:
            stats = averaged_metrics[map_type][sim_metric]
            print(f"  {sim_metric.upper():5s}: Mean={stats['mean']:.3f}, "
                  f"Median={stats['median']:.3f}, Std={stats['std']:.3f}")
    
    # Save results to Excel file
    rows = []
    for map_type in map_types:
        for sim_metric in similarity_metrics:
            stats = averaged_metrics[map_type][sim_metric]
            rows.append({
                'Map_Type': map_type,
                'Similarity_Metric': sim_metric,
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std']
            })
    
    df = pd.DataFrame(rows)
    output_file = 'averaged_similarity_metrics_for_comparison.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return averaged_metrics