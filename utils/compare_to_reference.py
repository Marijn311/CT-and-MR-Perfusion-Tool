from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from utils.load_and_preprocess import *
from utils.post_process import *
from utils.viewers import *
from config import *
import os
import numpy as np
import scipy.stats
import pandas as pd


def compare_perfusion_maps(gen_cbf_path, gen_cbv_path, gen_mtt_path, gen_ttp_path, gen_tmax_path, com_cbf_path, com_cbv_path, com_mtt_path, com_ttp_path, com_tmax_path, mask_path):
    """
    Compare generated perfusion maps with commercially generated maps.
    
    This function calculates similarity metrics between the generated and commercial maps for each perfusion parameter.
    Next it displays the images side-by-side for visual comparison.
    
    Parameters:
        gen_cbf_path (str): Path to generated CBF map.
        gen_cbv_path (str): Path to generated CBV map.
        gen_mtt_path (str): Path to generated MTT map.
        gen_ttp_path (str): Path to generated TTP map.
        gen_tmax_path (str): Path to generated Tmax map.
        com_cbf_path (str): Path to commercially generated CBF map.
        com_cbv_path (str): Path to commercially generated CBV map.
        com_mtt_path (str): Path to commercially generated MTT map.
        com_ttp_path (str): Path to commercially generated TTP map.
        com_tmax_path (str): Path to commercially generated Tmax map.
        mask_path (str): Path to brain mask.
        plot (bool): If True, displays comparison plots for each perfusion map.

    Returns:
        dict: Dictionary containing similarity metrics (NCC, SSIM, MAE, MSE, RMSE) 
              for each perfusion map type (cbf, cbv, mtt, tmax).
    """

    # Load the generated images
    cbf = load_image(gen_cbf_path)
    cbv = load_image(gen_cbv_path)
    mtt = load_image(gen_mtt_path)
    ttp = load_image(gen_ttp_path)
    tmax = load_image(gen_tmax_path)

    # Load the mask
    brain_mask = load_image(mask_path)

    # Check which commercial files exist and load only available ones
    commercial_paths = {
        'cbf': com_cbf_path,
        'cbv': com_cbv_path,
        'mtt': com_mtt_path,
        'ttp': com_ttp_path,
        'tmax': com_tmax_path
    }
    
    available_commercial = {}
    for map_type, path in commercial_paths.items():
        if os.path.exists(path):
            img = load_image(path)
            available_commercial[map_type] = img
        else:
            print(f"Warning: Commercial {map_type.upper()} map not found: {path}")
    
    # Get the data arrays for available commercial maps
    commercial_data = {}
    for map_type, img in available_commercial.items():
        commercial_data[map_type] = img

    # Apply the same post-processing to the commercial maps
    for map_type, img in commercial_data.items():
        commercial_data[map_type] = post_process_perfusion_map(img, brain_mask, map_type)

    # Define perfusion maps for comparison
    generated_maps = {
        'cbf': cbf,
        'cbv': cbv,
        'mtt': mtt,
        'ttp': ttp,
        'tmax': tmax
    }
    
    perfusion_maps = []
    for map_type in available_commercial:
        if map_type in generated_maps:
            perfusion_maps.append((map_type, generated_maps[map_type], commercial_data[map_type]))
    
    # Calculate and display similarity metrics for each available map type
    all_metrics = {}
    
    print(f"\nComparing {len(perfusion_maps)} available perfusion map(s)...")
    
    for map_name, generated, commercial in perfusion_maps:
        print(f"\n{map_name.upper()} Comparison:")
        print("-" * 30)
        
        # Calculate similarity metrics
        metrics = calculate_similarity_metrics(generated, commercial, brain_mask, apply_mask=False)
        all_metrics[map_name] = metrics
        
        # Display metrics
        print(f"Normalized Cross-Correlation: {metrics['ncc']:.3f}")
        print(f"Structural Similarity Index: {metrics['ssim']:.3f}")
        print(f"Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"Mean Squared Error: {metrics['mse']:.3f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.3f}")
        
        # Display side-by-side comparison
        if SHOW_COMPARISONS:
            show_comparison_maps(generated, commercial, map_name, brain_mask, apply_mask=False)

    return all_metrics


def calculate_similarity_metrics(generated, commercial, mask, apply_mask):
    """
    Calculate comprehensive similarity metrics between generated and commercial perfusion maps.
    This function computes multiple similarity metrics (NCC, SSIM, MSE, MAE, RMSE) between 
    two perfusion maps while handling different brain masks and background values appropriately.
    The comparison can be made more fair by optionally masking the commercial maps with the 
    generated maps' mask.
    
    Parameters:
        generated : numpy.ndarray
            The generated perfusion map (2D or 3D array).
        commercial : numpy.ndarray
            The commercial perfusion map to compare against (same shape as generated).
        mask : numpy.ndarray
            Binary brain mask (same shape as input maps) where 1 indicates brain tissue 
            and 0 indicates background.
        apply_mask : bool
            If True, applies a mask to make sure we only consider pixels which are non-background in both images.
    """

    # Find the mode and set it to NaN. This mode value is the background value.
    mode_commercial = scipy.stats.mode(commercial.flatten()).mode
    commercial_masked = np.where(commercial == mode_commercial, np.nan, commercial)
    mode_generated = scipy.stats.mode(generated.flatten()).mode
    generated_masked = np.where(generated == mode_generated, np.nan, generated)

    # The commercial and generated maps have different brain masks.
    # We only use the pixels which are not nan nor inf in both images.
    if apply_mask==True:
        mask = np.where(np.isfinite(commercial_masked) & np.isfinite(generated_masked), 1, 0)
        generated_masked = np.where(mask > 0, generated, np.nan)
        commercial_masked = np.where(mask > 0, commercial, np.nan)

    # Flatten arrays
    gen_flat = generated_masked.flatten()
    com_flat = commercial_masked.flatten()

    # remove the NaN and inf values that may still be present in the flat arrays.
    valid_mask = np.isfinite(gen_flat) & np.isfinite(com_flat)
    gen_valid = gen_flat[valid_mask]
    com_valid = com_flat[valid_mask]
    
    # Calculate NCC (Normalized Cross-Correlation)
    gen_norm = (gen_valid - np.mean(gen_valid)) / np.std(gen_valid)
    com_norm = (com_valid - np.mean(com_valid)) / np.std(com_valid)
    ncc = np.mean(gen_norm * com_norm)
    
    # Calculate SSIM
    # Replace NaN and infinite values with 0 for SSIM calculation
    # SSIM requires 2d images with finite values, so we convert NaN/inf to 0 which represents background
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
    """Calculate average similarity metrics across multiple subjects for perfusion map comparisons.
    This function processes similarity metrics from multiple subjects, computes statistical
    summaries (mean, median, standard deviation) for each metric type, and saves results
    to an Excel file for further analysis.
    """
 
    # Filter out empty dictionaries (subjects with no commercial maps)
    valid_metrics = [metrics for metrics in all_metrics if metrics]
    
    print(f"Averaging metrics across {len(valid_metrics)} subjects with available commercial maps "
          f"(out of {len(all_metrics)} total subjects).")
    
    # Get all perfusion map types from the first valid subject
    map_types = list(valid_metrics[0].keys())
    
    # Get all similarity metrics from the first valid subject's first map type
    similarity_metrics = list(valid_metrics[0][map_types[0]].keys())
    
    averaged_metrics = {}
    
    for map_type in map_types:
        averaged_metrics[map_type] = {}
        
        for sim_metric in similarity_metrics:
            # Extract values for this specific metric across all valid subjects
            # Only include subjects that have this specific map type
            values = []
            for subject in valid_metrics:
                if map_type in subject:
                    values.append(subject[map_type][sim_metric])
            
            if values:  # Only calculate if we have values
                # Calculate statistics
                averaged_metrics[map_type][sim_metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'count': len(values)  # Add count to show how many subjects contributed
                }
            else:
                print(f"Warning: No values found for {map_type} {sim_metric}")
    
    # Print results to screen
    print("\n" + "="*80)
    print("AVERAGED METRICS ACROSS ALL SUBJECTS")
    print("="*80)
    
    for map_type in map_types:
        if map_type in averaged_metrics:
            print(f"\n{map_type.upper()}:")
            print("-" * 50)
            
            for sim_metric in similarity_metrics:
                if sim_metric in averaged_metrics[map_type]:
                    stats = averaged_metrics[map_type][sim_metric]
                    print(f"  {sim_metric.upper():5s}: Mean={stats['mean']:.3f}, "
                          f"Median={stats['median']:.3f}, Std={stats['std']:.3f}, "
                          f"N={stats['count']}")
    
    # Save results to Excel file
    rows = []
    for map_type in map_types:
        if map_type in averaged_metrics:
            for sim_metric in similarity_metrics:
                if sim_metric in averaged_metrics[map_type]:
                    stats = averaged_metrics[map_type][sim_metric]
                    rows.append({
                        'Map_Type': map_type,
                        'Similarity_Metric': sim_metric,
                        'Mean': stats['mean'],
                        'Median': stats['median'],
                        'Std': stats['std'],
                        'Count': stats['count']
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        output_file = 'averaged_similarity_metrics_for_comparison.xlsx'
        df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results to save.")
    
    return averaged_metrics