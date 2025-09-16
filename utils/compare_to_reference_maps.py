from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from viewers.view_generated_vs_reference import *
from utils.load_and_preprocess import *
from utils.post_process import *
from config import *
import os
import numpy as np
import scipy.stats
import pandas as pd


def compare_with_reference_maps(gen_cbf_path, gen_cbv_path, gen_mtt_path, gen_ttp_path, gen_tmax_path, ref_cbf_path, ref_cbv_path, ref_mtt_path, ref_ttp_path, ref_tmax_path, mask_path):
    """
    Compare generated perfusion maps with reference maps.
    This function calculates similarity metrics between the generated and reference maps for each perfusion map that is available.
    Next it displays the images side-by-side for visual comparison.
    
    Parameters:
        - gen_cbf_path (str): Path to generated CBF map.
        - gen_cbv_path (str): Path to generated CBV map.
        - gen_mtt_path (str): Path to generated MTT map.
        - gen_ttp_path (str): Path to generated TTP map.
        - gen_tmax_path (str): Path to generated Tmax map.
        - ref_cbf_path (str): Path to reference CBF map.
        - ref_cbv_path (str): Path to reference CBV map.
        - ref_mtt_path (str): Path to reference MTT map.
        - ref_ttp_path (str): Path to reference TTP map.
        - ref_tmax_path (str): Path to reference Tmax map.
        - mask_path (str): Path to brain mask.

    Returns:
        - all_metrics(dict): Dictionary containing similarity metrics (NCC, SSIM, MAE, MSE, RMSE) 
              for each perfusion map type.
    """

    # Load the generated images and brain mask
    cbf = load_and_preprocess_perf_map(gen_cbf_path)
    cbv = load_and_preprocess_perf_map(gen_cbv_path)
    mtt = load_and_preprocess_perf_map(gen_mtt_path)
    ttp = load_and_preprocess_perf_map(gen_ttp_path)
    tmax = load_and_preprocess_perf_map(gen_tmax_path)
    mask = load_and_preprocess_perf_map(mask_path)

    # Check which reference maps exist and load only available ones
    reference_paths = {
        'cbf': ref_cbf_path,
        'cbv': ref_cbv_path,
        'mtt': ref_mtt_path,
        'ttp': ref_ttp_path,
        'tmax': ref_tmax_path
    }
    
    available_reference = {}
    for map_type, path in reference_paths.items():
        if os.path.exists(path):
            img = load_and_preprocess_perf_map(path)
            available_reference[map_type] = img
        else:
            print(f"Warning: Commercial {map_type.upper()} map not found: {path}")
    
    # Get the data arrays for available reference maps
    reference_data = {}
    for map_type, img in available_reference.items():
        reference_data[map_type] = img

    # Apply the same post-processing to the reference maps
    for map_type, img in reference_data.items():
        reference_data[map_type] = post_process_perfusion_map(img, mask, map_type)

    # Check which generated files exist
    generated_maps = {
        'cbf': cbf,
        'cbv': cbv,
        'mtt': mtt,
        'ttp': ttp,
        'tmax': tmax
    }
    
    perfusion_maps = []
    for map_type in available_reference:
        if map_type in generated_maps:
            perfusion_maps.append((map_type, generated_maps[map_type], reference_data[map_type]))
    
    # Calculate and display similarity metrics for each available map type
    all_metrics = {}
    
    print(f"\nComparing {len(perfusion_maps)} available perfusion map(s)...")
    
    for map_name, generated, reference in perfusion_maps:
        print(f"\n{map_name.upper()} Comparison:")
        print("-" * 30)
        
        # Calculate similarity metrics
        if CALCULATE_METRICS:
            metrics = calculate_similarity_metrics(generated, reference, apply_mask=True)
            # Display metrics
            print(f"Normalized Cross-Correlation: {metrics['ncc']:.3f}")
            print(f"Structural Similarity Index: {metrics['ssim']:.3f}")
            print(f"Mean Absolute Error: {metrics['mae']:.3f}")
            print(f"Mean Squared Error: {metrics['mse']:.3f}")
            print(f"Root Mean Squared Error: {metrics['rmse']:.3f}")
        else:
            metrics = {} # Empty dict if metrics calculation is disabled
        all_metrics[map_name] = metrics
        
        # Display side-by-side comparison
        if SHOW_COMPARISONS:
            view_generated_vs_reference(generated, reference, map_name, apply_mask=False, vmin=2, vmax=98)

    return all_metrics


def calculate_similarity_metrics(generated, reference, apply_mask):
    """
    Calculate comprehensive similarity metrics between generated and reference perfusion maps.
    This function computes multiple similarity metrics (NCC, SSIM, MSE, MAE, RMSE) between two perfusion maps.
    
    Parameters:
        - generated (nd.array): The generated perfusion map 3D (z,y,x).
        - reference (nd.array): The reference perfusion map to compare against 3D (z,y,x).
        - apply_mask (bool): If True, apply a mask to only compare voxels that are not background/nan/inf in both maps.
    Returns:
        - metrics (dict): Dictionary containing similarity metrics (NCC, SSIM, MAE, MSE, RMSE).
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

    # Flatten arrays
    gen_flat = generated_masked.flatten()
    ref_flat = reference_masked.flatten()

    # Remove the NaN and inf values that are in the flat arrays, because the metrics cannot deal with this.
    valid_mask = np.isfinite(gen_flat) & np.isfinite(ref_flat)
    gen_valid = gen_flat[valid_mask]
    ref_valid = ref_flat[valid_mask]
    
     
    # Calculate NCC (Normalized Cross-Correlation)
    gen_norm = (gen_valid - np.mean(gen_valid)) / np.std(gen_valid)
    ref_norm = (ref_valid - np.mean(ref_valid)) / np.std(ref_valid)
    ncc = np.nanmean(gen_norm * ref_norm)
    
    # Calculate SSIM
    # SSIM requires 2D images with finite values because it uses a 2D kernel.
    # Hence we convert NaN/inf to 0 which represents background
    # Note that the SSIM will be quite high because we are comparing the same background values.
    generated_masked = np.where(np.isfinite(generated_masked), generated_masked, 0)
    reference_masked = np.where(np.isfinite(reference_masked), reference_masked, 0)

    ssim_values = []    
    for slice_idx in range(generated_masked.shape[0]):
        gen_slice = generated_masked[slice_idx, :, :]
        ref_slice = reference_masked[slice_idx, :, :]

        # Calculate data range using 1st and 99th percentile to avoid outliers influencing the SSIM's internal normalization
        gen_p1, gen_p99 = np.percentile(gen_slice[np.isfinite(gen_slice)], [1, 99])
        ref_p1, ref_p99 = np.percentile(ref_slice[np.isfinite(ref_slice)], [1, 99])
        data_range = max(gen_p99, ref_p99) - min(gen_p1, ref_p1)

        if data_range == 0:
            print("There is a slice with zero data range, skipping SSIM for this slice.")
            continue  # Skip this slice if data range is zero to avoid division by zero

        ssim_value = ssim(gen_slice, ref_slice, data_range=data_range)
        ssim_values.append(ssim_value)
    
    # Take the average SSIM across all slices
    ssim_value = np.nanmean(ssim_values)

    # Calculate MSE and MAE 
    mse = mean_squared_error(ref_valid, gen_valid)
    mae = mean_absolute_error(ref_valid, gen_valid)
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
    Calculate average similarity metrics across multiple subjects for perfusion map comparisons.
    This function processes similarity metrics from multiple subjects, computes statistical
    summaries (mean, median, standard deviation) for each metric type, and saves results
    to an Excel file for further analysis.

    Parameters:
        - all_metrics (list): List of dictionaries containing similarity metrics for each subject.
    Returns:
        - averaged_metrics (dict): Dictionary containing averaged similarity metrics across subjects.
    """
 
    # Filter out empty dictionaries (subjects with no reference maps)
    valid_metrics = [metrics for metrics in all_metrics if metrics]
    
    print(f"Averaging metrics across {len(valid_metrics)} subjects with available reference maps "
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
                    'mean': np.nanmean(values),
                    'median': np.nanmedian(values),
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



