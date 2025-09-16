from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import curve_fit
import cc3d
import numpy as np
import pandas as pd


def gamma_variate(t, t0, alpha, beta, amplitude=1.0):
    """
    Helper function, which defines the Gamma variate function for modeling arterial input function.
    Standard gamma variate: A * (t-t0)^alpha * exp(-(t-t0)/beta) for t > t0
    
    Parameters:
        t (nd.array): 1D time array, same length as the contrast curve. Contains a time value for each time point.
        t0 (float): time delay/onset time
        alpha (float): shape parameter
        beta (float): time constant
        amplitude (float): peak amplitude scaling factor
    Returns:
        -result (nd.array): 1D array of modeled arterial input function
    """
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    result = np.zeros_like(t_shifted)
    mask = t > t0
    result[mask] = amplitude * (t_shifted[mask]**alpha) * np.exp(-t_shifted[mask]/beta)
    return result


def fit_gamma_variate(time_index, curve, sigma=None):
    """
    Fit a gamma variate function to the given contrast curve using non-linear least squares optimization.
    and the helper function `gamma_variate`.
    
    Parameters:
        - time_index (nd.array): 1D array of time points corresponding to the contrast curve
        - curve (nd.array): 1D array of contrast values to fit the gamma variate to
        - sigma (nd.array, optional): 1D array of standard deviations for each point in the curve, used as weights in fitting
    Returns:
        - properties (nd.array): 1D array containing the optimized parameters [t0, alpha, beta, amplitude] for the fitted gamma variate function
    """
    # bounds to in which to search: [t0_min, alpha_min, beta_min, amplitude_min], [t0_max, alpha_max, beta_max, amplitude_max]
    properties, _ = curve_fit(gamma_variate, time_index, curve, bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2]), sigma=sigma)
    return properties


def calculate_signal_smoothness(signal):
    """
    Calculate the smoothness of a signal using the sum of squared second derivatives.
    Lower values indicate smoother signals that better resemble the gamma variate function.
    
    Parameters:
        - signal (nd.array): 1D signal array
        
    Returns:
        - float: Smoothness metric (lower values = smoother signal)
    """
    if len(signal) < 3:
        return float('inf')  # Cannot calculate second derivative for very short signals
    
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    second_derivative = np.diff(signal, n=2)
    smoothness = np.sum(second_derivative ** 2)
    smoothness = smoothness / len(signal)
    
    return smoothness


def majority(array, ignore=None):
    """
    Compute the majority label in a 3D array.
    """
    labels, counts = np.unique(array.ravel(), return_counts=True)
    if ignore is not None:
        mask = labels.isin(ignore)
        counts = counts[~mask]
        labels = labels[~mask]
        return labels[np.argmax(counts)]
    else:
        return labels[np.argmax(counts)]


def determine_aif(ctc_volumes, time_index, mask, ttp, dilate_radius=5, erode_radius=2,
            max_vol_thres=50, min_vol_thres=5, smoothness_threshold=0.04):
    """
    Identify the Arterial Input Function (AIF) from the contrast time curves (CTC).
    This function uses adaptive thresholding with progressively relaxed criteria to identify AIF candidates
    based on AUC (Area Under Curve), TTP (Time-to-Peak), and signal smoothness. Morphological 
    operations are applied to refine candidates, and the best candidate is selected based on a score 
    combining volume, peak difference, and fitting error.

    Parameters:
        - ctc_volumes (list): List of 3D nd.arrays (z,y,x) representing contrast time curves (CTC).
        - time_index (list): List of time indices corresponding to the CTC data.
        - mask (nd.array): 3D binary nd.array (z,y,x) brain mask.
        - ttp (nd.array): 3D nd.array (z,y,x) time-to-peak values for each voxel.
        - dilate_radius (int, optional): Radius for dilation operation during morphological filtering.
        - erode_radius (int, optional): Radius for erosion operation during morphological filtering. 
        - max_vol_thres (float, optional): Maximum volume threshold for AIF candidates.
        - min_vol_thres (float, optional): Minimum volume threshold for AIF candidates. 
        - smoothness_threshold (float, optional): Maximum allowed smoothness metric for AIF candidates.
                        Candidates with higher values (less smooth) will be rejected. 

    Returns:
        - aif_properties (nd.array): 1D array storing the parameters of the best-fit gamma variate function for the AIF.
        - aif_mask (nd.array): 3D Binary mask of the selected AIF region.
        
    """

    # Due to slight patient motion or sub-optimal masking, skull signal can be inside the mask near the mask's edges. 
    # We want to prevent the skull from being selected as AIF
    # Hence we erode the brain mask with a 2D kernel
    kernel_size = mask.shape[2] // 15
    kernel = np.ones((1, kernel_size, kernel_size), dtype=bool)  # x,x kernel in x-y plane, no erosion in z, since the number of slices is very low compared to in-plane resolution
    mask = binary_erosion(mask, kernel)

    # Convert the list of 3D arrays to a 4D array
    ctc_volumes = np.stack(ctc_volumes, axis=0)

    # Get AUC and TTP distributions
    auc_values = ctc_volumes.sum(0)[mask > 0]
    ttp_values = ttp[mask > 0]

    # Adaptive thresholding approach
    # Set the TTP and AUC thresholds based on the TTP and AUC distributions, as a percentage of the distribution
    attempts = [(np.percentile(ttp_values, 100-p), np.percentile(auc_values, p)) 
                for p in range(95, 4, -10)]

    # Try to find an AIF with the thresholds for AUC and TTP. If it fails, try the next set of thresholds. Thresholds are progressively relaxed.
    for attempt_ttp, attempt_auc in attempts:
        # Try different smoothness thresholds if no candidates found
        smoothness_attempts = [smoothness_threshold, smoothness_threshold * 2, smoothness_threshold * 5, smoothness_threshold * 10]
        
        for current_smoothness_threshold in smoothness_attempts:
            # Identify initial AIF candidates based on AUC and TTP thresholds.
            aif_candidate = (ctc_volumes.sum(0) * mask > attempt_auc) * (ttp < attempt_ttp) * (ttp > 0)

            # Apply morphological operations (dilation followed by erosion) to refine AIF candidates
            aif_candidate = binary_erosion(
                binary_dilation(aif_candidate, np.ones([1, dilate_radius, dilate_radius], bool)), 
                np.ones([1, erode_radius, erode_radius], bool)
            )

            # Perform connected component analysis to separate individual AIF candidates
            aif_candidate, nr_components = cc3d.connected_components(aif_candidate, return_N=True)

            # Skip to next attempt (with more relaxed thresholds) if no components found
            if nr_components == 0:
                continue

            # Initialize a list to store properties of AIF candidates
            cands = []

            # Iterate through each connected component (candidate region)
            for idx in range(1, nr_components + 1):
                curve_candidates = aif_candidate == idx  # Extract the current candidate region
                vol = curve_candidates.sum()  # Calculate the volume of the candidate region
                curve = ctc_volumes[:, curve_candidates]  # Extract the CTC data for the candidate region
                curve_mean = curve.mean(axis=1)  # Compute the mean CTC curve for the candidate region
                peak_difference = np.max(curve_mean) - np.mean(curve_mean[-3:])  # Difference between peak and end values of the curve

                # Skip candidates that are too large or too small
                if vol > max_vol_thres or vol < min_vol_thres:
                    continue

                # Calculate signal smoothness
                smoothness = calculate_signal_smoothness(curve_mean)
                
                # Skip candidates that are not smooth enough (lower smoothness score means more smooth)
                if smoothness > current_smoothness_threshold:
                    continue

                try:
                    # Fit a gamma variate function to the mean CTC curve of the candidate region
                    aif_properties = fit_gamma_variate(time_index, curve_mean)
                except:
                    # If fitting fails, skip this candidate
                    continue

                # Calculate the fitting error for the candidate region
                fitting_error = np.sqrt(np.sum((curve - gamma_variate(time_index, *aif_properties)[:, np.newaxis]) ** 2, axis=1))

                # Append candidate properties to the list
                cands.append({
                    'idx': idx,
                    'vol': vol,
                    'aif_properties': aif_properties,
                    'mean_error': fitting_error.mean(),
                    'max_error': fitting_error.max(),
                    'peak_difference': peak_difference,
                    'smoothness': smoothness
                })

            # Convert the list of candidate properties to a DataFrame
            cands = pd.DataFrame(cands)
            
            # If we found valid candidates, proceed with selection
            if not cands.empty:
                # Compute a score for each candidate based on volume, peak-end difference, and mean error
                cands['score'] = cands.vol * cands.peak_difference / cands.mean_error

                # Select the candidate with the highest score
                bestCand = np.argmax(cands.score)

                # Extract characteristics of the best candidate
                aifSegIdx = cands.iloc[bestCand].idx  # Index of the best candidate
                aif_properties = cands.iloc[bestCand].aif_properties  # Propperties of the best-fit gamma variate function for the best candidate
                cands.loc[:, 'chosen'] = 0
                cands.loc[bestCand, 'chosen'] = 1  # Mark the selected candidate

                # Make a binary mask of the selected AIF region
                aif_mask = aif_candidate == aifSegIdx

                return aif_properties, aif_mask

    # If no candidates found after all attempts
    raise ValueError("No AIF candidates found after adaptive thresholding attempts., try relaxing the thresholds further by changing the parameters `max_vol_thres`, `min_vol_thres`, or `smoothness_threshold`.")
