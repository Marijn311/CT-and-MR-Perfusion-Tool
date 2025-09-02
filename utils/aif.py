import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation, binary_erosion
import cc3d
import traceback
import pandas as pd



def gv(t, t0, alpha, beta, amplitude=1.0):
    """
    Gamma variate function for modeling arterial input function.
    
    Parameters:
        t: time array
        t0: time delay/onset time
        alpha: shape parameter 
        beta: time constant
        amplitude: peak amplitude scaling factor
    """
    t = np.array(t)
    t_shifted = np.maximum(0, t - t0)
    # Standard gamma variate: A * (t-t0)^alpha * exp(-(t-t0)/beta) for t > t0
    result = np.zeros_like(t_shifted)
    mask = t > t0
    result[mask] = amplitude * (t_shifted[mask]**alpha) * np.exp(-t_shifted[mask]/beta)
    return result

def fit_gv(time_index, curve, sigma=None):
    # Updated bounds to include amplitude parameter
    # bounds: [t0_min, alpha_min, beta_min, amplitude_min], [t0_max, alpha_max, beta_max, amplitude_max]
    return curve_fit(gv, time_index, curve, bounds=([0, 0.1, 0.1, 0], [20, 8, 8, np.max(curve)*2]), sigma=sigma)

def targetF(x_obs, y_obs):
    def gv_target(t0, alpha, beta, amplitude=1.0):
        return (gv(x_obs, t0, alpha, beta, amplitude) - y_obs) ** 2
    return gv_target

def calculate_signal_smoothness(signal):
    """
    Calculate the smoothness of a signal using the sum of squared second derivatives.
    Lower values indicate smoother signals that better resemble the gamma variate function.
    
    Parameters:
        signal (numpy.ndarray): 1D signal array
        
    Returns:
        float: Smoothness metric (lower values = smoother signal)
    """
    if len(signal) < 3:
        return float('inf')  # Cannot calculate second derivative for very short signals
    
    # Calculate second derivative using finite differences
    second_derivative = np.diff(signal, n=2)
    
    # Sum of squared second derivatives as smoothness metric
    smoothness = np.sum(second_derivative ** 2)
    
    # Normalize by signal length to make comparable across different signal lengths
    smoothness = smoothness / len(signal)
    
    return smoothness

def majority(array, ignore=None):
    labels, counts = np.unique(array.ravel(), return_counts=True)
    if ignore is not None:
        mask = labels.isin(ignore)
        counts = counts[~mask]
        labels = labels[~mask]
        return labels[np.argmax(counts)]
    else:
        return labels[np.argmax(counts)]

def determine_aif(ctc_img, time_index, brain_mask, ttp, 
            dilate_radius=5, erode_radius=2,
            max_vol_thres=50, min_vol_thres=5, smoothness_threshold=30.0):
    """
    Identify the Arterial Input Function (AIF) from contrast-enhanced imaging data.

    Parameters:
        ctc_img (numpy.ndarray): Contrast Time Curve (CTC) data as a 4D array.
        time_index (numpy.ndarray): Time indices corresponding to the CTC data.
        brain_mask (SimpleITK.Image): Binary mask indicating the brain region.
        ttp (numpy.ndarray): Time-to-peak values for each voxel.
        dilate_radius (int, optional): Radius for dilation operation during morphological filtering. Defaults to 5.
        erode_radius (int, optional): Radius for erosion operation during morphological filtering. Defaults to 2.
        max_vol_thres (float, optional): Maximum volume threshold for AIF candidates. Defaults to 50.
        min_vol_thres (float, optional): Minimum volume threshold for AIF candidates. Defaults to 5.
        smoothness_threshold (float, optional): Maximum allowed smoothness metric for AIF candidates. 
                                               Candidates with higher values (less smooth) will be rejected. Defaults to 10.0.

    Returns:
        tuple:
            - aif_propperties (numpy.ndarray): Parameters of the best-fit gamma variate function for the AIF.
            - aif_candidate (numpy.ndarray): Binary mask of the selected AIF region.
            - mean_error (float): Mean fitting error between the actual AIF signal and the fitted gamma variate function.
            - smoothness (float): Smoothness metric of the selected AIF signal.

    Raises:
        ValueError: If no AIF candidates are found after all adaptive thresholding attempts.

    Notes:
        The function uses adaptive thresholding with progressively relaxed criteria to identify AIF candidates
        based on AUC (Area Under Curve), TTP (Time-to-Peak) percentiles, and signal smoothness. Morphological 
        operations are applied to refine candidates, and the best candidate is selected based on a score 
        combining volume, peak difference, and fitting error.
    """

    # Erode the brain mask with a flat kernel
    # Due to slight patient motion or sub-optimal masking, skull signal can be inside the mask near the mask's edges. 
    # We want to prevent the skull from being selected as AIF
    kernel = np.ones((1, 15, 15), dtype=bool)  # x,x kernel in x-y plane, no erosion in z, since the number of slices is very low compared to in-plane resolution
    brain_mask = binary_erosion(brain_mask, kernel)

    # Convert the list of 3D volumes to a 4D array
    ctc_img = np.stack(ctc_img, axis=0)

    # Get AUC and TTP distributions
    auc_values = ctc_img.sum(0)[brain_mask > 0]
    ttp_values = ttp[brain_mask > 0]

    # Adaptive thresholding approach
    # Set the TTP and AUC thresholds based on the TTP and AUC distributions
    attempts = [(np.percentile(ttp_values, 100-p), np.percentile(auc_values, p)) 
                for p in range(95, 4, -10)]

    # Try to find an AIF with the thresholds for AUC and TTP. If it fails, try the next set of thresholds. Thresholds are progressively relaxed.
    for attempt_ttp, attempt_auc in attempts:
        # Identify initial AIF candidates based on AUC and TTP thresholds.
        aif_candidate = (ctc_img.sum(0) * brain_mask > attempt_auc) * (ttp < attempt_ttp) * (ttp > 0)

        # Apply morphological operations (dilation followed by erosion) to refine AIF candidates
        aif_candidate = binary_erosion(
            binary_dilation(aif_candidate, np.ones([1, dilate_radius, dilate_radius], bool)), 
            np.ones([1, erode_radius, erode_radius], bool)
        )

        # Perform connected component analysis to separate individual AIF candidates
        aif_candidate, nr_components = cc3d.connected_components(aif_candidate, return_N=True)
        
        # Skip if no components found
        if nr_components == 0:
            continue

        # Initialize a list to store properties of AIF candidates
        cands = []

        # Iterate through each connected component (candidate region)
        for idx in range(1, nr_components + 1):
            curve_candidates = aif_candidate == idx  # Extract the current candidate region
            vol = curve_candidates.sum()  # Calculate the volume of the candidate region
            curve = ctc_img[:, curve_candidates]  # Extract the CTC data for the candidate region
            curve_mean = curve.mean(axis=1)  # Compute the mean CTC curve for the candidate region
            peak_difference = np.max(curve_mean) - np.mean(curve_mean[-3:])  # Difference between peak and end values of the curve

            # Skip candidates with volume exceeding the threshold
            if vol > max_vol_thres or vol < min_vol_thres:
                continue

            # Calculate signal smoothness
            smoothness = calculate_signal_smoothness(curve_mean)
            
            # Skip candidates that are not smooth enough (exceed smoothness threshold)
            if smoothness > smoothness_threshold:
                continue

            try:
                # Fit a gamma variate function to the mean CTC curve
                popts, _ = fit_gv(time_index, curve_mean)
            except:
                # Handle fitting errors
                traceback.print_exc()
                continue

            # Calculate the fitting error for the candidate region
            err = np.sqrt(np.sum((curve - gv(time_index, *popts)[:, np.newaxis]) ** 2, axis=1))

            # Append candidate properties to the list
            cands.append({
                'idx': idx,
                'vol': vol,
                'popts': popts,
                'mean_error': err.mean(),
                'max_error': err.max(),
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
            aifSegIdx = cands.iloc[bestCand].idx  # Index of the selected AIF region
            aif_propperties = cands.iloc[bestCand].popts  # Parameters of the best-fit gamma variate function
            selected_smoothness = cands.iloc[bestCand].smoothness  # Smoothness of the selected candidate
            cands.loc[:, 'chosen'] = 0
            cands.loc[bestCand, 'chosen'] = 1  # Mark the selected candidate

            

            # Return the AIF properties, binary mask of the selected region, mean error, and smoothness
            return aif_propperties, aif_candidate == aifSegIdx, cands.iloc[bestCand].mean_error, selected_smoothness
    
    # If no candidates found after all attempts
    raise ValueError("No AIF candidates found after adaptive thresholding attempts.")

