import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import binary_dilation, binary_erosion
import cc3d
import traceback
import pandas as pd
import SimpleITK as sitk


def gv(t, t0, alpha, beta):
    t = np.array(t) 
    return np.maximum(0, t-t0)**alpha * np.exp(-t/beta) * (np.sign(t - t0) + 1) / 2

def fit_gv(time_index, curve, sigma=None):
    return curve_fit(gv, time_index, curve, bounds=([0, 0.1, 0.1], [20, 8, 8]), sigma=sigma)

def targetF(x_obs, y_obs):
    def gv_target(t0, alpha, beta):
        return (gv(x_obs, t0, alpha, beta) - y_obs) ** 2
    return gv_target

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
            cand_vol_thres=2000, roi=None):
    """
    Identify the Arterial Input Function (AIF) from contrast-enhanced imaging data.

    Parameters:
        ctc_img (numpy.ndarray): Contrast Time Curve (CTC) data as a 4D array.
        time_index (numpy.ndarray): Time indices corresponding to the CTC data.
        brain_mask (SimpleITK.Image): Binary mask indicating the brain region.
        ttp (numpy.ndarray): Time-to-peak values for each voxel.
        dilate_radius (int, optional): Radius for dilation operation during morphological filtering. Defaults to 5.
        erode_radius (int, optional): Radius for erosion operation during morphological filtering. Defaults to 2.
        cand_vol_thres (float, optional): Maximum volume threshold for AIF candidates in mm³. Defaults to 2000.
        roi (tuple, optional): Region of interest (ROI) specified as (x_min, x_max, y_min, y_max, z_min, z_max).

    Returns:
        tuple:
            - aif_propperties (numpy.ndarray): Parameters of the best-fit gamma variate function for the AIF.
            - aif_candidate (numpy.ndarray): Binary mask of the selected AIF region.

    Raises:
        ValueError: If no AIF candidates are found after all adaptive thresholding attempts.

    Notes:
        The function uses adaptive thresholding with progressively relaxed criteria to identify AIF candidates
        based on AUC (Area Under Curve) and TTP (Time-to-Peak) percentiles. Morphological operations are
        applied to refine candidates, and the best candidate is selected based on a score combining volume,
        peak difference, and fitting error.
    """

    brain_mask_sitk = brain_mask
    
    # Convert the brain mask to a NumPy array
    brain_mask = sitk.GetArrayFromImage(brain_mask)

    # Erode the brain mask with a flat kernel
    # Due to the patient motion, skull can fall inside the mask at the edges. We want to prevent the skull from being seleted as AIF
    kernel = np.ones((1, 8, 8), dtype=bool)  # x,x kernel in x-y plane, no erosion in z, since the number of slices is very low compared to in-plane resolution
    brain_mask = binary_erosion(brain_mask, kernel)
    
    # Get AUC and TTP distributions
    auc_values = ctc_img.sum(0)[brain_mask > 0]
    ttp_values = ttp[brain_mask > 0]

    # Adaptive thresholding approach
    # Set the TTP and AUC thresholds based on the TTP and AUC distributions
    attempts = [
        (np.percentile(ttp_values, 25), np.percentile(auc_values, 75)),
        (np.percentile(ttp_values, 30), np.percentile(auc_values, 70)),
        (np.percentile(ttp_values, 35), np.percentile(auc_values, 65)),
        (np.percentile(ttp_values, 40), np.percentile(auc_values, 60)),
        (np.percentile(ttp_values, 45), np.percentile(auc_values, 55)),
        (np.percentile(ttp_values, 50), np.percentile(auc_values, 50)),
        (np.percentile(ttp_values, 55), np.percentile(auc_values, 45)),
        (np.percentile(ttp_values, 60), np.percentile(auc_values, 40)),
        (np.percentile(ttp_values, 65), np.percentile(auc_values, 35)),
        (np.percentile(ttp_values, 70), np.percentile(auc_values, 30)),
        (np.percentile(ttp_values, 75), np.percentile(auc_values, 25)),
    ]

    # Try to find an AIF with the thresholds for AUC and TTP. If it fails, try the next set of thresholds. Thresholds are progressively relaxed.
    for attempt_ttp, attempt_auc in attempts:
        # Identify initial AIF candidates based on AUC and TTP thresholds.
        aif_candidate = (ctc_img.sum(0) * brain_mask > attempt_auc) * (ttp < attempt_ttp) * (ttp > 0)

        # If a region of interest (ROI) is provided, restrict AIF candidates to the ROI
        if roi is not None:
            aif_candidate_roi = np.zeros_like(aif_candidate)
            aif_candidate_roi[roi[4]:roi[5], roi[0]:roi[1], roi[2]:roi[3]] = aif_candidate[roi[4]:roi[5], roi[0]:roi[1], roi[2]:roi[3]]
            aif_candidate = aif_candidate_roi

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
            if vol * np.prod(brain_mask_sitk.GetSpacing()) > cand_vol_thres:
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
                'vol': vol * np.prod(brain_mask_sitk.GetSpacing()),
                'popts': popts,
                'mean_error': err.mean(),
                'max_error': err.max(),
                'peak_difference': peak_difference
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
            cands.loc[:, 'chosen'] = 0
            cands.loc[bestCand, 'chosen'] = 1  # Mark the selected candidate

            # Return the AIF properties, binary mask of the selected region
            return aif_propperties, aif_candidate == aifSegIdx
    
    # If no candidates found after all attempts
    raise ValueError("No AIF candidates found after adaptive thresholding attempts.")

