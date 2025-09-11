import numpy as np
from config import *
from scipy.linalg import toeplitz
from utils.determine_aif import gamma_variate
from utils.boxNLR import boxNLR


def extract_ctc(volume_list, mask, bolus_threshold=0.01):
    """
    Extracts the Contrast Time Curve (CTC) from a 4D perfusion image.
    This function identifies the starting index of significant contrast agent presence, 
    creates a baseline from pre-contrast volumes, and subtracts the baseline to isolate the contrast enhancement signal.

    Parameters:
        - volume_list (list): A list of 3D nd.array (z,y,x) images representing a time series of perfusion data.
        - mask (nd.array): 3D (z,y,x) binary brain mask.
        - bolus_threshold (float, optional): The threshold for the ratio of contrast agent signal change 
                                        to determine the starting index. Defaults to 0.01.

    Returns:
        - volume_list (list): A list of 3D nd.array (z,y,x) images representing the contrast agent signal over time.
        - s0_index (int): The starting index of significant contrast agent presence.
        - norm_list_of_means (list): List of mean intensity values inside the brain mask for each timepoint, normalized by the first timepoint.
    """


    if IMAGE_TYPE == 'mrp':
        # If the image type is MRP, we to invert the perfusion image.
        # This way the contrast agent appears as high values (just like in CTP), this make processing easier.
        global_max = max(np.max(vol) for vol in volume_list)
        global_min = min(np.min(vol) for vol in volume_list)
        volume_list = [global_max + global_min - i for i in volume_list]

    # For each timepoint, store the mean signal in the brain.
    list_of_means = []
    
    # Save the mean intensity value of all pixels in side the brain mask for every timepoint
    nr_timepoints = len(volume_list)
    for t in range(nr_timepoints):
        current_volume = volume_list[t]
        current_volume_mean = current_volume[mask.astype('bool')].mean()
        list_of_means.append(current_volume_mean)
        
    # Normalize the mean values by the first volume.
    # We assume the first timepoint contains no contrast agent.
    # hence by normalizing we get a relative measure of contrast agent presence.
    norm_list_of_means = (list_of_means - list_of_means[0]) / list_of_means[0]

    # Find the index where the contrast agent starts to appear (aka the bolus start, aka s0_index).
    s0_index = None
    differences = []
    for t in range(nr_timepoints - 1):
        # For each timepoint we want to determine the average of the normalized mean values over the last 3 timepoints.
        # Taking the average over the last 3 timepoints makes the measure more stable.
        # If the difference between the average at timepoint t and t+1 is larger than the bolus_threshold,
        # then we assume the contrast agent starts to appear at timepoint t.

        # We skip the 2 initial timepoints, since we don't have enough data there to calculate a stable average.
        if t < 2:
            differences.append(0)  # Store a placeholder 0 for skipped timepoints
            continue

        # Calculate average over last 3 datapoints at time t and t+1
        start_window = t - 2
        start_window_plus_1 = t - 1  
        end_window = t
        end_window_plus_1 = t + 1
        average_window = np.mean(norm_list_of_means[start_window:end_window + 1])
        average_window_plus_1 = np.mean(norm_list_of_means[start_window_plus_1:end_window_plus_1 + 1])
        difference = average_window_plus_1 - average_window
        differences.append(difference)

        if difference > bolus_threshold: # If the relative change is larger than the threshold we have found the bolus start
            s0_index = t
            break

    if s0_index is None:
        # If no timepoint crosses the threshold, take the timepoint with the biggest positive difference
        max_diff_index = np.argmax(differences)
        s0_index = max_diff_index
        print(f"Warning: No bolus start found with the given threshold. Using timepoint {s0_index} with max difference instead.")
        print(f"Consider lowering the bolus_threshold parameter if this is not desired.")

    s0_index = s0_index - 1 # We take the timepoint just before we see the big jump in contrast agent

    # Average the volumes before the contrast agent starts to appear, to make a more stable baseline (S0)
    volumes = np.stack(volume_list, axis=0)
    s0 = volumes[:s0_index,:,:,:].mean(axis=0)

    # Determine the contrast agent signal
    if IMAGE_TYPE == 'ctp':
        # For CTP the relationship between the iodine contrast agent and image intensity is approximately linear.
        # Because in CTP the iodine contrast directly attentuates the xrays, causing the image signal.
        # Remove the baseline signal S0 from the images
        # This leaves us with the contrast agent signal only
        volumes = volumes - s0[np.newaxis, :, :, :]
    if IMAGE_TYPE == 'mrp':
        # For MRP the gadolinium contrast doesn’t show up as a “signal” itself. 
        # Instead, it changes relaxation times (mainly T2* and T1, depending on the sequence).
        # The MR signal intensity is not linearly related to contrast concentration. 
        # Instead, it depends on an exponential relationship between signal and relaxation times.
        epsilon = 1e-8  # Small value to prevent division by zero
        s0 = np.where(s0 == 0, epsilon, s0)
        ratio = volumes / (s0[np.newaxis, :, :, :])
        ratio = np.where(ratio <= 0, epsilon, ratio)  # Prevent log of zero or negative values
        volumes = (1 / ECHO_TIME) * np.log(ratio)
    
    # Apply the brain mask
    volumes = volumes * mask[np.newaxis, :, :, :]

    # Transform the 4D array back to a list of 3D arrays
    volume_list = [volumes[i] for i in range(volumes.shape[0])]

    return volume_list, s0_index, norm_list_of_means


def generate_ttp(ctc_volumes, time_index, s0_index, mask, outside_value=-1):
    """
    Generate the Time to Peak (TTP) map from contrast time curves (ctc).
    TTP is simply the time at which the CTC reaches its maximum value.

    Parameters:
        - ctc_volumes (list): List of 3D nd.array volumes (z,y,x) representing contrast time curves.
        - time_index (list): list of time indexes corresponding to each timepoint in ctc_volumes.
        - s0_index (int): Starting index for analysis. Time points before this index are excluded from TTP calculation (e.g., baseline measurements).
        - mask (nd.array): Binary 3D nd.array (z,y,x) of the brain mask.
        - outside_value (int or float, optional): Value assigned to voxels outside the brain mask. Default is -1.
    
    Returns
        - TTP (nd.array): 3D nd.array (z,y,x) of time to peak map with the same spatial dimensions as the input 
        contrast data. Values represent the time (in same units as time_index).
    """
    # Convert the list of 3D arrays to one 4D array
    ctc_volumes = np.stack(ctc_volumes, axis=0)
    time_index = np.array(time_index)

    # Filter time indices starting from s0_index and adjust relative to start time
    time_index = time_index[s0_index:] - time_index[s0_index]

    # Filter contrast data to exclude baseline measurements
    ctc_volumes = ctc_volumes[s0_index:, :, :, :]

    # Find time to peak by getting time index of maximum enhancement for each voxel
    ttp = time_index[ctc_volumes.argmax(axis=0)]

    # Set voxels outside brain mask to specified outside value
    ttp[mask == 0] = outside_value
    
    return ttp



def generate_perfusion_maps(ctc_volumes, time_index, mask, aif_properties, method='SVD', SVD_truncation_threshold=0.1, oSVD_OI_threshold=0.035, outside_value=-1, rho=1.05, hcf=0.73):
    """
    Generate perfusion maps (MTT, CBV, CBF, Tmax) from contrast time curves (CTC) via deconvolution with the arterial input function (AIF).
    This function supports multiple SVD-based deconvolution methods.
    Additionally, there is the option to use the box-shaped model with non-linear regression optimization,
    as described in Bennink et al. (2004) (https://doi.org/10.1117/1.JMI.3.2.026003). Though this method is still a work in progress.

    Parameters: 
        - ctc_volumes (list): List of 3D nd.arrays (z,y,x) representing contrast time curves.
        - time_index (list): list of time indexes corresponding to the ctc_volumes.
        - mask (nd.array): 3D binary brain mask (z,y,x).
        - aif_properties (nd.array): 1D array with 4 elements representing the parameters of the fitted gamma variate function for the AIF.
        - method (str, optional): Which deconvolution method to use, either 'bcSVD1', 'bcSVD2', 'SVD', 'oSVD', or 'boxNLR' (default: 'SVD')
        - SVD_truncation_threshold (float, optional): threshold for svd regularization as fraction of maximum singular value (default: 0.1)
        - oSVD_OI_threshold (float, optional): threshold for oscillation index in oSVD method (default: 0.035)
        - outside_value (float, optional): Value assigned to voxels outside the brain mask (default: -1)
        - rho (float, optional): Tissue density in g/ml (default: 1.05 g/ml)
        - hcf (float, optional): Hematocrit correction factor (default: 0.73)
    Returns:
        - mtt (nd.array): 3D array (z,y,x) containing the Mean Transit Time in seconds
        - cbv (nd.array): 3D array (z,y,x) containing the Cerebral Blood Volume in ml/100g 
        - cbf (nd.array): 3D array (z,y,x) containing the Cerebral Blood Flow in ml/100g/min
        - tmax (nd.array): 3D array (z,y,x) containing the Time to maximum of residue function in seconds
    """
        
    if method not in ['SVD', 'bcSVD1', 'bcSVD2', 'oSVD', 'boxNLR']:
        raise ValueError("Invalid method. Choose either 'SVD', 'bcSVD1', 'bcSVD2', 'oSVD', or 'boxNLR'.")

    # Generate AIF with the gamma variate function and the provided parameters
    aif = gamma_variate(time_index, *aif_properties)
    
    # Calculate time step for numerical integration
    deltaT = np.mean(np.diff(time_index))

    # Get number of time points
    nr_timepoints = len(time_index)  

    # Construct convolution matrix G based on selected deconvolution method
    if method == 'SVD':
        # Original SVD method using standard Toeplitz matrix
        G = toeplitz(aif, np.zeros(nr_timepoints))
        ctc_volumes_pad = ctc_volumes

    if method == 'bcSVD1':
        # Block-circulant SVD method 1 with Simpson's rule integration
        colG = np.zeros(2 * nr_timepoints)
        colG[0] = aif[0]
        
        # Apply Simpson's rule for numerical integration at boundaries
        colG[nr_timepoints - 1] = (aif[nr_timepoints - 2] + 4 * aif[nr_timepoints - 1]) / 6
        colG[nr_timepoints] = aif[nr_timepoints - 1] / 6
        
        # Apply Simpson's rule for interior points
        for k in range(1, nr_timepoints - 1):
            colG[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6

        # Construct row vector for circulant matrix
        rowG = np.zeros(2 * nr_timepoints)
        rowG[0] = colG[0]
        for k in range(1, 2 * nr_timepoints):
            rowG[k] = colG[2 * nr_timepoints - k]

        # Create block-circulant matrix
        G = toeplitz(colG, rowG)
   
        # Pad contrast data by doubling temporal dimension
        ctc_volumes_pad = np.pad(ctc_volumes, [(0, len(ctc_volumes)),] + [(0, 0)] * 3)
   
    if method == 'bcSVD2':
        # Block-circulant SVD method 2 with manual matrix construction
        cmat = np.zeros([nr_timepoints, nr_timepoints])
        B = np.zeros([nr_timepoints, nr_timepoints])
        
        # Build circulant and anti-circulant blocks
        for i in range(nr_timepoints):
            for j in range(nr_timepoints):
                if i == j:
                    cmat[i, j] = aif[0]
                elif i > j:
                    cmat[i, j] = aif[(i - j)]
                else:
                    B[i, j] = aif[nr_timepoints - (j - i)]
        
        # Construct 2x2 block matrix
        G = np.vstack([np.hstack([cmat, B]), np.hstack([B, cmat])])
        
        # Pad contrast data by doubling temporal dimension
        ctc_volumes_pad = np.pad(ctc_volumes, [(0, len(ctc_volumes)),] + [(0, 0)] * 3)
    
    if method == 'oSVD':
        # Oscillation index SVD method with adaptive regularization
        # Uses block-circulant matrix construction similar to bcSVD1
        colG = np.zeros(2 * nr_timepoints)
        colG[0] = aif[0]
        
        # Apply Simpson's rule for numerical integration at boundaries
        colG[nr_timepoints - 1] = (aif[nr_timepoints - 2] + 4 * aif[nr_timepoints - 1]) / 6
        colG[nr_timepoints] = aif[nr_timepoints - 1] / 6
        
        # Apply Simpson's rule for interior points
        for k in range(1, nr_timepoints - 1):
            colG[k] = (aif[k - 1] + 4 * aif[k] + aif[k + 1]) / 6

        # Construct row vector for circulant matrix
        rowG = np.zeros(2 * nr_timepoints)
        rowG[0] = colG[0]
        for k in range(1, 2 * nr_timepoints):
            rowG[k] = colG[2 * nr_timepoints - k]

        # Create block-circulant matrix
        G = toeplitz(colG, rowG)
   
        # Pad contrast data by doubling temporal dimension
        ctc_volumes_pad = np.pad(ctc_volumes, [(0, len(ctc_volumes)),] + [(0, 0)] * 3)
        
        # Use oSVD-specific processing instead of standard SVD approach
        mtt, cbv, cbf, tmax = oSVD(ctc_volumes_pad, G, deltaT, mask, oSVD_OI_threshold, nr_timepoints, outside_value, rho, hcf, time_index)
        return mtt, cbv, cbf, tmax

    if method == 'boxNLR':
        mtt, cbv, cbf, tmax = boxNLR(ctc_volumes, aif, deltaT, mask, outside_value=-1)
        return mtt, cbv, cbf, tmax


    # Perform SVD decomposition on scaled convolution matrix
    U, S, V = np.linalg.svd(G * deltaT)
    
    # Apply threshold-based regularization to singular values
    thres = SVD_truncation_threshold * np.max(S)
    filteredS = 1 / (S + 1e-5)  # Add small epsilon to avoid division by zero
    filteredS[S < thres] = 0  # Zero out small singular values below threshold
    
    # Reconstruct pseudo-inverse matrix using filtered singular values
    Ginv = V.T @ np.diag(filteredS) @ U.T
    
    # Perform deconvolution to obtain residue function R
    R = np.abs(np.einsum('ab, bcde->acde', Ginv, ctc_volumes_pad))
    
    # Truncate residue function to original temporal length
    R = R[:nr_timepoints] 
    
    # Calculate perfusion parameters from residue function
    # CBF: Maximum of residue function scaled by physiological constants (ml/100g/min)
    cbf = hcf / rho * R.max(axis=0) * 60 * 100
    # CBV: Area under residue function scaled by physiological constants (ml/100g)
    cbv = hcf / rho * R.sum(axis=0) * 100
    # MTT: Mean transit time calculated as CBV/CBF ratio (seconds)
    mtt = np.divide(cbv, cbf, out=np.zeros_like(cbv), where=cbf!=0) * 60

    # Calculate time to maximum of residue function (tmax)
    time_index = np.array(time_index, dtype=int) 
    tmax = time_index[R.argmax(axis=0)]
    tmax = tmax.astype(np.float64)
    
    # Apply brain mask to set outside values for all perfusion maps
    tmax[mask == 0] = outside_value
    cbv[mask == 0] = outside_value
    cbf[mask == 0] = outside_value
    mtt[mask == 0] = outside_value

    return mtt, cbv, cbf, tmax


def oSVD(ctc_volumes_pad, G, deltaT, mask, oSVD_OI_threshold, nr_timepoints, outside_value, rho, hcf, time_index):
    """
    Helper function to generate perfusion maps using the oSVD method.
    
    This function implements the oscillation index SVD method which adaptively selects
    the optimal SVD truncation threshold for each voxel based on minimizing oscillations
    in the residue function.
    
    Parameters:
        - ctc_volumes_pad (np.array): 4D array of padded contrast time curves
        - G (np.array): Block-circulant convolution matrix 
        - deltaT (float): Time step for numerical integration
        - mask (np.array): 3D binary brain mask
        - oSVD_OI_threshold (float): Oscillation index threshold
        - nr_timepoints (int): Number of original timepoints
        - outside_value (float): Value for voxels outside mask
        - rho (float): Tissue density
        - hcf (float): Hematocrit correction factor
        - time_index (list): Time indices
        
    Returns:
        - mtt, cbv, cbf, tmax: Perfusion parameter maps
    """
    
    # Perform SVD decomposition on scaled convolution matrix
    U, S, V = np.linalg.svd(G * deltaT)
    
    # Initialize output arrays
    z, y, x = mask.shape
    cbf = np.zeros((z, y, x))
    cbv = np.zeros((z, y, x))
    mtt = np.zeros((z, y, x))
    tmax = np.zeros((z, y, x))
    
    # Process each voxel individually for adaptive threshold selection
    pixel_count = 0
    total_pixels = np.sum(mask)
    
    for zi in range(z):
        for yi in range(y):
            for xi in range(x):
                if mask[zi, yi, xi]:
                    pixel_count += 1
                    
                    # Print progress every 1000 pixels
                    if pixel_count % 1000 == 0:
                        percentage = (pixel_count / total_pixels) * 100
                        print(f"Processing pixel {pixel_count}/{total_pixels} ({percentage:.1f}%)")
                    
                    # Extract concentration time curve for current voxel
                    vett_conc = ctc_volumes_pad[:, zi, yi, xi]
                    
                    # Find optimal threshold using oscillation index
                    best_residue = None
                    
                    # Test thresholds from 5% to 95% of maximum singular value
                    for threshold_percent in range(5, 100, 5):
                        threshold = (threshold_percent / 100.0) * S[0]
                        
                        # Create filtered inverse matrix
                        filtered_S = 1.0 / (S + 1e-5)
                        filtered_S[S < threshold] = 0
                        G_inv = V.T @ np.diag(filtered_S) @ U.T
                        
                        # Calculate residue function
                        residue = G_inv @ vett_conc
                        residue = np.abs(residue)
                        
                        # Calculate oscillation index
                        oscillation = 0.0
                        L = len(residue)
                        for j in range(2, L):
                            oscillation += abs(residue[j] - 2*residue[j-1] + residue[j-2])
                        
                        max_residue = np.max(residue)
                        if max_residue > 0:
                            OI = (1.0 / L) * (1.0 / max_residue) * oscillation
                        else:
                            OI = float('inf')
                        
                        # Use this threshold if oscillation index is below threshold
                        if OI < oSVD_OI_threshold:
                            best_residue = residue
                            break
                    
                    # If no threshold met criteria, use the most restrictive one
                    if best_residue is None:
                        threshold = 0.95 * S[0]
                        filtered_S = 1.0 / (S + 1e-5)
                        filtered_S[S < threshold] = 0
                        G_inv = V.T @ np.diag(filtered_S) @ U.T
                        best_residue = np.abs(G_inv @ vett_conc)
                    
                    # Truncate residue function to original temporal length
                    residue_truncated = best_residue[:nr_timepoints]
                    
                    # Calculate perfusion parameters for this voxel
                    cbf[zi, yi, xi] = hcf / rho * np.max(residue_truncated) * 60 * 100
                    cbv[zi, yi, xi] = hcf / rho * np.sum(residue_truncated) * 100
                    if cbf[zi, yi, xi] != 0:
                        mtt[zi, yi, xi] = (cbv[zi, yi, xi] / cbf[zi, yi, xi]) * 60
                    
                    # Calculate time to maximum
                    time_index_array = np.array(time_index[:nr_timepoints], dtype=int)
                    tmax[zi, yi, xi] = float(time_index_array[np.argmax(residue_truncated)])
    
    # Apply brain mask to set outside values
    tmax[mask == 0] = outside_value
    cbv[mask == 0] = outside_value
    cbf[mask == 0] = outside_value
    mtt[mask == 0] = outside_value
    
    return mtt, cbv, cbf, tmax