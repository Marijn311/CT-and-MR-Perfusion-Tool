import numpy as np
from utils.determine_aif import gv
from config import *
from scipy.linalg import toeplitz


def extract_ctc(volume_list, brain_mask):
    """
    Extracts the Contrast Time Curve (CTC) from a sequence of CTP/MRP images.

    This function processes a list of raw CTP images and a brain mask to compute the 
    contrast agent signal over time. It identifies the starting index of significant 
    contrast agent presence, creates a baseline from pre-contrast volumes, and 
    subtracts the baseline to isolate the contrast enhancement signal.

    Parameters:
        volume_list (list of SimpleITK.Image): A list of raw CTP images representing a time series.
        brain_mask (SimpleITK.Image): A binary mask indicating the brain region in the images.
        bolus_threshold (float, optional): The threshold for the ratio of contrast agent signal change 
                                        to determine the starting index. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - img_np (numpy.ndarray): The baseline-corrected contrast enhancement signal 
                                        (raw signal minus averaged pre-contrast baseline).
            - s0_index (int): The starting index of significant contrast agent presence.
    """

    if IMAGE_TYPE == 'ctp':
        bolus_threshold=0.01

    if IMAGE_TYPE == 'mrp': 
        bolus_threshold=0.01

        # For the visualization we want the contrast to be high values and the background to be low
        # Invert the image using global min and max across all volumes
        global_max = max(np.max(vol) for vol in volume_list)
        global_min = min(np.min(vol) for vol in volume_list)
        volume_list = [global_max + global_min - i for i in volume_list]

    # For each timepoint, store the mean signal in the brain.
    total_mean = []
    
    # Iterate through each time point in the image series
    nr_timepoints = len(volume_list)
    for t in range(nr_timepoints):
        
        # Get the current volume at time point t
        current_volume = volume_list[t]

        # Take the mean of the pixels in side the brain mask 
        current_volume_mean = current_volume[brain_mask.astype('bool')].mean()

        # Add this volume's mean to the list
        total_mean.append(current_volume_mean)
    

    # Normalize the total contrast agent value by the first volume, to remove the baseline
    total_mean = (total_mean - total_mean[0]) / total_mean[0]
    
    # Extract the volume and index where the contrast agent starts to appear,
    # S0 is the baseline volume just before the actual contrast agent inflow   
    # If the difference between the mean up to the current time point and the mean up to the next time point is smaller than the threshold, 
    # we go on to the next time point.
    s0_index = None
    differences = []
    
    for t in range(nr_timepoints - 1):
        if t < 2:  # we skip the first few timepoints, this gives us a more stable signal. We don't expect the bolus to start within the first 3 timepoints.
            differences.append(0)  # Store 0 for skipped timepoints
            continue
        # Calculate average over last 3 datapoints at time t and t+1
        start_t = max(0, t - 2)  # Ensure we don't go below index 0
        start_t_plus_1 = max(0, t - 1)  # Ensure we don't go below index 0
        average_at_t = np.mean(total_mean[start_t:t + 1])
        average_at_t_plus_1 = np.mean(total_mean[start_t_plus_1:t + 2])
        difference = average_at_t_plus_1 - average_at_t
        differences.append(difference)
        if difference > bolus_threshold: # if the relative change is smaller than the threshold we skip
            s0_index = t
            break
    if s0_index is None:
        # If no step crosses the threshold, take the timepoint with the biggest positive difference
        max_diff_index = np.argmax(differences)
        s0_index = max_diff_index
    s0_index = s0_index - 1 # We take the timepoint just before we see the big jump in contrast agent


    # Average the volumes before the contrast agent starts to appear, to make a more stable baseline (S0)
    volumes = np.stack(volume_list, axis=0)
    s0 = volumes[:s0_index,:,:,:].mean(axis=0)

    if IMAGE_TYPE == 'ctp':
        # For CTP the relationship between contrast agent and image intensity is approximately linear.
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
    volumes = volumes * brain_mask[np.newaxis, :, :, :]

    # Unstack back into a list of 3d arrays
    volume_list = [volumes[i] for i in range(volumes.shape[0])]

    return volume_list, s0_index, total_mean

def generate_ttp(ctc_img, time_index, s0_index, brain_mask, outside_value=-1):
    """
    Calculate Time to Peak (TTP) from contrast time curve data.
    
    This function computes the time to peak enhancement for each voxel in a 
    perfusion imaging dataset by finding the time point at which maximum 
    contrast enhancement occurs.
    
    Parameters
    ----------
    ctc_img : array_like
        Contrast time curve data with shape (time_points, height, width) or 
        (time_points, height, width, depth). Each time point represents a 
        different contrast phase.
    time_index : array_like
        Time indices corresponding to each time point in the contrast curve.
        Should have the same length as the first dimension of ctc_img.
    s0_index : int
        Starting index for analysis. Time points before this index are 
        excluded from TTP calculation (e.g., baseline measurements).
    brain_mask : SimpleITK.Image
        Binary brain mask image used to exclude non-brain regions from 
        analysis. Voxels with value 0 are considered outside the brain.
    outside_value : int or float, optional
        Value assigned to voxels outside the brain mask. Default is -1.
    
    Returns
    -------
    numpy.ndarray
        Time to peak map with the same spatial dimensions as the input 
        contrast data. Values represent the time (in same units as time_index) 
        at which peak enhancement occurs for each voxel. Voxels outside 
        the brain mask are set to outside_value.
    """
    # Convert the list of 3d volumes to one 4d array
    ctc_img = np.stack(ctc_img, axis=0)
    time_index = np.array(time_index)

    # Filter time indices starting from s0_index and adjust relative to start time
    time_index = time_index[s0_index:] - time_index[s0_index]

    # Filter contrast data to exclude baseline measurements
    ctc_img = ctc_img[s0_index:, :, :, :]

    # Find time to peak by getting time index of maximum enhancement for each voxel
    ttp = time_index[ctc_img.argmax(axis=0)]

    # Set voxels outside brain mask to specified outside value
    ttp[brain_mask == 0] = outside_value
    
    return ttp


def generate_perfusion_maps(ctc_img, time_index, brain_mask, aif_propperties, outside_value=-1):
    """
    Generate perfusion maps (MTT, CBV, CBF, tmax) from CT perfusion data using deconvolution methods.
    This function performs deconvolution of tissue concentration curves with an arterial input function (AIF)
    to calculate cerebral perfusion parameters. It supports multiple SVD-based deconvolution methods
    including original SVD (oSVD) and block-circulant SVD variants (bcSVD1, bcSVD2).
    The deconvolution process:
    1. Constructs a convolution matrix G from the AIF based on the selected method
    2. Applies singular value decomposition (SVD) with thresholding for regularization
    3. Computes the residue function k through matrix inversion
    4. Derives perfusion parameters from the residue function
    Parameters
    ----------
    ctc_img : numpy.ndarray
        4D array of tissue concentration curves with shape (time, z, y, x)
    time_index : list or numpy.ndarray
        Time points corresponding to the temporal dimension of ctc_img
    brain_mask : Binary mask defining brain tissue regions (1 = brain, 0 = background)
    aif_propperties : list or numpy.ndarray
        Arterial input function values or fitting parameters for AIF estimation
    CSVD_THRES : float, optional
        SVD threshold for regularization as fraction of maximum singular value (default: 0.1)
    METHOD : str, optional
        Deconvolution method - 'oSVD', 'bcSVD1', or 'bcSVD2' (default: 'bcSVD1')
        - 'oSVD': Original SVD using Toeplitz matrix
        - 'bcSVD1': Block-circulant SVD method 1 with padding
        - 'bcSVD2': Block-circulant SVD method 2 with custom matrix construction
    outside_value : float, optional
        Value assigned to voxels outside the brain mask (default: -1)
    Returns
    -------
    tuple of numpy.ndarray
        MTT : Mean Transit Time in seconds (3D array)
        CBV : Cerebral Blood Volume in ml/100g (3D array)  
        CBF : Cerebral Blood Flow in ml/100g/min (3D array)
        tmax : Time to maximum of residue function in seconds (3D array)
    Notes
    -----
    - Uses standard perfusion constants: density rho=1.05 g/ml, hematocrit H=0.85
    - Applies SVD regularization to handle ill-conditioned deconvolution
    - Block-circulant methods (bcSVD1, bcSVD2) pad the data to reduce boundary artifacts
    - All output maps are masked using the provided brain_mask
    """
        
    # Define tissue constants: density (g/ml) and hematocrit correction factor
    rho, H = 1.05, 0.73
    
    # Generate AIF values if parameters are provided, otherwise use direct values
    if len(aif_propperties) != len(time_index):
        # Estimate AIF using gamma variate function with provided parameters
        estimated_aif_value = gv(time_index, *aif_propperties)
        aif_value = estimated_aif_value
    else:
        # Use provided AIF values directly
        aif_value = aif_propperties
    
    # Calculate time step for numerical integration
    deltaT = np.mean(np.diff(time_index))

    # Get number of time points
    nr_timepoints = len(time_index)  

    # Construct convolution matrix G based on selected deconvolution method
    if METHOD == 'oSVD':
        # Original SVD method using standard Toeplitz matrix
        G = toeplitz(aif_value, np.zeros(nr_timepoints))
        ctc_img_pad = ctc_img
    elif METHOD == 'bcSVD1':
        # Block-circulant SVD method 1 with Simpson's rule integration
        colG = np.zeros(2 * nr_timepoints)
        colG[0] = aif_value[0]
        # Apply Simpson's rule for numerical integration at boundaries
        colG[nr_timepoints - 1] = (aif_value[nr_timepoints - 2] + 4 * aif_value[nr_timepoints - 1]) / 6
        colG[nr_timepoints] = aif_value[nr_timepoints - 1] / 6
        # Apply Simpson's rule for interior points
        for k in range(1, nr_timepoints - 1):
            colG[k] = (aif_value[k - 1] + 4 * aif_value[k] + aif_value[k + 1]) / 6

        # Construct row vector for circulant matrix
        rowG = np.zeros(2 * nr_timepoints)
        rowG[0] = colG[0]
        for k in range(1, 2 * nr_timepoints):
            rowG[k] = colG[2 * nr_timepoints - k]

        # Create block-circulant matrix
        G = toeplitz(colG, rowG)
        # Pad contrast data by doubling temporal dimension
        ctc_img_pad = np.pad(ctc_img, [(0, len(ctc_img)),] + [(0, 0)] * 3)
    elif METHOD == 'bcSVD2':
        # Block-circulant SVD method 2 with manual matrix construction
        cmat = np.zeros([nr_timepoints, nr_timepoints])
        B = np.zeros([nr_timepoints, nr_timepoints])
        # Build circulant and anti-circulant blocks
        for i in range(nr_timepoints):
            for j in range(nr_timepoints):
                if i == j:
                    cmat[i, j] = aif_value[0]
                elif i > j:
                    cmat[i, j] = aif_value[(i - j)]
                else:
                    B[i, j] = aif_value[nr_timepoints - (j - i)]
        # Construct 2x2 block matrix
        G = np.vstack([np.hstack([cmat, B]), np.hstack([B, cmat])])
        # Pad contrast data by doubling temporal dimension
        ctc_img_pad = np.pad(ctc_img, [(0, len(ctc_img)),] + [(0, 0)] * 3)
    else:
        raise NotImplementedError(f"method {METHOD} is not supported.")

    # Perform SVD decomposition on scaled convolution matrix
    U, S, V = np.linalg.svd(G * deltaT)
    
    # Apply threshold-based regularization to singular values
    thres = CSVD_THRES * np.max(S)
    filteredS = 1 / (S + 1e-5)  # Add small epsilon to avoid division by zero
    filteredS[S < thres] = 0  # Zero out small singular values below threshold
    
    # Reconstruct pseudo-inverse matrix using filtered singular values
    Ginv = V.T @ np.diag(filteredS) @ U.T
    
    # Perform deconvolution to obtain residue function k
    k = np.abs(np.einsum('ab, bcde->acde', Ginv, ctc_img_pad))
    
    # Truncate residue function to original temporal length
    k = k[:nr_timepoints] 
    
    # Calculate perfusion parameters from residue function
    # CBF: Maximum of residue function scaled by physiological constants (ml/100g/min)
    cbf = H / rho * k.max(axis=0) * 60 * 100
    # CBV: Area under residue function scaled by physiological constants (ml/100g)
    cbv = H / rho * k.sum(axis=0) * 100
    # MTT: Mean transit time calculated as CBV/CBF ratio (seconds)
    mtt = np.divide(cbv, cbf, out=np.zeros_like(cbv), where=cbf!=0) * 60

    # Calculate time to maximum of residue function (tmax)
    time_index = np.array(time_index, dtype=int) 
    tmax = time_index[k.argmax(axis=0)]
    tmax = tmax.astype(np.float64)
    
    # Apply brain mask to set outside values for all perfusion maps
    tmax[brain_mask == 0] = outside_value
    cbv[brain_mask == 0] = outside_value
    cbf[brain_mask == 0] = outside_value
    mtt[brain_mask == 0] = outside_value

    return mtt, cbv, cbf, tmax






    