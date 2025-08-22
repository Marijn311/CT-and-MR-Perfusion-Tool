from scipy.linalg import toeplitz
import SimpleITK as sitk
import numpy as np
from utils.aif import gv

def extract_ctc(img_list, brain_mask, hu_threshold=150, ratio_threshold=0.05):
    """
    Extracts the Contrast Time Curve (CTC) from a sequence of CTP images.

    This function processes a list of raw CTP images and a brain mask to compute the 
    contrast agent signal over time. It identifies the starting index of significant 
    contrast agent presence, creates a baseline from pre-contrast volumes, and 
    subtracts the baseline to isolate the contrast enhancement signal.

    Parameters:
        img_list (list of SimpleITK.Image): A list of raw CTP images representing a time series.
        brain_mask (SimpleITK.Image): A binary mask indicating the brain region in the images.
        hu_threshold (int, optional): The Hounsfield Unit (HU) threshold to identify contrast agent presence. 
                                    Defaults to 150.
        ratio_threshold (float, optional): The threshold for the ratio of contrast agent signal change 
                                        to determine the starting index. Defaults to 0.05.

    Returns:
        tuple: A tuple containing:
            - img_np (numpy.ndarray): The baseline-corrected contrast enhancement signal 
                                        (raw signal minus averaged pre-contrast baseline).
            - s0_index (int): The starting index of significant contrast agent presence.
    """
    # Convert the list of SimpleITK images to a NumPy array
    img_np = np.stack([sitk.GetArrayFromImage(img) for img in img_list])
    brain_mask = sitk.GetArrayFromImage(brain_mask)

    # For each timepoint, store the amount of contrast agent that is present in the volume
    total_contrast_value = []
    
    # Iterate through each time point in the image series
    for t in range(img_np.shape[0]):
        
        # Get the current volume at time point t
        current_volume = img_np[t]

        # Extract the pixels which we deem to be part of the contrast signal. 
        # To be part of the contrast signal, pixels must be above the HU threshold and within the brain mask.
        contrast_mask = (current_volume > hu_threshold) & brain_mask
        contrast_pixels = current_volume[contrast_mask.astype('bool')]
        
        # Sum all contrast pixel values for this time point
        volume_contrast_sum = contrast_pixels.sum()
        
        # Add this volume's contrast sum to the list
        total_contrast_value.append(volume_contrast_sum)
    
    # Convert to numpy array
    total_contrast_value = np.array(total_contrast_value) 

    # Normalize the total contrast agent value by the first volume, to remove the baseline
    total_contrast_value = (total_contrast_value - total_contrast_value[0]) / total_contrast_value[0]
    
    # Extract the volume and index where the contrast agent starts to appear,
    # S0 is the baseline volume just before the actual contrast agent inflow
    cands = np.where(total_contrast_value > ratio_threshold)[0]
    if cands.shape[0] == 0:
        raise ValueError("No significant contrast agent presence detected.")
    s0_index = cands[0]
    
    # Average the volumes before the contrast agent starts to appear, to make a more stable baseline (S0)
    s0_img_np = img_np[:s0_index].mean(axis=0)
    
    # Remove the baseline signal S0 from the images
    # This leaves us with the contrast agent signal only
    img_np = img_np - s0_img_np
    img_np[img_np<0] = 0
    
    return img_np, s0_index
    
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
    time_index = np.array(time_index)
    
    # Filter time indices starting from s0_index and adjust relative to start time
    time_index_filtered = time_index[s0_index:] - time_index[s0_index]

    # Filter contrast data to exclude baseline measurements
    ctc_img_filtered = ctc_img[s0_index:]

    # Find time to peak by getting time index of maximum enhancement for each voxel
    ttp = time_index_filtered[ctc_img_filtered.argmax(axis=0)]
    
    # Set voxels outside brain mask to specified outside value
    ttp[sitk.GetArrayFromImage(brain_mask) == 0] = outside_value
    
    return ttp


def generate_perfusion_maps(ctc_img, time_index, brain_mask, aif_propperties, cSVD_thres=0.1, method='bcSVD1', outside_value=-1):
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
    brain_mask : sitk.Image or numpy.ndarray
        Binary mask defining brain tissue regions (1 = brain, 0 = background)
    aif_propperties : list or numpy.ndarray
        Arterial input function values or fitting parameters for AIF estimation
    cSVD_thres : float, optional
        SVD threshold for regularization as fraction of maximum singular value (default: 0.1)
    method : str, optional
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
    # Convert brain mask to numpy array if it's a SimpleITK image
    if type(brain_mask) is sitk.Image:
        brain_mask = sitk.GetArrayFromImage(brain_mask)
    else:
        brain_mask = brain_mask
        
    # Define tissue constants: density (g/ml) and hematocrit correction factor
    rho, H = 1.05, 0.85
    
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
    if method == 'oSVD':
        # Original SVD method using standard Toeplitz matrix
        G = toeplitz(aif_value, np.zeros(nr_timepoints))
        ctc_img_pad = ctc_img
    elif method == 'bcSVD1':
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
        ctc_img_pad = np.pad(ctc_img, [(0, ctc_img.shape[0]),] + [(0, 0)] * 3)
    elif method == 'bcSVD2':
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
        ctc_img_pad = np.pad(ctc_img, [(0, ctc_img.shape[0]),] + [(0, 0)] * 3)
    else:
        raise NotImplementedError(f"method {method} is not supported.")

    # Perform SVD decomposition on scaled convolution matrix
    U, S, V = np.linalg.svd(G * deltaT)
    
    # Apply threshold-based regularization to singular values
    thres = cSVD_thres * np.max(S)
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






    