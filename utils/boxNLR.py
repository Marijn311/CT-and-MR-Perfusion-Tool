from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np


def boxNLR(ctc_volumes, aif, dt, mask, outside_value=-1):
    """
    This is the main function which generates perfusion maps using the
    boxNLR (box Non-Linear Regression) approach by bennink et al.
    (https://doi.org/10.1117/1.JMI.3.2.026003)
    
    Be aware that this is just my attempt at implementing the method,
    This approach is very slow and I am not sure it is implemented correctly.
    So please use with caution and verify results carefully. 

    Parameters:
        - ctc_volumes (list): List of 3D arrays (z,y,x) containing the contrast time curves (CTCs).
        - aif (nd.array): 1D array containing the AIF signal.
        - dt (float): Time step (sampling interval) in seconds.
        - mask (nd.array): 3D binary brain mask (z,y,x).
        - outside_value (float, optional): Value to assign to voxels outside the brain mask in the output maps.

    Returns:
        - mtt (nd.array): Mean transit time (MTT) map.
        - cbv (nd.array): Cerebral blood volume (CBV) map.
        - cbf (nd.array): Cerebral blood flow (CBF) map.
        - tmax (nd.array): Time to peak (Tmax) map.
    """
    
    # Non-linear regression method using boxNLR model
    print("Using NLR method - this may take longer than SVD methods...")

    # Convert list of 3D volumes to a 4D array (t, z, y, x)
    ctc_volumes = np.stack(ctc_volumes, axis=0)

    # Initialize output arrays
    cbf = np.zeros(ctc_volumes.shape[1:])
    cbv = np.zeros(ctc_volumes.shape[1:])
    mtt = np.zeros(ctc_volumes.shape[1:])
    tmax = np.zeros(ctc_volumes.shape[1:])
    
    # Get brain voxel coordinates
    brain_coords = np.where(mask == 1)
    total_voxels = len(brain_coords[0])
    
    print(f"Processing {total_voxels} brain voxels with NLR...")
    
    # Process each brain voxel
    for idx, (z, y, x) in enumerate(zip(*brain_coords)):
        if idx % 1000 == 0:
            print(f"Processed {idx}/{total_voxels} voxels ({100*idx/total_voxels:.1f}%)")
        
        # Extract tissue curve for this voxel
        ctc = ctc_volumes[:, z, y, x]
        
        # Fit NLR model
        fit_result = fit_boxNLR(aif, ctc, dt)
        
        # Store results
        cbf[z, y, x] = fit_result['cbf']
        cbv[z, y, x] = fit_result['cbv']
        mtt[z, y, x] = fit_result['mtt']
        tmax[z, y, x] = fit_result['tmax']
    
    # Apply brain mask to set outside values for all perfusion maps
    cbf[mask == 0] = outside_value
    cbv[mask == 0] = outside_value
    mtt[mask == 0] = outside_value
    tmax[mask == 0] = outside_value
    
    print("NLR processing completed!")
    return mtt, cbv, cbf, tmax



def fit_boxNLR(aif, ctc, dt):
    """
    Optimize (fit) the box-shaped residue function such that it explains the measured CTC with the given AIF. 
    The box-shaped residue function is defined by three parameter: CBV, MTT, and delay. 
    The function returns the optimal values for these parameters. 
    Hence by fitting the box-shaped residue function, we obtain perfusion parameters directly.
    
    Parameters:
        - aif (nd.array): 1D array containing the arterial input function
        - ctc (nd.array): 1D array containing the contrast time curve for a single voxel
        - dt (float): Sample interval (time step)

    Returns: 
        - results (dict): Dictionary containing the optimized perfusion parameters [CBV, MTT, delay].
    """

    
    # Initial estimates for CBV, MTT, and delay, respectively.
    # Divide MTT and delay by dt seconds to convert to unitless values.
    params = np.array([0.05, 4/dt, 1/dt])

    # Create a 3-point bandlimiting kernel with a FWHM of 2 samples.
    kernel = np.array([0.25, 0.5, 0.25])
    
    # Extend arrays with nearest neighbor extrapolation, such that convolution does not reduce length
    aif_extended = np.concatenate([[aif[0]], aif, [aif[-1]]])
    ctc_extended = np.concatenate([[ctc[0]], ctc, [ctc[-1]]])
    
    # Convolve the AIF and measured CTC with the kernel to obtain bandlimited versions
    aif_band_limited = np.convolve(aif_extended, kernel, mode='valid')
    ctc_band_limited = np.convolve(ctc_extended, kernel, mode='valid')

    # Calculate the numerical integrand of the bandlimited AIF.
    # Note that this cumulative sum introduces a half sample shift.
    auc = np.cumsum(aif_band_limited)

    # Use scipy minimize with Nelder-Mead method to find the optimal parameters for the box-shaped residue function.
    # Optimal is defined as the box-model parameters that minimize the sum of squared errors between the measured CTC and the estimated CTC.
    # The estimated CTC is generated from the AIF (auc) and the box-shaped residue function.
    def objective(params):
        ctc_estimated = generate_ctc(params, auc)
        sse = np.sum((ctc_estimated - ctc_band_limited)**2)
        return sse
    optimal_params = minimize(objective, params, method='Nelder-Mead').x


    # Multiply MTT and delay by dt seconds to convert from unitless values.
    optimal_params = np.array([optimal_params[0], optimal_params[1]*dt, optimal_params[2]*dt])

    cbv, mtt, delay = optimal_params
    cbf = cbv / (mtt / 60) if mtt > 0 else 0  # Convert MTT from seconds to minutes for CBF calculation
    
    # Calculate tmax as delay + mtt/2 (center of box function)
    tmax = delay + mtt / 2

    results = {
        'cbf': cbf * 60,  # Convert to ml/100g/min
        'mtt': mtt,       # Already in seconds
        'cbv': cbv * 100, # Convert to ml/100g  
        'tmax': tmax,
    }
    return results
        



def generate_ctc(params, auc):
    """
    Generate a CTC from the AIF and a set of perfusion parameters (box-model parameters). 
    When a box-shaped residue function is assumed, then the CTC can be calculated as the
    difference of shifted integrands of AIF. Note that an additional half
    sample shift is required to correct for the shift introduced by the
    cumulative sum. In case negative shifts should be handled correctly,
    then interp1 needs to extrapolate the right side of the integrated AIF.
    
    Parameters: 
        - params (nd.array): 1D array containing the three box-model parameters [CBF, MTT, delay].
        - auc (nd.array): 1D array containing the numerical integrand of the AIF.

    Returns:
        - ctc (nd.array): Generated tissue concentration curve
    """

    cbf, mtt, delay = params

    n = len(auc)
    
    # Create interpolation indices with half sample shift correction
    indices = np.arange(1, n+1) - 0.5 - delay
    indices_shifted = np.arange(1, n+1) - 0.5 - delay - mtt
    
    # Create interpolation function for AUC
    auc_interp = interp1d(np.arange(len(auc)), auc, kind='linear', 
                         bounds_error=False, fill_value=0)
    
    # Interpolate at shifted indices
    a = auc_interp(indices)
    b = auc_interp(indices_shifted)
    
    # Handle NaN values (set to 0)
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    
    # Calculate TAC as difference of shifted integrands
    ctc = cbf * (a - b)
    
    return ctc

