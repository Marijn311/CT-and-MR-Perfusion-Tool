from scipy.optimize import minimize
from scipy.interpolate import interp1d
import numpy as np


def NLR(ctc_volumes, aif, deltaT, mask, outside_value=-1):
    
    # Non-linear regression method using boxNLR model
    print("Using NLR method - this may take longer than SVD methods...")

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
        tissue_curve = ctc_volumes[:, z, y, x]
        
        # Fit NLR model
        fit_result = fit_nlr_single_voxel(aif, tissue_curve, deltaT)
        
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



def boxnlr(aif, tac, dt):
    """
    Estimate the perfusion parameters (CBV, MTT, and delay) that explain the
    tissue signal (tac) resulting from the input signal (aif). dt is the sample
    interval.
    
    This implementation follows the MATLAB boxnlr function structure.
    
    Parameters
    ----------
    aif : array_like
        Arterial input function
    tac : array_like
        Tissue concentration curve
    dt : float
        Sample interval (time step)
    
    Returns
    -------
    x : array_like
        Perfusion parameters [CBV, MTT, delay] with MTT and delay in seconds
    """
    # Initial estimates for CBV, MTT, and delay, respectively.
    # Divide MTT and delay by dt seconds to convert to unitless values.
    x0 = np.array([0.05, 4/dt, 1/dt])
    
    # Convolve AIF and TAC with a 3-point bandlimiting kernel with a FWHM of 2 samples.
    # Handle edges by nearest neighbor extrapolation.
    k = np.array([0.25, 0.5, 0.25])
    
    # Extend arrays with nearest neighbor extrapolation
    aif_extended = np.concatenate([[aif[0]], aif, [aif[-1]]])
    tac_extended = np.concatenate([[tac[0]], tac, [tac[-1]]])
    
    # Convolve and take valid part
    aif_k = np.convolve(aif_extended, k, mode='valid')
    tac_k = np.convolve(tac_extended, k, mode='valid')
    
    # Calculate the numerical integrand of the bandlimited AIF.
    # Note that this cumulative sum introduces a half sample shift.
    auc = np.cumsum(aif_k)
    
    # Find optimal values. Use scipy minimize with Nelder-Mead method
    # (equivalent to MATLAB's fminsearch)
    def objective(x):
        return fun(x, auc, tac_k)
    
    result = minimize(objective, x0, method='Nelder-Mead')
    x = result.x
    
    # Multiply MTT and delay by dt seconds to convert from unitless values.
    x = np.array([x[0], x[1]*dt, x[2]*dt])
    
    return x


def fun(x, auc, tac_k):
    """
    Calculate the sum of squared errors (sse) between the measured TAC and the
    TAC generated from the AIF using the perfusion parameters in x.
    
    Parameters
    ----------
    x : array_like
        Parameters [CBV, MTT (unitless), delay (unitless)]
    auc : array_like
        Cumulative sum of bandlimited AIF
    tac_k : array_like
        Bandlimited tissue concentration curve
    
    Returns
    -------
    sse : float
        Sum of squared errors
    """
    cbv, mtt_unitless, delay_unitless = x
    cbf = cbv / mtt_unitless if mtt_unitless > 0 else 0
    
    tac_est = gen_tac(cbf, mtt_unitless, delay_unitless, auc)
    sse = np.sum((tac_est - tac_k)**2)
    
    return sse


def gen_tac(cbf, mtt, d, auc):
    """
    Generate a TAC from the AIF and a set of perfusion parameters. When a box-
    shaped impulse response is assumed, then the TAC can be calculated as the
    difference of shifted integrands of AIF. Note that an additional half
    sample shift is required to correct for the shift introduced by the
    cumulative sum. In case negative shifts should be handled correctly,
    then interp1 needs to extrapolate the right side of the integrated AIF.
    
    Parameters
    ----------
    cbf : float
        Cerebral blood flow
    mtt : float
        Mean transit time (unitless)
    d : float
        Delay (unitless)
    auc : array_like
        Cumulative sum of AIF
    
    Returns
    -------
    tac : array_like
        Generated tissue concentration curve
    """
    n = len(auc)
    
    # Create interpolation indices with half sample shift correction
    indices = np.arange(1, n+1) - 0.5 - d
    indices_shifted = np.arange(1, n+1) - 0.5 - d - mtt
    
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
    tac = cbf * (a - b)
    
    return tac


def fit_nlr_single_voxel(aif, tissue_curve, dt):
    """
    Fit NLR model to a single voxel tissue curve using the boxnlr method.
    
    Parameters
    ----------
    aif : array_like
        Arterial input function
    tissue_curve : array_like
        Tissue concentration curve for single voxel
    dt : float
        Time step
    
    Returns
    -------
    result : dict
        Fitted parameters {'cbf': value, 'mtt': value, 'cbv': value, 'tmax': value}
    """
    try:
        # Use the boxnlr function following MATLAB implementation
        params = boxnlr(aif, tissue_curve, dt)
        
        cbv, mtt, delay = params
        cbf = cbv / (mtt / 60) if mtt > 0 else 0  # Convert MTT from seconds to minutes for CBF calculation
        
        # Calculate tmax as delay + mtt/2 (center of box function)
        tmax = delay + mtt / 2
        
        return {
            'cbf': cbf * 60,  # Convert to ml/100g/min
            'mtt': mtt,       # Already in seconds
            'cbv': cbv * 100, # Convert to ml/100g  
            'tmax': tmax,
            'success': True
        }
        
    except Exception as e:
        return {
            'cbf': 0,
            'mtt': 0,
            'cbv': 0,
            'tmax': 0,
            'success': False
        }


def process_voxel_chunk(voxel_coords_chunk, ctc_img, aif_value, deltaT):
    """
    Process a chunk of voxels for NLR fitting in parallel.
    
    Parameters
    ----------
    voxel_coords_chunk : list
        List of (z, y, x) coordinates for voxels to process
    ctc_img : numpy.ndarray
        4D array of tissue concentration curves
    aif_value : array_like
        Arterial input function values
    deltaT : float
        Time step
    
    Returns
    -------
    results : list
        List of tuples (z, y, x, fit_result) for each processed voxel
    """
    results = []
    for z, y, x in voxel_coords_chunk:
        # Extract tissue curve for this voxel
        tissue_curve = ctc_img[:, z, y, x]
        
        # Fit NLR model
        fit_result = fit_nlr_single_voxel(aif_value, tissue_curve, deltaT)
        
        # Store coordinate and result
        results.append((z, y, x, fit_result))
    
    return results
