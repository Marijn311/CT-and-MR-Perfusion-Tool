from utils.loading_and_preprocessing import *
from utils.generate_perfusion_maps import *
from utils.generate_brain_mask import *
from utils.foss_vs_commercial import *
from utils.post_processing import *
from utils.viewers import *
from utils.aif import *
import os 
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def process_ctp(ctp_path, SCAN_INTERVAL, DEBUG, IMAGE_TYPE, ECHO_TIME, brain_mask_path=None):
    """
    Process Computed Tomography Perfusion (CTP) scans to generate perfusion maps.
    This function performs a complete CTP analysis pipeline including image preprocessing,
    concentration time curve extraction, arterial input function determination, and
    perfusion parameter calculation using deconvolution methods.
    
    Parameters
    ----------
    ctp_path : str
        Path to the 4D CTP .nii.gz scan file.
    SCAN_INTERVAL : float
        Time interval between consecutive 3D scans in seconds.
    DEBUG : bool
        If True, displays intermediate results and debugging visualizations.
    brain_mask_path : str, optional
        Path to a pre-existing brain mask file. If None, a brain mask will be
        automatically generated and saved. Default is None.
    
    Returns
    -------
    cbf, cbv, mtt, tmax, brain_mask : numpy.ndarray

        The function saves the following perfusion maps as NIfTI files in a 
        'perfusion-maps' subdirectory:
        - foss_ttp.nii.gz: Time-to-peak map
        - foss_mtt.nii.gz: Mean transit time map
        - foss_cbv.nii.gz: Cerebral blood volume map
        - foss_cbf.nii.gz: Cerebral blood flow map
        - foss_tmax.nii.gz: Time to maximum of residue function map
    """ 

    # Determine the projection type based on the image type
    if IMAGE_TYPE == 'ctp':
        projection = 'max'
    elif IMAGE_TYPE == 'mrp':
        projection = 'min'


    # ---------------------------------------------------------------------------------
    # Step 1: Read-in the 4D CTP .nii.gz scans, returns a list of 3D images and corresponding time indices
    # ---------------------------------------------------------------------------------
    
    volume_list, time_index = load_image(ctp_path)

    # Convert time index to seconds. For this we need to know the scan interval.
    time_index = [i * SCAN_INTERVAL for i in time_index]

    if DEBUG == True:
        view_4d_img(volume_list, title=f"Input {IMAGE_TYPE.upper()}", projection=projection)

    # ---------------------------------------------------------------------------------
    # Step 2: Image preprocessing (smoothing and masking)
    # ---------------------------------------------------------------------------------

    # Smooth the input volunes to reduce noise and make results more stable 
    # Unfortunately this step can be very slow.
    volume_list = [gaussian_filter(volume, sigma=0.5) for volume in volume_list]

    if DEBUG == True:
        view_4d_img(volume_list, title=f"Smoothed {IMAGE_TYPE.upper()}", projection=projection)

    # Either load a pre-existing brain mask or generate a one automatically
    if brain_mask_path is None:
        if IMAGE_TYPE == 'ctp':
            brain_mask, brain_mask_path = generate_brain_mask_ctp(volume_list[0], ctp_path)
        if IMAGE_TYPE == 'mrp':
            brain_mask, brain_mask_path = generate_brain_mask_mrp(volume_list[0], ctp_path)
    else:
        brain_mask = load_image(brain_mask_path)

    if DEBUG == True:
        view_brain_mask(brain_mask, volume_list)

    # ---------------------------------------------------------------------------------
    # Step 3: Concentration Time Curve (CTC) Generation
    # ---------------------------------------------------------------------------------
    
    ctc_img, s0_index, bolus_data = extract_ctc(volume_list, brain_mask, echo_time=ECHO_TIME, image_type=IMAGE_TYPE) 
  
    if DEBUG == True:
            shows_contrast_curve(bolus_data, s0_index)
            view_4d_img(ctc_img, title="Contrast Signal", projection='max')

    # ---------------------------------------------------------------------------------
    # Step 4: TTP (time-to-peak) map generation
    # ---------------------------------------------------------------------------------
    
    ttp = generate_ttp(ctc_img, time_index, s0_index, brain_mask)
    
    if DEBUG == True:
        show_perfusion_map(ttp, "TTP (Before Post-Processing)", vmin=0, vmax=98)
    
    # ---------------------------------------------------------------------------------    
    # Step 5: Arterial input function (AIF) fitting
    # ---------------------------------------------------------------------------------

    aif_propperties, aif_candidate_segmentations, mean_fitting_error, aif_smoothness = determine_aif(ctc_img, time_index, brain_mask, ttp)

    if DEBUG == True:
        view_aif_selection(time_index, ctc_img, aif_propperties, aif_candidate_segmentations, brain_mask, mean_fitting_error, aif_smoothness, vmin=1, vmax=98)
    
    # ---------------------------------------------------------------------------------
    # Step 6: Generate perfusion maps via deconvolution
    # ---------------------------------------------------------------------------------
    
    mtt, cbv, cbf, tmax = generate_perfusion_maps(ctc_img, time_index, brain_mask, aif_propperties, method='oSVD', cSVD_thres=0.1)

    if DEBUG == True:
        show_perfusion_map(ttp, "TTP", vmin=1, vmax=99)
        show_perfusion_map(cbf, "CBF", vmin=1, vmax=99)
        show_perfusion_map(cbv, "CBV", vmin=1, vmax=99)
        show_perfusion_map(mtt, "MTT", vmin=1, vmax=99)
        show_perfusion_map(tmax, "TMAX", vmin=1, vmax=99)

    # ---------------------------------------------------------------------------------
    # Step 7: Post-processing (Whole brain normalization)
    # ---------------------------------------------------------------------------------
    
    ttp = post_process_perfusion_map(ttp, brain_mask, "TTP")
    cbf = post_process_perfusion_map(cbf, brain_mask, "CBF")
    cbv = post_process_perfusion_map(cbv, brain_mask, "CBV")
    mtt = post_process_perfusion_map(mtt, brain_mask, "MTT")
    tmax = post_process_perfusion_map(tmax, brain_mask, "TMAX")

    # ---------------------------------------------------------------------------------
    # Step 8: Save output as nii files
    # ---------------------------------------------------------------------------------
    
    # Ensure the arrays are properly saved by transposing them to the correct orientation (height, width, slices)
    ttp_nii = nib.Nifti1Image(np.transpose(ttp, (2, 1, 0)), np.eye(4))
    mtt_nii = nib.Nifti1Image(np.transpose(mtt, (2, 1, 0)), np.eye(4))
    cbv_nii = nib.Nifti1Image(np.transpose(cbv, (2, 1, 0)), np.eye(4))
    cbf_nii = nib.Nifti1Image(np.transpose(cbf, (2, 1, 0)), np.eye(4))
    tmax_nii = nib.Nifti1Image(np.transpose(tmax, (2, 1, 0)), np.eye(4))

    # Save the transposed arrays as NIfTI files
    nib.save(ttp_nii, os.path.join(os.path.dirname(ctp_path), "perfusion-maps", 'foss_ttp.nii.gz'))
    nib.save(mtt_nii, os.path.join(os.path.dirname(ctp_path), "perfusion-maps", 'foss_mtt.nii.gz'))
    nib.save(cbv_nii, os.path.join(os.path.dirname(ctp_path), "perfusion-maps", 'foss_cbv.nii.gz'))
    nib.save(cbf_nii, os.path.join(os.path.dirname(ctp_path), "perfusion-maps", 'foss_cbf.nii.gz'))
    nib.save(tmax_nii, os.path.join(os.path.dirname(ctp_path), "perfusion-maps", 'foss_tmax.nii.gz'))

    return brain_mask_path
