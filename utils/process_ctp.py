from skimage.restoration import denoise_tv_chambolle
from utils.generate_brain_mask import *
from utils.generate_perfusion_maps import *
from utils.post_processing import *
from utils.viewers import *
from utils.data_loading import *
from utils.foss_vs_commercial import *
from utils.aif import *
import nibabel as nib
import numpy as np
import os 

def process_ctp(ctp_path, SCAN_INTERVAL, DEBUG, brain_mask_path=None):
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

    # ---------------------------------------------------------------------------------
    # Step 1: Read-in the 4D CTP .nii.gz scans, returns a list of 3D images and corresponding time indices
    # ---------------------------------------------------------------------------------
    
    volume_list, time_index = load_image(ctp_path)

    # Convert time index to seconds. For this we need to know the scan interval.
    time_index = [i * SCAN_INTERVAL for i in time_index]

    if DEBUG == True:
        view_4d_img(volume_list, title="Input CTP")
    
    # ---------------------------------------------------------------------------------
    # Step 2: Image preprocessing (smoothing and masking)
    # ---------------------------------------------------------------------------------

    # Smooth the input image while keeping edges intact.
    volume_list = [denoise_tv_chambolle(i, weight=0.2) for i in volume_list]

    if DEBUG == True:
        view_4d_img(volume_list, title="Smoothed CTP")

    # Either load a pre-existing brain mask or generate a one automatically
    if brain_mask_path is None:
        brain_mask, brain_mask_path = generate_brain_mask(volume_list[0], ctp_path)
    else:
        brain_mask = load_image(brain_mask_path)
    
    """
    Currently, we use a single 3D brain mask to mask the entire 4D CTP scan.
    However, if the patient moves during the scan, this mask may no longer allign with all 3D volumes.
    This leads to skull being included in the mask. Skull has high attenuation similar to the contrast agent,
    This may cause wrong aif selection and high CBF/CBV values at the edge of our perfusion maps.

    TODO:
    -One can either, exclude patients with motion from the dataset.
    -Or one can implement a motion correction algorithm to realign the volumes.
    -Or one could adapt the code to use 4D brain masks. This would require changes in many parts of the code though, increasing complexity and processing time.
    """

    if DEBUG == True:
        view_brain_mask(brain_mask, volume_list)

    # ---------------------------------------------------------------------------------
    # Step 3: Concentration Time Curve (CTC) Generation
    # ---------------------------------------------------------------------------------
    
    ctc_img, s0_index = extract_ctc(volume_list, brain_mask) 

    if DEBUG == True:
            view_4d_img(ctc_img, title="Contrast Signal (Smoothed CTP Minus S0)")

    # ---------------------------------------------------------------------------------
    # Step 4: TTP (time-to-peak) map generation
    # ---------------------------------------------------------------------------------
    
    ttp = generate_ttp(ctc_img, time_index, s0_index, brain_mask)
    
    if DEBUG == True:
        show_perfusion_map(ttp, "TTP (Before Post-Processing)")
    
    # ---------------------------------------------------------------------------------    
    # Step 5: Arterial input function (AIF) fitting
    # ---------------------------------------------------------------------------------

    aif_propperties, aif_candidate_segmentations = determine_aif(ctc_img, time_index, brain_mask, ttp, roi=None)

    if DEBUG == True:
        view_aif_selection(volume_list, time_index, ctc_img, s0_index, aif_propperties, aif_candidate_segmentations, brain_mask)

    # ---------------------------------------------------------------------------------
    # Step 6: Generate perfusion maps via deconvolution
    # ---------------------------------------------------------------------------------
    
    mtt, cbv, cbf, tmax = generate_perfusion_maps(ctc_img, time_index, brain_mask, aif_propperties, method='oSVD', cSVD_thres=0.1)

    if DEBUG == True:
        show_perfusion_map(ttp, "TTP")
        show_perfusion_map(cbf, "CBF")
        show_perfusion_map(cbv, "CBV")
        show_perfusion_map(mtt, "MTT")
        show_perfusion_map(tmax, "TMAX")

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
