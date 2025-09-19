from utils.generate_perfusion_maps import *
from viewers.view_contrast_curve import *
from utils.compare_to_reference_maps import *
from viewers.view_perfusion_map import *
from utils.load_and_preprocess import *
from utils.generate_mask import *
from utils.determine_aif import *
from viewers.view_4d_img import *
from utils.post_process import *
from viewers.view_mask import *
from viewers.view_aif import *
from config import *
import os 
import SimpleITK as sitk

def core(perf_path, mask_path=None):
    """
    Core function to process perfusion scans and generate perfusion maps.
    This function performs the complete processing pipeline, by calling on different modular functions.

    Parameters
    ----------
    perf_path : str
        Path to the raw 4D perfusion data .nii.gz file.
    mask_path : str, optional
        Path to a brain mask file. If None, a brain mask will be
        automatically generated and saved.
    
    Returns
    -------
        The function saves the following perfusion maps as NIfTI files in a 
        'perfusion-maps' subdirectory:
        - generated_ttp.nii.gz: Time-to-peak map
        - generated_mtt.nii.gz: Mean transit time map
        - generated_cbv.nii.gz: Cerebral blood volume map
        - generated_cbf.nii.gz: Cerebral blood flow map
        - generated_tmax.nii.gz: Time to maximum of residue function map
        The brain mask is saved to the perfusion directory if it was generated automatically.
    """

    # ---------------------------------------------------------------------------------
    # Step 1: Load and Preprocess the Input Data
    # ---------------------------------------------------------------------------------
    
    volume_list, time_index = load_and_preprocess_raw_perf(perf_path)

    if DEBUG:
        view_4d_img(volume_list, title=f"Input {IMAGE_TYPE.upper()} After Preprocessing", projection=PROJECTION, vmin=1, vmax=95)

    # ---------------------------------------------------------------------------------
    # Step 2: Brain Mask Generation
    # ---------------------------------------------------------------------------------

    # Either load a pre-existing brain mask or generate a one automatically
    if mask_path is None:
        if IMAGE_TYPE == 'ctp':
            mask, mask_path = generate_mask_ctp(volume_list[0], perf_path)
        if IMAGE_TYPE == 'mrp':
            mask, mask_path = generate_mask_mrp(volume_list[0], perf_path)
    else:
        mask = load_and_preprocess_perf_map(mask_path)

    if DEBUG == True:
        view_mask(mask, volume_list)

    # ---------------------------------------------------------------------------------
    # Step 3: Contrast Time Curve (CTC) Generation
    # ---------------------------------------------------------------------------------
    
    ctc_volumes, s0_index, bolus_data = extract_ctc(volume_list, mask) 
  
    if DEBUG == True:
            view_contrast_curve(bolus_data, time_index, s0_index)
            view_4d_img(ctc_volumes, title="Contrast Signal", projection='max')

    # ---------------------------------------------------------------------------------
    # Step 4: TTP (time-to-peak) map generation
    # ---------------------------------------------------------------------------------
    
    ttp = generate_ttp(ctc_volumes, time_index, s0_index, mask)
    
    if DEBUG == True:
        view_perfusion_map(ttp, "TTP (Before Post-Processing)", vmin=0, vmax=98)
    
    # ---------------------------------------------------------------------------------    
    # Step 5: Arterial input function (AIF) fitting
    # ---------------------------------------------------------------------------------

    aif_properties, aif_candidate_segmentations = determine_aif(ctc_volumes, time_index, mask, ttp)

    if DEBUG == True:
        view_aif(time_index, ctc_volumes, aif_properties, aif_candidate_segmentations, mask, vmin=1, vmax=98)
    
    # ---------------------------------------------------------------------------------
    # Step 6: Generate perfusion maps via deconvolution
    # ---------------------------------------------------------------------------------
    
    mtt, cbv, cbf, tmax = generate_perfusion_maps(ctc_volumes, time_index, mask, aif_properties)

    if DEBUG == True:
        view_perfusion_map(ttp, "TTP", vmin=2, vmax=98)
        view_perfusion_map(cbf, "CBF", vmin=2, vmax=98)
        view_perfusion_map(cbv, "CBV", vmin=2, vmax=98)
        view_perfusion_map(mtt, "MTT", vmin=2, vmax=98)
        view_perfusion_map(tmax, "TMAX", vmin=2, vmax=98)

    # ---------------------------------------------------------------------------------
    # Step 7: Post-processing (Whole brain normalization)
    # ---------------------------------------------------------------------------------
    
    ttp = post_process_perfusion_map(ttp, mask, "TTP")
    cbf = post_process_perfusion_map(cbf, mask, "CBF")
    cbv = post_process_perfusion_map(cbv, mask, "CBV")
    mtt = post_process_perfusion_map(mtt, mask, "MTT")
    tmax = post_process_perfusion_map(tmax, mask, "TMAX")

    # ---------------------------------------------------------------------------------
    # Step 8: Save output as nii files
    # ---------------------------------------------------------------------------------

    # Make image objects from the arrays
    ttp = sitk.GetImageFromArray(ttp)
    mtt = sitk.GetImageFromArray(mtt)
    cbv = sitk.GetImageFromArray(cbv)
    cbf = sitk.GetImageFromArray(cbf)
    tmax = sitk.GetImageFromArray(tmax)

    # Save the images to the 'perfusion-maps' subdirectory
    sitk.WriteImage(ttp, os.path.join(os.path.dirname(perf_path), "perfusion-maps", 'generated_ttp.nii.gz'))
    sitk.WriteImage(mtt, os.path.join(os.path.dirname(perf_path), "perfusion-maps", 'generated_mtt.nii.gz'))
    sitk.WriteImage(cbv, os.path.join(os.path.dirname(perf_path), "perfusion-maps", 'generated_cbv.nii.gz'))
    sitk.WriteImage(cbf, os.path.join(os.path.dirname(perf_path), "perfusion-maps", 'generated_cbf.nii.gz'))
    sitk.WriteImage(tmax, os.path.join(os.path.dirname(perf_path), "perfusion-maps", 'generated_tmax.nii.gz'))