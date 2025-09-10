from utils.compare_to_reference_maps import *
from core import *
from config import *
import os

"""
This main script loops over all patients in a perfusion imaging dataset.
For each patient, it generates perfusion maps (CBF, CBV, MTT, TTP, TMAX) from the raw 4D perfusion data.
It then compares the generated maps to provided reference maps.
In case reference maps are not available, the comparison step can be skipped by setting the `SHOW_COMPARISONS` flag to `True`.
The main options can be set in the config.py file.
Smaller adjustments such as thresholds and kernel sizes can be adjusted in the respective utility files.
PyPeT assumes a specific directory structure for the dataset. The required structure is described in the README file.
"""

print("\nStarting CTP processing...")

# Store similarity metrics for every scan in the dataset that was compared to a reference map
all_metrics = []

# Loop over all directories in the dataset
for root, dirs, files in os.walk(DATASET_PATH):
    for dir in dirs:
        print("\n\n")
        print("="*60)
        print(f"Processing directory: {dir}")    
        print("="*60)

        # Define paths to input files
        dir_path = os.path.join(root, dir)
        perf_path = os.path.join(dir_path, "ses-01", dir+"_ses-01_ctp.nii.gz")
        mask_path = os.path.join(dir_path, "ses-01", "brain_mask.nii.gz")
        if not os.path.exists(mask_path):
            mask_path=None
        
        # Generate perfusion maps from inputted perfusion data
        if GENERATE_PERFUSION_MAPS:
            core(perf_path, mask_path)
        
        if SHOW_COMPARISONS or CALCULATE_METRICS:
            # Define paths to files used for validation 
            ref_perfusion_path = os.path.join(dir_path, "ses-01", "perfusion-maps")
            ref_cbf_path = os.path.join(ref_perfusion_path, dir+"_ses-01_cbf.nii.gz")
            ref_cbv_path = os.path.join(ref_perfusion_path, dir+"_ses-01_cbv.nii.gz")
            ref_mtt_path = os.path.join(ref_perfusion_path, dir+"_ses-01_mtt.nii.gz")
            ref_ttp_path = os.path.join(ref_perfusion_path, dir+"_ses-01_ttp.nii.gz")
            ref_tmax_path = os.path.join(ref_perfusion_path, dir+"_ses-01_tmax.nii.gz")
            gen_cbf_path = os.path.join(ref_perfusion_path, "generated_CBF.nii.gz")
            gen_cbv_path = os.path.join(ref_perfusion_path, "generated_CBV.nii.gz")
            gen_mtt_path = os.path.join(ref_perfusion_path, "generated_MTT.nii.gz")
            gen_ttp_path = os.path.join(ref_perfusion_path, "generated_TTP.nii.gz")
            gen_tmax_path = os.path.join(ref_perfusion_path, "generated_TMAX.nii.gz")
            mask_path = os.path.join(dir_path, "ses-01", "brain_mask.nii.gz")

            # Compare the generated perfusion maps with reference maps
            metrics = compare_with_reference_maps(gen_cbf_path, gen_cbv_path, gen_mtt_path, gen_ttp_path, gen_tmax_path, ref_cbf_path, ref_cbv_path, ref_mtt_path, ref_ttp_path, ref_tmax_path, mask_path)
            all_metrics.append(metrics)

    # Compute average metrics across the dataset
    if CALCULATE_METRICS:
        average_metrics_on_dataset(all_metrics)
    
    print("\nFinished Processing CTP data!")
    break

