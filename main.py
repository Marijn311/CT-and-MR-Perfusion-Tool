import os
from utils.process_ctp import *
from utils.foss_vs_commercial import *

print("\nStarting CTP processing...")

DATASET_PATH = r"demo_data" # Path to the dataset directory containing CTP scans
SCAN_INTERVAL = 2.0         # Time between two 3D consecutive images in seconds
DEBUG = False                # Opens interactive plots during processing to visualize intermediate results


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
        ctp_path = os.path.join(dir_path, "ses-01", dir+"_ses-01_ctp.nii.gz")
        brain_mask_path = os.path.join(dir_path, "ses-01", "brain_mask.nii.gz")
        if not os.path.exists(brain_mask_path):
            brain_mask_path=None
        
        # Define paths to files used for validation 
        com_perfusion_path = os.path.join(dir_path, "ses-01", "perfusion-maps")
        com_cbf_path = os.path.join(com_perfusion_path, dir+"_ses-01_cbf.nii.gz")
        com_cbv_path = os.path.join(com_perfusion_path, dir+"_ses-01_cbv.nii.gz")
        com_mtt_path = os.path.join(com_perfusion_path, dir+"_ses-01_mtt.nii.gz")
        com_tmax_path = os.path.join(com_perfusion_path, dir+"_ses-01_tmax.nii.gz")

        # Generate perfusion maps from input CTP 
        cbf, cbv, mtt, tmax, brain_mask = process_ctp(ctp_path, SCAN_INTERVAL, DEBUG, brain_mask_path)

        # Compare the generated perfusion maps with reference maps
        metrics = compare_perfusion_maps(cbf, cbv, mtt, tmax, com_cbf_path, com_cbv_path, com_mtt_path, com_tmax_path, brain_mask, plot=True)
        all_metrics.append(metrics)

    # Compute average metrics across the dataset
    average_metrics_on_dataset(all_metrics)
    
    print("Finished Processing CTP data!")
    break

