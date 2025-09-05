
DATASET_PATH = r"demo_data_unitobrain"      # Path to the dataset directory containing CTP scans
SCAN_INTERVAL = 1.0                         # Time between two 3D consecutive volumes in seconds
ECHO_TIME = 0.03                            # Echo time in seconds (only for MRP)
IMAGE_TYPE = 'ctp'                          # Either 'mrp' or 'ctp'
METHOD = 'oSVD'                             # Which deconvolution method to use, either 'bcSVD1', 'bcSVD2' or 'oSVD'
CSVD_THRES = 0.1                            # Threshold for the SVD truncation in case of the block-circulant SVD methods    
DEBUG = True                               # if True, shows plots during processing to visualize intermediate results
SHOW_COMPARISONS = True                     # If True, shows comparison plots between generated and reference perfusion maps



# --------------------------------------------------------------------------
assert IMAGE_TYPE in ['ctp', 'mrp'], "IMAGE_TYPE must be either 'ctp' or 'mrp'"
assert METHOD in ['bcSVD1', 'bcSVD2', 'oSVD'], "METHOD must be either 'bcSVD1', 'bcSVD2' or 'oSVD'"
# --------------------------------------------------------------------------

# Determine the projection type based on the image type
if IMAGE_TYPE == 'ctp':
    PROJECTION = 'max'
elif IMAGE_TYPE == 'mrp':
    PROJECTION = 'min'
