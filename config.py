
DATASET_PATH = r"demo_data"        # Path to the dataset directory containing CTP scans
IMAGE_TYPE = 'ctp'                      # Either 'mrp' or 'ctp'. NOTE: CTP implementation is based in iodine contrast scans, MRP on gadolinium contrast DSC scans.
SCAN_INTERVAL = 1.41                    # Time between two 3D consecutive volumes in seconds
ECHO_TIME = 0.03                        # Echo time in seconds (only for MRP)

#--------------------------------------------------------------------------

DEBUG = False                                # If True, activates debug mode which shows plots during processing to visualize intermediate results
GENERATE_PERFUSION_MAPS = True              # If True, generates perfusion maps from the inputted raw 4D perfusion data
SHOW_COMPARISONS = True                     # If True, shows comparison plots between generated and reference perfusion maps
CALCULATE_METRICS = True                    # If True, calculates similarity metrics between generated and reference perfusion maps

# --------------------------------------------------------------------------

assert IMAGE_TYPE in ['ctp', 'mrp'], "IMAGE_TYPE must be either 'ctp' or 'mrp'"
if DEBUG == True:
    assert GENERATE_PERFUSION_MAPS == True, "DEBUG mode only applies when you are generating perfusion maps."
if IMAGE_TYPE == 'ctp':
    PROJECTION = 'max'
elif IMAGE_TYPE == 'mrp':
    PROJECTION = 'min'
    
# --------------------------------------------------------------------------

