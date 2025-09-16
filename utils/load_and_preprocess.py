import SimpleITK as sitk
import numpy as np
import ants
from config import *
from scipy.ndimage import gaussian_filter


def load_and_preprocess_raw_perf(perf_path, gaussian_sigma=1):
    """ 
    Load and preprocess raw 4D perfusion data from a NIfTI file.
    Parameters:
        - perf_path (str): Path to the .nii.gz perfusion image file. Expected input image shape is (t, z, y, x).
        - gaussian_sigma (float): Standard deviation for Gaussian kernel used in smoothing.
    Returns:
        - volume_list (list): List of 3D numpy arrays, each representing a volume at a specific timepoint.
        - time_index (list): List of time indices in seconds corresponding to each volume. 
    """
  
    image = sitk.ReadImage(perf_path)
    image = reorient_to_ras(image)
    image_array = sitk.GetArrayFromImage(image)

    # Create time index (in seconds) based on the number of timepoints and scan interval
    time_index = [i * SCAN_INTERVAL for i in range(image_array.shape[0])] 

    volumes, time_index = downsample_image(image_array, time_index) 
    volumes = motion_correction(volumes)

    # Convert the 4D array into a list of 3D arrays and corresponding time indices for easier processing
    volume_list = []
    for timepoint in range(volumes.shape[0]):
        volume = volumes[timepoint, :, :, :]
        volume_list.append(volume)
    
    # Smooth the input volumes to reduce noise and make results more stable
    volume_list = [gaussian_filter(volume, sigma=gaussian_sigma) for volume in volume_list]
        
    return volume_list, time_index


def load_and_preprocess_perf_map(perf_path):
    """ 
    Load and preprocess 3D perfusion maps (or brain mask) from a NIfTI file.
    Parameters:
        - perf_path (str): Path to the .nii.gz perfusion map file. Expected input image shape is (z, y, x).
    Returns:
        - volume (nd.array): 3D numpy array of the perfusion map (or brain mask)
    """
    image = sitk.ReadImage(perf_path)
    image = reorient_to_ras(image)
    image_array = sitk.GetArrayFromImage(image)
    volume = downsample_image(image_array)

    return volume


def reorient_to_ras(image):
    """
    Reorient an image to RAS (Right-Anterior-Superior) coordinate system.
    Sets the direction matrix to identity for 3D or 4D images, ensuring
    consistent orientation metadata without changing the actual image data.
    When arrays are extracted from the image, they will all be in the same orientation.
    
    Parameters:
        - image (sitk.image): A SimpleITK image object (3D or 4D)
    Returns:
        - image (sitk.image): A SimpleITK image object with updated direction matrix
    """

    array = sitk.GetArrayFromImage(image)
    
    if array.ndim == 4:
        image.SetDirection((1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0))
   
    if array.ndim == 3:
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
        
    return image


def downsample_image(image_array, time_index=None, t_limit=100, z_limit=25, y_limit=250, x_limit=250):
    """
    Downsample a 3D or 4D image array to reduce dimensions if they exceed specified limits.
    For spatial dimensions (x, y), maintains aspect ratio when downsampling. For temporal 
    and z dimensions, applies direct size limits. 
    We opt for downsampling instead of interpolation to significantly reduce computation time.
    
    Parameters:
        - image_array (numpy.ndarray): 3D or 4D image array to downsample (z, y, x) or (t, z, y, x)
        - time_index (list, optional): List of time indices in seconds corresponding to each volume.
        - t_limit (int, optional): Maximum time dimension size.
        - z_limit (int, optional): Maximum z dimension size.
        - y_limit (int, optional): Maximum y dimension size.
        - x_limit (int, optional): Maximum x dimension size.
    Returns:
        - downsampled_image (nd.array): downsampled image array with reduced dimensions
        - downsampled_time_index (list, optional): Downsampled time index if 4D array and time_index provided
    """

    def generate_indices(original_size, target_size):
        """Generate evenly spaced indices to downsample an array dimension"""
        if target_size >= original_size:
            return np.arange(original_size)
        indices = np.linspace(0, original_size - 1, target_size, dtype=int)
        return indices
    
    def calculate_spatial_dims(x_size, y_size, x_limit, y_limit):
        """Calculate new x and y dimensions while maintaining aspect ratio"""
        if x_size <= x_limit and y_size <= y_limit:
            return x_size, y_size
        
        x_scale = x_limit / x_size if x_size > x_limit else 1.0
        y_scale = y_limit / y_size if y_size > y_limit else 1.0
        scale_factor = min(x_scale, y_scale)
        
        return int(x_size * scale_factor), int(y_size * scale_factor)
    
    if image_array.ndim == 4:
        
        t_size, z_size, y_size, x_size = image_array.shape
        new_t = min(t_size, t_limit) # Limit time dimension in case the the time dimension exceeds the limit
        new_z = min(z_size, z_limit) # Limit z dimension in case the the z dimension exceeds the limit
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit) # If either x or y dimension exceeds the limit, limit both dimensions in the same way to maintain aspect ratio

        # Generate indices for each dimension, the indices are evenly spaced over the original dimension
        indices = [
            generate_indices(t_size, new_t),
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]

        # Downsample the image by keeping only the voxels at the generated indices
        downsampled_image = image_array[np.ix_(*indices)]
        
        # If time_index is provided, downsample it using the same indices as the time dimension
        if time_index is not None:
            downsampled_time_index = [time_index[i] for i in indices[0]]
            return downsampled_image, downsampled_time_index
        
        return downsampled_image

    if image_array.ndim == 3:
    
        z_size, y_size, x_size = image_array.shape
        
        new_z = min(z_size, z_limit) # Limit z dimension in case the the z dimension exceeds the limit
        new_x, new_y = calculate_spatial_dims(x_size, y_size, x_limit, y_limit) # If either x or y dimension exceeds the limit, limit both dimensions in the same way to maintain aspect ratio
        
        # Generate indices for each dimension, the indices are evenly spaced over the original dimension
        indices = [
            generate_indices(z_size, new_z),
            generate_indices(y_size, new_y),
            generate_indices(x_size, new_x)
        ]
        
        # Downsample the image by keeping only the voxels at the generated indices
        downsampled_image = image_array[np.ix_(*indices)]
        
        return downsampled_image


def motion_correction(image_array, type_of_transform='QuickRigid'):
    """
    Perform motion correction on a 4D image array by registering all 3D volumes to the first 3D volume.
    Registration is done using ANTsPy with the specified transformation type.
    
    Parameters:
        - image_array (nd.array): 4D numpy array with shape (t, z, y, x) representing the image sequence to be motion corrected
        - type_of_transform (str, optional): Type of transformation for registration.
    Returns:
        - motion_corrected_array (nd.array): Motion-corrected 4D numpy array with the same shape as input,
            where all volumes are aligned to the first volume as reference
    """
    
    # Convert the 4D numpy array a list of 3D ANTs images
    ants_images = []
    for t in range(image_array.shape[0]):
        volume = image_array[t, :, :, :]
        ants_img = ants.from_numpy(volume)
        ants_images.append(ants_img)
    
    # Use first volume as reference
    reference = ants_images[0]
    
    # Align all other volumes to the reference
    aligned_volumes = []
    aligned_volumes.append(ants_images[0])  # The first volume is already aligned to itself
    
    # Apply motion correction to all subsequent volumes
    for t in range(1, len(ants_images)):
        print(f"Motion correcting volume {t+1}/{len(ants_images)}")
        moving = ants_images[t]
        
        # Perform rigid registration
        registration = ants.registration(
            fixed=reference,
            moving=moving,
            type_of_transform=type_of_transform
        )
        
        # Apply transformation to get aligned volume
        aligned_volume = registration['warpedmovout']
        aligned_volumes.append(aligned_volume)
    
    # Convert list of aligned 3D ANTs images back to a 4D numpy array
    motion_corrected_array = np.zeros_like(image_array)
    for t, aligned_vol in enumerate(aligned_volumes):
        motion_corrected_array[t, :, :, :] = aligned_vol.numpy()
    
    return motion_corrected_array