import ants
import numpy as np
import SimpleITK as sitk

def load_image(path):
    """ Reads a NIfTI file, reshape the volume(s) if required, and returns a list of 3D volumes plus their time indices for a 4d input.
    Or return a 3D volume for a 3D input.
    """
    # Read the volumes using sitk
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    
    if image_array.ndim == 4:
    
        image = reorient_to_ras(image)
        image_array = sitk.GetArrayFromImage(image)

        # First downsample the image if needed, this will speed up computation.
        volumes = resize_images(image_array)  

        
        volumes = motion_correction(volumes)

        img_list = []
        time_index = []
        nr_timepoints = volumes.shape[0]
        for timepoint in range(nr_timepoints):
            volume = volumes[timepoint, :, :, :]
            img_list.append(volume)
            time_index.append(timepoint)
        return img_list, time_index
    
    if image_array.ndim == 3:

        image = reorient_to_ras(image)
        image_array = sitk.GetArrayFromImage(image)

        # First downsample the image if needed, this will speed up computation.
        volume = resize_images(image_array)

        return volume



def resize_images(image_array):
    """Takes a 3D or 4D array and resamples the images to have smaller dimensions 
    if some of the dimension exceed a given limit.

    x and y limit are 250
    z limit is 20
    t limit is 50 (only for 4D images)
    
    Args:
        nii_path: Path to the 3D or 4D NIfTI file
      
    """
    
    if image_array.ndim == 4:
        # Shape: (t, z, y, x)
        t_size, z_size, y_size, x_size = image_array.shape

        # Define limits
        t_limit, z_limit, y_limit, x_limit = 50, 20, 250, 250

        # Calculate target sizes for t and z
        new_t = min(t_size, t_limit)
        new_z = min(z_size, z_limit)

        # Calculate target sizes for x and y while maintaining aspect ratio
        if x_size <= x_limit and y_size <= y_limit:
            # Both within limits, no resizing needed
            new_x = x_size
            new_y = y_size
        else:
            # Calculate scale factors
            x_scale = x_limit / x_size if x_size > x_limit else 1.0
            y_scale = y_limit / y_size if y_size > y_limit else 1.0
            
            # Use the most restrictive scale factor to maintain aspect ratio
            scale_factor = min(x_scale, y_scale)
            
            new_x = int(x_size * scale_factor)
            new_y = int(y_size * scale_factor)

        # Generate indices for each dimension
        def generate_indices(original_size, target_size):
            if target_size >= original_size:
                return np.arange(original_size)
            indices = np.linspace(0, original_size - 1, target_size, dtype=int)
            return indices
        
        t_indices = generate_indices(t_size, new_t)
        z_indices = generate_indices(z_size, new_z)
        y_indices = generate_indices(y_size, new_y)
        x_indices = generate_indices(x_size, new_x)
        
        # Use advanced indexing to sample the array
        resampled_array = image_array[np.ix_(t_indices, z_indices, y_indices, x_indices)]
        return resampled_array
        
    if image_array.ndim == 3:
        # Shape: (z, y, x)
        z_size, y_size, x_size = image_array.shape

        # Define limits (no t dimension)
        z_limit, y_limit, x_limit = 20, 250, 250

        # Calculate target size for z
        new_z = min(z_size, z_limit)
        
        # Calculate target sizes for x and y while maintaining aspect ratio
        if x_size <= x_limit and y_size <= y_limit:
            # Both within limits, no resizing needed
            new_x = x_size
            new_y = y_size
        else:
            # Calculate scale factors
            x_scale = x_limit / x_size if x_size > x_limit else 1.0
            y_scale = y_limit / y_size if y_size > y_limit else 1.0
            
            # Use the most restrictive scale factor to maintain aspect ratio
            scale_factor = min(x_scale, y_scale)
            
            new_x = int(x_size * scale_factor)
            new_y = int(y_size * scale_factor)
        
        # Generate indices for each dimension
        def generate_indices(original_size, target_size):
            if target_size >= original_size:
                return np.arange(original_size)
            indices = np.linspace(0, original_size - 1, target_size, dtype=int)
            return indices
        
        z_indices = generate_indices(z_size, new_z)
        y_indices = generate_indices(y_size, new_y)
        x_indices = generate_indices(x_size, new_x)
        
        # Use advanced indexing to sample the array
        resampled_array = image_array[np.ix_(z_indices, y_indices, x_indices)]
        return resampled_array


def reorient_to_ras(image):

    array = sitk.GetArrayFromImage(image)
    if array.ndim == 4:
        # Simply set the direction matrix for the 4D image directly
        # This doesn't change the actual data, just the orientation metadata.
        # Such that the extracted arrays from the image will all be in the same orientation.
        image.SetDirection((1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.0, 0.0, 0.0, 1.0))
   
    if array.ndim == 3:
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
    return image



def motion_correction(image_4d):
    # Convert numpy array to ANTs images
    ants_images = []
    for t in range(image_4d.shape[0]):
        volume = image_4d[t, :, :, :]
        ants_img = ants.from_numpy(volume)
        ants_images.append(ants_img)
    
    # Use first volume as reference
    reference = ants_images[0]
    
    # Align all volumes to the reference
    aligned_volumes = []
    aligned_volumes.append(ants_images[0])  # First volume doesn't need alignment
    
    for t in range(1, len(ants_images)):
        print(f"Motion correcting volume {t+1}/{len(ants_images)}")
        moving = ants_images[t]
        
        # Perform rigid registration
        registration = ants.registration(
            fixed=reference,
            moving=moving,
            type_of_transform='QuickRigid'
        )
        
        # Apply transformation to get aligned volume
        aligned_volume = registration['warpedmovout']
        aligned_volumes.append(aligned_volume)
    
    # Convert back to numpy array
    corrected_array = np.zeros_like(image_4d)
    for t, aligned_vol in enumerate(aligned_volumes):
        corrected_array[t, :, :, :] = aligned_vol.numpy()
    
    return corrected_array