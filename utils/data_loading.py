import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np

def load_image(path):
    """ Reads a NIfTI file, samples it if required and returns a list of 3D SimpleITK images and their corresponding time indices."""
    
    # First resize the image if needed
    volumes = resize_images(path)
    array = sitk.GetArrayFromImage(volumes)
    # print(array.ndim)

    # check the number of dimensions
    if array.ndim == 4:
        img_list = []
        time_index = []
        for count, volume in enumerate(sitk.GetArrayFromImage(volumes)):
            img = sitk.GetImageFromArray(volume)
            img = sitk.Cast(img, sitk.sitkFloat32)
            img = sitk.DICOMOrient(img, 'RAS')
       
            # Copy spacing and metadata from the resized volume
            img.SetSpacing(volumes.GetSpacing())
            img.SetOrigin(volumes.GetOrigin())
            img.SetDirection(volumes.GetDirection())
            
            img_list.append(img)
            time_index.append(count)
        return img_list, time_index

    # In case we are loading a brain mask or perfusion image, just return
    if array.ndim == 3:
        return volumes



def load_brain_mask(brain_mask_path, ctp_img):
    """
    Given a path to the brain mask image, this function loads the mask and resizes it to match the input image.
    """

    # Use only the first 3D volume
    ctp_img = ctp_img[0]

    # Load the brain mask
    brain_mask = sitk.ReadImage(brain_mask_path)

    # Convert to 8-bit unsigned integer
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)

    # Reshape the slice dimensions x and y to match the ctp image using zoom
    # Get the size of the input image (first image in the list)
    input_size = ctp_img.GetSize()
    mask_size = brain_mask.GetSize()
    
    # Calculate zoom factors for x and y dimensions
    zoom_factors = [input_size[0] / mask_size[0], input_size[1] / mask_size[1], 1.0]
    
    # Convert to numpy array, apply zoom, and convert back to SimpleITK
    mask_array = sitk.GetArrayFromImage(brain_mask)
    resized_mask_array = zoom(mask_array, zoom_factors[::-1], order=0)  # Reverse order for numpy (z,y,x)
    brain_mask = sitk.GetImageFromArray(resized_mask_array.astype(np.uint8))
    
    # Copy spacing and origin from input image
    brain_mask.SetSpacing(ctp_img.GetSpacing())
    brain_mask.SetOrigin(ctp_img.GetOrigin())
    brain_mask.SetDirection(ctp_img.GetDirection())

    return brain_mask


def resize_images(nii_path):
    """Takes the path to a 3D or 4D nii.gz images and resamples the images to have smaller dimensions 
    if some of the dimension exceed a given limit.

    x and y limit are 250
    z limit is 20
    t limit is 50 (only for 4D images)
    
    Args:
        nii_path: Path to the 3D or 4D NIfTI file
      
    """
    
    # Load the image
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    
    # Check if image is 3D or 4D
    is_4d = len(img_array.shape) == 4
    
    if is_4d:
        # Shape: (t, z, y, x)
        t_size, z_size, y_size, x_size = img_array.shape
        
        # Define limits
        t_limit, z_limit, y_limit, x_limit = 50, 20, 250, 250

        # Calculate target sizes
        new_t = min(t_size, t_limit)
        new_z = min(z_size, z_limit)
        new_y = min(y_size, y_limit)
        new_x = min(x_size, x_limit)
        
        # Check if resizing is needed
        needs_resize = (new_t < t_size) or (new_z < z_size) or (new_y < y_size) or (new_x < x_size)
        
        if not needs_resize:
            return img
        
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
        resampled_array = img_array[np.ix_(t_indices, z_indices, y_indices, x_indices)]
        
    else:
        # Shape: (z, y, x)
        z_size, y_size, x_size = img_array.shape
        
        # Define limits (no t dimension)
        z_limit, y_limit, x_limit = 20, 250, 250

        # Calculate target sizes
        new_z = min(z_size, z_limit)
        new_y = min(y_size, y_limit)
        new_x = min(x_size, x_limit)
        
        # Check if resizing is needed
        needs_resize = (new_z < z_size) or (new_y < y_size) or (new_x < x_size)
        
        if not needs_resize:
            return img
        
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
        resampled_array = img_array[np.ix_(z_indices, y_indices, x_indices)]
    
    # Convert back to SimpleITK image
    resampled_img = sitk.GetImageFromArray(resampled_array)

    return resampled_img

