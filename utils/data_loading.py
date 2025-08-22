import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np

def load_ctp(path):
    """ Reads a 4D NIfTI file and returns a list of 3D SimpleITK images and their corresponding time indices."""
    volumes = sitk.ReadImage(path)
    img_list = []
    time_index = []
    for count, volume in enumerate(sitk.GetArrayFromImage(volumes)):
        img= sitk.GetImageFromArray(volume)
        img= sitk.DICOMOrient(img, 'RAS')
        img_list.append(img)
        time_index.append(count)
    return img_list, time_index


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