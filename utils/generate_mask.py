import os
import numpy as np
import SimpleITK as sitk
from skimage import measure, segmentation
from scipy.ndimage import binary_fill_holes
from utils.load_and_preprocess import reorient_to_ras
from skimage.morphology import disk, binary_erosion, binary_dilation



def generate_mask_ctp(volume, perf_path, bone_threshold=300, fast_marching_threshold=400):
    """
    Generate a brain mask from a 3D CT perfusion volume.
    This function creates a binary brain mask by identifying the largest connected bone structure (the skull)
    as a seed point and using fast marching algorithm to segment the brain region.
    
    Parameters:
        volume (numpy.ndarray): 3D CT perfusion volume array (z, y, x)
        perf_path (str): Path to the perfusion file, used to determine output directory for saving the generated mask.
        bone_threshold (int): Intensity threshold to identify bone structures in the CT image.
        fast_marching_threshold (int): Stopping value for the fast marching algorithm to define the brain region.
    Returns:
        tuple: A tuple containing:
            - mask (numpy.ndarray): Binary brain mask as a 3D array (z, y, x)
            - mask_path (str): Path where the brain mask was saved
    """

    print("Generating a brain mask...")

    image = sitk.GetImageFromArray(volume)
    image = reorient_to_ras(image)
    
    # Smooth the input image using a Gaussian filter to reduce noise
    image = sitk.DiscreteGaussian(image)

    # Identify connected components in the image where intensity is greater than the bone threshold
    boneLabelMap = sitk.ConnectedComponent(image > bone_threshold)

    # Compute statistics for each connected component
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(boneLabelMap)

    # Extract labels, centroids, and physical sizes of connected components
    label_list = [
        (i, image.TransformPhysicalPointToIndex(label_statistics.GetCentroid(i)), label_statistics.GetPhysicalSize(i))
        for i in label_statistics.GetLabels()
    ]

    # Find the largest connected component based on physical size and use its centroid as the seed point
    seed = max(label_list, key=lambda x: x[2])[1]

    # Compute the gradient magnitude of the image to highlight edges
    feature_image = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=0.5)

    # Compute the reciprocal of the gradient magnitude to create a speed image for fast marching
    speed_image = sitk.BoundedReciprocal(feature_image)

    # Initialize the fast marching filter with the seed point and stopping value
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(fast_marching_threshold)

    # Perform fast marching to segment the region around the seed point
    fm_image = fm_filter.Execute(speed_image)

    # Create a binary mask by thresholding the fast marching result
    mask = fm_image < fast_marching_threshold

    # Save the generated brain mask to a file
    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    sitk.WriteImage(mask, mask_path)
    
    mask = sitk.GetArrayFromImage(mask)

    return mask, mask_path



def generate_mask_mrp(volume, perf_path, brain_threshold=100):
    """
    Generate a brain mask from a 3D mrp volume.
    This function generates a brain mask by using simple thresholding and morphological active contours.
    
    Parameters:
        volume (numpy.ndarray): 3D input volume data
        perf_path (str): Path to the CTP file, used to determine output directory
        brain_threshold (int): Intensity threshold to create the initial seed for the active contour segmentation.
    Returns:
        tuple: A tuple containing:
            - mask (numpy.ndarray): Binary brain mask as uint8 array
            - mask_path (str): Path where the mask file was saved
    """
    
    print("Generating a brain mask...")
    mask = np.zeros(volume.shape)  

    # Make a 3D binary mask, this is needed as input for the active contours.
    seed_mask = volume > brain_threshold

    # Perform the segmentation using active contours
    mask = segmentation.morphological_chan_vese(volume, num_iter=300, init_level_set=seed_mask).astype(float)
    

    # Sometimes a bit of the skull is still included in the segmentation of the active contour.
    # This may happen if the skull seems to touch the brain for example. 
    # To detach these skull fragments, we erode the mask a bit. 
    # This will hopefully un-connect the skull fragments from the brain.
    # Then we remove all components that are not attached to the main component (which is the brain).  
    # Then we dilate the mask again to get the original size of the brain mask back.
    # This also smooths the mask a bit.

    # Use a disk structuring element for 2D slices and extend it to 3D.
    # Since we have way fewer slices in z-direction, we only erode/dilate in x and y direction
    struct_elem = disk(4)
    struct_elem = struct_elem[np.newaxis, :, :] # Make it 3D
    mask = binary_erosion(mask, struct_elem)
    labeled_mask_3d = measure.label(mask)
    regions_3d = measure.regionprops(labeled_mask_3d)
    
    # If there are regions found, keep only the largest one (the brain)
    if regions_3d:
        areas_3d = [region.area for region in regions_3d]
        max_area_idx_3d = np.argmax(areas_3d)
        max_area_3d = areas_3d[max_area_idx_3d] # Find the largest component
        
        # Remove all components except the largest one
        for region in regions_3d:
            if region.area < max_area_3d:
                mask[labeled_mask_3d == region.label] = 0
    
    mask = binary_dilation(mask, struct_elem)
    mask = binary_fill_holes(mask)

    mask = mask.astype(np.uint8)
    
    # Save the generated mask to a file
    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    mask_sitk = sitk.GetImageFromArray(mask)

    # Set the orientation to RAS
    mask_sitk = reorient_to_ras(mask_sitk)
    sitk.WriteImage(mask_sitk, mask_path)

    
    return mask, mask_path

