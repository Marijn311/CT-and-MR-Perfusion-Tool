import os
import numpy as np
import SimpleITK as sitk
from skimage import measure, segmentation
from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_erosion, binary_dilation


def generate_brain_mask_ctp(volume, perf_path):
    """
    Generate a brain mask from a CT perfusion volume using fast marching segmentation.
    This function creates a binary brain mask by identifying the largest bone structure
    as a seed point and using fast marching algorithm to segment the brain region.
    Args:
        volume (numpy.ndarray): 3D CT perfusion volume array
        perf_path (str): Path to the perfusion file, used to determine output directory
    Returns:
        tuple: A tuple containing:
            - mask (numpy.ndarray): Binary brain mask as a 3D array
            - mask_path (str): Path where the brain mask was saved
    """


    print("Generating a brain mask...")
    # Convert the input array to sitk image
    volume = sitk.GetImageFromArray(volume)
    
    # Set the orientation to RAS 
    volume.SetDirection((1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0))
    
    # Smooth the input image using a Gaussian filter to reduce noise
    volume= sitk.DiscreteGaussian(volume)

    # Identify connected components in the image where intensity is greater than 300 (likely bone regions)
    boneLabelMap = sitk.ConnectedComponent(volume > 300)

    # Compute statistics for each connected component
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(boneLabelMap)

    # Extract labels, centroids, and physical sizes of connected components
    label_list = [
        (i, volume.TransformPhysicalPointToIndex(label_statistics.GetCentroid(i)), label_statistics.GetPhysicalSize(i))
        for i in label_statistics.GetLabels()
    ]

    # Find the largest connected component based on physical size and use its centroid as the seed point
    seed = max(label_list, key=lambda x: x[2])[1]

    # Compute the gradient magnitude of the image to highlight edges
    feature_volume = sitk.GradientMagnitudeRecursiveGaussian(volume, sigma=0.5)

    # Compute the reciprocal of the gradient magnitude to create a speed image for fast marching
    speed_volume = sitk.BoundedReciprocal(feature_volume)

    # Initialize the fast marching filter with the seed point and stopping value
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(400)

    # Perform fast marching to segment the region around the seed point
    fm_volume = fm_filter.Execute(speed_volume)

    # Create a binary mask by thresholding the fast marching result
    mask = fm_volume < 400
    
    # Save the generated brain mask to a file
    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    sitk.WriteImage(mask, mask_path)
    mask = sitk.GetArrayFromImage(mask)

    return mask, mask_path



def generate_brain_mask_mrp(volume, perf_path):
    """
    Generate a brain mask from a 3D volume using morphological active contours.
    Args:
        volume (numpy.ndarray): 3D input volume data
        perf_path (str): Path to the CTP file, used to determine output directory
    Returns:
        tuple: A tuple containing:
            - mask (numpy.ndarray): Binary brain mask as uint8 array
            - mask_path (str): Path where the mask file was saved
    """
    
    print("Generating a brain mask...")
    mask = np.zeros(volume.shape)  

    # Make a 3D binary mask, this is needed as input for the active contours.
    seed_mask = volume > 100
    
    # Perform the segmentation using active contours
    mask = segmentation.morphological_chan_vese(volume, num_iter=300, 
                                                       init_level_set=seed_mask)
    mask = mask.astype(float)
    
    # Per slice: Fill the holes in the mask, and remove small unconnected components
    for s in range(volume.shape[0]):
        mask_slice = mask[s, :, :] 

        # Sometimes a bit of the skull is still attached in the segmentation because some part
        # of the skull was very close to the brain. To remove this, we erode the mask a bit. 
        # This will hopefully un-connect the skull fragments from the brain in 3D.
        # Then we remove all components that are not attached to the main component (which is the brain).  
        # Then we dilate the mask again to get the original size of the brain mask back.
        # This also smooths the mask a bit.
        structure_element = disk(5)
        mask_slice = binary_erosion(mask_slice, structure_element)
        labeled_mask = measure.label(mask_slice)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            areas = [region.area for region in regions]
            max_area_idx = np.argmax(areas)
            max_area = areas[max_area_idx]   # Find the largest component
    
            for region in regions:
                if region.area < max_area: # Remove smaller components
                    mask_slice[labeled_mask == region.label] = 0
        
        mask_slice = binary_dilation(mask_slice, structure_element)
        
        # Fill holes in the slice and update the mask to include the slice version with filled holes
        mask[s, :, :] = binary_fill_holes(mask_slice)
    
    # Remove all components that are not attached to the main component in 3D
    labeled_mask_3d = measure.label(mask)
    regions_3d = measure.regionprops(labeled_mask_3d)
    
    if regions_3d:
        areas_3d = [region.area for region in regions_3d]
        max_area_idx_3d = np.argmax(areas_3d)
        max_area_3d = areas_3d[max_area_idx_3d] # Find the largest component
        
        # Remove all components except the largest one
        for region in regions_3d:
            if region.area < max_area_3d:
                mask[labeled_mask_3d == region.label] = 0

    mask = mask.astype(np.uint8)
    
    # Save the generated mask to a file
    mask_path = os.path.join(os.path.dirname(perf_path), 'brain_mask.nii.gz')
    mask_sitk = sitk.GetImageFromArray(mask)

    # Set the orientation to RAS
    mask_sitk.SetDirection((1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0))
    sitk.WriteImage(mask_sitk, mask_path)

    
    return mask, mask_path

