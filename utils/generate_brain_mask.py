import SimpleITK as sitk
import os
import numpy as np
from skimage import measure, segmentation
from skimage.morphology import disk, binary_erosion, binary_dilation
from scipy.ndimage import binary_fill_holes



def generate_brain_mask_ctp(volume, ctp_path):
    """
    Generate a brain mask from a given image using connected components and fast marching methods.
    """

    # Convert the input array to sitk image
    volume = sitk.GetImageFromArray(volume)
    # Identity matrix = RAS 
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
    out = fm_volume < 400
    
    brain_mask_path = os.path.join(os.path.dirname(ctp_path), 'brain_mask.nii.gz')
    sitk.WriteImage(out, brain_mask_path)

    # Convert the output to a binary mask
    out = sitk.GetArrayFromImage(out)

    return out, brain_mask_path



def generate_brain_mask_mrp(volume, ctp_path):
    
    print("Generating a brain mask...")
    mask = np.zeros(volume.shape)  # Initialize mask for brain data

    
    # Make a 3D binary mask, this is needed as input for the active contours.
    # Creates a binary mask by setting elements to true (1) where the corresponding 
    # elements in the volume are greater than 100, and false (0) otherwise.
    seed_mask = volume > 100
    
    
    # Perform the segmentation using active contours (morphological snakes)
    # Using morphological active contours as a Python equivalent to MATLAB's activecontour
    mask = segmentation.morphological_chan_vese(volume, num_iter=300, 
                                                       init_level_set=seed_mask)
    mask = mask.astype(float)  # Convert mask from boolean to numeric (0 or 1)
    
    # Per slice: Fill the holes in the mask, and remove small unconnected components
    for s in range(volume.shape[0]):
        mask_slice = mask[s, :, :]  # take a slice of the mask

        # Sometimes a bit of the skull is still attached in the segmentation because some part
        # of the skull was very close to the brain. To remove this, we erode the mask a bit. 
        # This will hopefully un-connect the skull fragments from the brain in 3D.
        # Then we remove all components that are not attached to the main component (which is the brain).  
        # Then we dilate the mask again to get the original size of the brain mask back.
        # This also smooths the mask a bit.
        structure_element = disk(5)
        mask_slice = binary_erosion(mask_slice, structure_element)
        
        # Eliminate smaller connected components and keep those larger than options['mask']['npixel']
        labeled_mask = measure.label(mask_slice)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            # Find the largest component
            areas = [region.area for region in regions]
            max_area_idx = np.argmax(areas)
            max_area = areas[max_area_idx]
            
            # Remove smaller components
            for i, region in enumerate(regions):
                if region.area < max_area:
                    mask_slice[labeled_mask == region.label] = 0
        
        mask_slice = binary_dilation(mask_slice, structure_element)
        
        # Fill holes in the slice and update the mask to include the slice version with filled holes
        mask[s, :, :] = binary_fill_holes(mask_slice)
    
    # Remove all components that are not attached to the main component in 3D
    labeled_mask_3d = measure.label(mask)
    regions_3d = measure.regionprops(labeled_mask_3d)
    
    if regions_3d:
        # Find the largest component
        areas_3d = [region.area for region in regions_3d]
        max_area_idx_3d = np.argmax(areas_3d)
        max_area_3d = areas_3d[max_area_idx_3d]
        
        # Remove all components except the largest one
        for region in regions_3d:
            if region.area < max_area_3d:
                mask[labeled_mask_3d == region.label] = 0

    # Convert to unit8
    mask = (mask * 255).astype(np.uint8)
    
    # Save the generated mask to a file
    brain_mask_path = os.path.join(os.path.dirname(ctp_path), 'brain_mask.nii.gz')
    mask_sitk = sitk.GetImageFromArray(mask)

    # Identity matrix = RAS 
    mask_sitk.SetDirection((1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0))
    sitk.WriteImage(mask_sitk, brain_mask_path)

    
    return mask, brain_mask_path

