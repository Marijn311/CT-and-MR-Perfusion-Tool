import SimpleITK as sitk
import os

def generate_brain_mask(img, ctp_path):
    """
    Generate a brain mask from a given image using connected components and fast marching methods.
    """

    # Convert the input array to sitk image
    img = sitk.GetImageFromArray(img)
    # Identity matrix = RAS 
    img.SetDirection((1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0))
    
    # Smooth the input image using a Gaussian filter to reduce noise
    img= sitk.DiscreteGaussian(img)

    # Identify connected components in the image where intensity is greater than 300 (likely bone regions)
    boneLabelMap = sitk.ConnectedComponent(img > 300)

    # Compute statistics for each connected component
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(boneLabelMap)

    # Extract labels, centroids, and physical sizes of connected components
    label_list = [
        (i, img.TransformPhysicalPointToIndex(label_statistics.GetCentroid(i)), label_statistics.GetPhysicalSize(i))
        for i in label_statistics.GetLabels()
    ]

    # Find the largest connected component based on physical size and use its centroid as the seed point
    seed = max(label_list, key=lambda x: x[2])[1]

    # Compute the gradient magnitude of the image to highlight edges
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(img, sigma=0.5)

    # Compute the reciprocal of the gradient magnitude to create a speed image for fast marching
    speed_img = sitk.BoundedReciprocal(feature_img)

    # Initialize the fast marching filter with the seed point and stopping value
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints([seed])
    fm_filter.SetStoppingValue(400)

    # Perform fast marching to segment the region around the seed point
    fm_img = fm_filter.Execute(speed_img)

    # Create a binary mask by thresholding the fast marching result
    out = fm_img < 400
    
    brain_mask_path = os.path.join(os.path.dirname(ctp_path), 'brain_mask.nii.gz')
    sitk.WriteImage(out, brain_mask_path)

    # Convert the output to a binary mask
    out = sitk.GetArrayFromImage(out)

    return out, brain_mask_path