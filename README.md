<div align="center">
  <img src="icon.png" alt="Icon" width="300"/>
</div>

## Introduction

This repository contains PyPeT (Python Perfusion Tool): a open Python tool for generating perfusion maps from head CTP scans using deconvolution methods. See the attached arXiv paper for an in-depth description of the processing steps and visual comparisons with a FDA-approved tools.

The tool supports multiple deconvolution methods:
- **oSVD**: Original Singular Value Decomposition using Toeplitz matrix
- **bcSVD1**: Block-circulant SVD method 1 with padding
- **bcSVD2**: Block-circulant SVD method 2 with custom matrix construction  
- **nlr**: Non-linear regression using boxNLR model (Bennink et al.)

**DISCLAIMER**: This algorithm has **not been formally validated** and is **not intended for clinical use**.

## Data Structure

This tool expects input data to follow a specific directory structure format. The expected structure is demonstrated in the `demo_data` folder:

```
data/
├── sub-<subject_id>/
│   └── ses-<session_id>/
│       ├── sub-<subject_id>_ses-<session_id>_ctp.nii.gz     # Raw CTP data (4D volume)
│       ├── brain_mask.nii.gz                                # (Optional) Brain mask
│       └── perfusion-maps/                                  # (Optional) Reference perfusion maps
│           ├── sub-<subject_id>_ses-<session_id>_cbf.nii.gz # Cerebral Blood Flow
│           ├── sub-<subject_id>_ses-<session_id>_cbv.nii.gz # Cerebral Blood Volume
│           ├── sub-<subject_id>_ses-<session_id>_mtt.nii.gz # Mean Transit Time
│           └── sub-<subject_id>_ses-<session_id>_tmax.nii.gz # Time to Maximum
```

This tool can either generate brain masks using a thresholding and fast-marching approach, or you may provide your own brain mask.

## Configuration

The tool is configured via the `config.py` file. Key parameters include:

- **METHOD**: Deconvolution method ('oSVD', 'bcSVD1', 'bcSVD2', or 'nlr')
- **DATASET_PATH**: Path to your data directory
- **SCAN_INTERVAL**: Time between consecutive 3D volumes (seconds)
- **DEBUG**: Enable visualization of intermediate results
- **CSVD_THRES**: SVD threshold for regularization (SVD methods only)
- **NLR_MAX_ITER**: Maximum iterations for NLR optimization (NLR method only)
- **NLR_TOL**: Convergence tolerance for NLR (NLR method only)

Example configuration for NLR method:
```python
METHOD = 'nlr'
NLR_MAX_ITER = 1000
NLR_TOL = 1e-6
```

Note: The NLR method typically takes longer than SVD methods but may provide more robust results in noisy conditions.

## Citation

If you use this tool, please cite: "PyPeT: A Python Tool for Automated Quantitative Brain CT Perfusion Analysis and Visualization" [insert arXiv DOI]

## References

This work was based on the tutorial by https://github.com/turtleizzy/ctp_csvd

The cSVD deconvolution algorithm was implemented based on:
- https://github.com/SethLirette/CTP 
- https://github.com/marcocastellaro/dsc-mri-toolbox
- Zanderigo F, Bertoldo A, Pillonetto G, Cobelli C. Nonlinear stochastic regularization to characterize tissue residue function in bolus-tracking MRI: assessment and comparison with SVD, block-circulant SVD, and Tikhonov. IEEE Trans Biomed Eng. 2009;56(5):1287-97. doi: 10.1109/TBME.2009.2013820.

The NLR (Non-Linear Regression) method uses a box-shaped residue function and optimization techniques:
- Bennink E, Oosterbroek J, Kudo K, Viergever MA, Velthuis BK, de Jong HWAM. Fast nonlinear regression method for CT brain perfusion analysis. J Med Imaging (Bellingham). 2013;1(2):026001. doi: 10.1117/1.JMI.1.2.026001.
