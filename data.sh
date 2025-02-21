#!/bin/bash

# Download the dataset
gdown -q 1GZarrBZg02m9ROIJdop4KY1rsC9IQt0w

# Unzip the downloaded dataset
unzip -q msk-mri-dataset-v2.zip

# Create directories for the test set
mkdir -p msk-mri-testset/BrainMRI
mkdir -p msk-mri-testset/BrainMSK

# Move BrainMRI files
mv BrainMRI/BraTS-SSA-00010-000-t2f.nii.gz msk-mri-testset/BrainMRI
mv BrainMRI/BraTS-SSA-00026-000-t2f.nii.gz msk-mri-testset/BrainMRI
mv BrainMRI/BraTS-SSA-00041-000-t2f.nii.gz msk-mri-testset/BrainMRI
mv BrainMRI/BraTS-SSA-00057-000-t2f.nii.gz msk-mri-testset/BrainMRI
mv BrainMRI/BraTS-SSA-00095-000-t2f.nii.gz msk-mri-testset/BrainMRI
mv BrainMRI/BraTS-SSA-00141-000-t2f.nii.gz msk-mri-testset/BrainMRI

# Move BrainMSK files
mv BrainMSK/BraTS-SSA-00010-000-t2f.nii.gz msk-mri-testset/BrainMSK
mv BrainMSK/BraTS-SSA-00026-000-t2f.nii.gz msk-mri-testset/BrainMSK
mv BrainMSK/BraTS-SSA-00041-000-t2f.nii.gz msk-mri-testset/BrainMSK
mv BrainMSK/BraTS-SSA-00057-000-t2f.nii.gz msk-mri-testset/BrainMSK
mv BrainMSK/BraTS-SSA-00095-000-t2f.nii.gz msk-mri-testset/BrainMSK
mv BrainMSK/BraTS-SSA-00141-000-t2f.nii.gz msk-mri-testset/BrainMSK


# chmod +x prepare_data.sh, make executable
# ./data.sh, run