#!/bin/bash

# Download Africa-dataset 
gdown -q 1GZarrBZg02m9ROIJdop4KY1rsC9IQt0w
mkdir -p msk-mri-africa-dataset

# Unzip the downloaded dataset
unzip -q msk-mri-dataset-v2.zip -d msk-mri-africa-dataset

# Create directories for the Africa test set
mkdir -p msk-mri-africa-testset/BrainMRI
mkdir -p msk-mri-africa-testset/BrainMSK

# Move Africa BrainMRI files
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00010-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00026-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00041-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00057-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00095-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI
mv msk-mri-africa-dataset/BrainMRI/BraTS-SSA-00141-000-t2f.nii.gz msk-mri-africa-testset/BrainMRI

# Move Africa BrainMSK files
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00010-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00026-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00041-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00057-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00095-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK
mv msk-mri-africa-dataset/BrainMSK/BraTS-SSA-00141-000-t2f.nii.gz msk-mri-africa-testset/BrainMSK

# Download BraTS2021 dataset
gdown -q 10U6-C2sd6ha03jVKjTdzvCRwCA5iVKze
unzip -q msk-mri-dataset-2021-1C.zip

# Create directories for the BraTS21 test set
mkdir -p msk-mri-brats21-testset/BrainMRI
mkdir -p msk-mri-brats21-testset/BrainMSK

# List of files you want to put into the test set
test_set_21=(
'BraTS2021_01149_flair.nii.gz'
'BraTS2021_00642_flair.nii.gz'
'BraTS2021_01665_flair.nii.gz'
'BraTS2021_00191_flair.nii.gz'
'BraTS2021_00334_flair.nii.gz'
'BraTS2021_01013_flair.nii.gz'
'BraTS2021_01003_flair.nii.gz'
'BraTS2021_00582_flair.nii.gz'
'BraTS2021_01129_flair.nii.gz'
'BraTS2021_00124_flair.nii.gz'
'BraTS2021_01223_flair.nii.gz'
'BraTS2021_00282_flair.nii.gz'
'BraTS2021_01506_flair.nii.gz'
'BraTS2021_01371_flair.nii.gz'
'BraTS2021_01392_flair.nii.gz'
'BraTS2021_01461_flair.nii.gz'
'BraTS2021_01513_flair.nii.gz'
'BraTS2021_00316_flair.nii.gz'
'BraTS2021_01042_flair.nii.gz'
'BraTS2021_01067_flair.nii.gz'
'BraTS2021_00459_flair.nii.gz'
'BraTS2021_01381_flair.nii.gz'
'BraTS2021_01181_flair.nii.gz'
'BraTS2021_01198_flair.nii.gz'
'BraTS2021_01648_flair.nii.gz'
'BraTS2021_01020_flair.nii.gz'
'BraTS2021_00340_flair.nii.gz'
'BraTS2021_00443_flair.nii.gz'
'BraTS2021_01545_flair.nii.gz'
'BraTS2021_01480_flair.nii.gz'
'BraTS2021_00704_flair.nii.gz'
'BraTS2021_00764_flair.nii.gz'
'BraTS2021_00270_flair.nii.gz'
'BraTS2021_01265_flair.nii.gz'
'BraTS2021_00194_flair.nii.gz'
'BraTS2021_01056_flair.nii.gz'
'BraTS2021_01241_flair.nii.gz'
'BraTS2021_01083_flair.nii.gz'
'BraTS2021_00426_flair.nii.gz'
'BraTS2021_00322_flair.nii.gz'
'BraTS2021_01277_flair.nii.gz'
'BraTS2021_01193_flair.nii.gz'
'BraTS2021_00571_flair.nii.gz'
'BraTS2021_00612_flair.nii.gz'
'BraTS2021_00705_flair.nii.gz'
'BraTS2021_01207_flair.nii.gz'
'BraTS2021_01580_flair.nii.gz'
'BraTS2021_01630_flair.nii.gz'
'BraTS2021_00488_flair.nii.gz'
'BraTS2021_00214_flair.nii.gz'
'BraTS2021_01350_flair.nii.gz'
'BraTS2021_00162_flair.nii.gz'
'BraTS2021_01170_flair.nii.gz'
'BraTS2021_01001_flair.nii.gz'
'BraTS2021_00309_flair.nii.gz'
'BraTS2021_00624_flair.nii.gz'
'BraTS2021_00251_flair.nii.gz'
'BraTS2021_00192_flair.nii.gz'
'BraTS2021_00572_flair.nii.gz'
'BraTS2021_01443_flair.nii.gz'
'BraTS2021_01442_flair.nii.gz'
'BraTS2021_01484_flair.nii.gz'
'BraTS2021_01242_flair.nii.gz'
'BraTS2021_01325_flair.nii.gz'
'BraTS2021_00332_flair.nii.gz'
'BraTS2021_00380_flair.nii.gz'
'BraTS2021_00275_flair.nii.gz'
'BraTS2021_01661_flair.nii.gz'
'BraTS2021_01499_flair.nii.gz'
'BraTS2021_00791_flair.nii.gz'
'BraTS2021_00271_flair.nii.gz'
'BraTS2021_01458_flair.nii.gz'
'BraTS2021_00201_flair.nii.gz'
'BraTS2021_00329_flair.nii.gz'
'BraTS2021_01008_flair.nii.gz'
'BraTS2021_00000_flair.nii.gz'
'BraTS2021_00496_flair.nii.gz'
'BraTS2021_00346_flair.nii.gz'
'BraTS2021_01572_flair.nii.gz'
'BraTS2021_01558_flair.nii.gz'
'BraTS2021_01257_flair.nii.gz'
'BraTS2021_01337_flair.nii.gz'
'BraTS2021_01416_flair.nii.gz'
'BraTS2021_00156_flair.nii.gz'
'BraTS2021_00407_flair.nii.gz'
'BraTS2021_01164_flair.nii.gz'
'BraTS2021_00605_flair.nii.gz'
'BraTS2021_01637_flair.nii.gz'
'BraTS2021_00390_flair.nii.gz'
'BraTS2021_00121_flair.nii.gz'
'BraTS2021_01365_flair.nii.gz'
'BraTS2021_01019_flair.nii.gz'
'BraTS2021_01310_flair.nii.gz'
'BraTS2021_01168_flair.nii.gz'
'BraTS2021_01387_flair.nii.gz'
'BraTS2021_00839_flair.nii.gz'
'BraTS2021_00558_flair.nii.gz'
'BraTS2021_00740_flair.nii.gz'
'BraTS2021_00103_flair.nii.gz'
'BraTS2021_01456_flair.nii.gz'
'BraTS2021_00122_flair.nii.gz'
'BraTS2021_01138_flair.nii.gz'
'BraTS2021_01642_flair.nii.gz'
'BraTS2021_00999_flair.nii.gz'
'BraTS2021_01596_flair.nii.gz'
'BraTS2021_01532_flair.nii.gz'
'BraTS2021_01347_flair.nii.gz'
'BraTS2021_01397_flair.nii.gz'
'BraTS2021_00324_flair.nii.gz'
'BraTS2021_01074_flair.nii.gz'
'BraTS2021_01070_flair.nii.gz'
'BraTS2021_00493_flair.nii.gz'
'BraTS2021_00105_flair.nii.gz'
'BraTS2021_00513_flair.nii.gz'
'BraTS2021_01388_flair.nii.gz'
'BraTS2021_01406_flair.nii.gz'
'BraTS2021_00110_flair.nii.gz'
'BraTS2021_00731_flair.nii.gz'
'BraTS2021_00138_flair.nii.gz'
'BraTS2021_01215_flair.nii.gz'
'BraTS2021_01049_flair.nii.gz'
'BraTS2021_01631_flair.nii.gz'
'BraTS2021_00290_flair.nii.gz'
'BraTS2021_01469_flair.nii.gz'
'BraTS2021_00293_flair.nii.gz'
)

# Now move each file from the unzipped 2021 dataset to your new testset dirs
# (Adjust the source paths if they differ from "msk-mri-dataset-2021-1C/BrainMRI" etc.)
for file in "${test_set_21[@]}"
do
  mv "msk-mri-dataset-2021/BrainMRI/${file}" msk-mri-brats21-testset/BrainMRI/
  mv "msk-mri-dataset-2021/BrainMSK-1C/${file}" msk-mri-brats21-testset/BrainMSK/
done

echo "All done preparing test sets!"

# !sed -i 's/\r$//' data.sh
# !chmod +x data.sh
# !./data.sh