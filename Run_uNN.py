import nnunetv2
print(nnunetv2.__file__)
import numpy as np
print(np.__file__)
print(np.__version__)
$env:nnUNet_raw="D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw"
$env:nnUNet_preprocessed="D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_preprocessed"
$env:nnUNet_results="D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_results"

nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres  