import os
import SimpleITK as sitk
import numpy as np


LABELS_TR = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw//Dataset001_InfantCSpineBin//labelsTr"



DATASET_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw//Dataset001_InfantCSpineBin"
print("imagesTr:", len([f for f in os.listdir(os.path.join(DATASET_ROOT, "imagesTr")) if f.endswith(".nii")]))
print("labelsTr:", len([f for f in os.listdir(os.path.join(DATASET_ROOT, "labelsTr")) if f.endswith(".nii")]))

# for fname in sorted(os.listdir(LABELS_TR)):
#     if not fname.lower().endswith(".nii"):
#         continue
#     path = os.path.join(LABELS_TR, fname)
#     img = sitk.ReadImage(path)
#     arr = sitk.GetArrayFromImage(img)
#     uniq = np.unique(arr)
#     print(f"{fname}: {uniq}")
# def fix_labels_to_binary(labels_tr):
#     for fname in os.listdir(labels_tr):
#         if not fname.lower().endswith(".nii"):
#             continue
#         path = os.path.join(labels_tr, fname)
#         img = sitk.ReadImage(path)
#         arr = sitk.GetArrayFromImage(img)

#         # Show current unique values once
#         uniq = np.unique(arr)
#         print(f"{fname}: unique values before = {uniq}")

#         # Binarize: anything > 0 becomes 1, background is 0
#         arr_bin = (arr > 0).astype(np.uint8)

#         # Check that both 0 and 1 exist
#         uniq_after = np.unique(arr_bin)
#         print(f"{fname}: unique values after  = {uniq_after}")

#         img_bin = sitk.GetImageFromArray(arr_bin)
#         img_bin.CopyInformation(img)
#         sitk.WriteImage(img_bin, path)
#         print(f"Fixed and saved {fname}\n")

# if __name__ == "__main__":
#     fix_labels_to_binary(LABELS_TR)
