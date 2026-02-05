import os
import json
import SimpleITK as sitk
import numpy as np

# ROOT with many subject folders; each subject folder contains:
# - one segmentation NRRD  (to labelsTr/)
# - one .mrk.json          (to imagesTr/ as a dummy image)
# INPUT_ROOT = r"D:\Studium\M.Sc. Medical Engineering\Lab Course\BM Child Project\Data C Project\Valeria-20251216T112143Z-3-001\Valeria"

# # nnU-Net raw root
# NNUNET_RAW_ROOT = r"D:\Studium\M.Sc. Medical Engineering\Lab Course\BM Child Project\Data C Project\Valeria-20251216T112143Z-3-001\nnUNet_raw"


# INPUT ROOT: contains many subfolders, each with 2 NRRD files
INPUT_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//Sheimaa-20251216T121105Z-3-001//Sheimaa"

# OUTPUT ROOT: new structure will be created here
NNUNET_RAW_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw"


# Dataset name/ID
DATASET_NAME = "Dataset001_InfantCSpineBin"

# Output dataset folders
DATASET_ROOT = os.path.join(NNUNET_RAW_ROOT, DATASET_NAME)
IMAGES_TR = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_TR = os.path.join(DATASET_ROOT, "labelsTr")


def ensure_dirs():
    os.makedirs(IMAGES_TR, exist_ok=True)
    os.makedirs(LABELS_TR, exist_ok=True)


def find_files_in_subject_folder(folder_path):
    """Return (seg_nrrd, mrk_json) or (None, None) if not found."""
    files = os.listdir(folder_path)

    seg_files = [f for f in files if f.lower().endswith(".nrrd")]
    mrk_files = [f for f in files if f.lower().endswith(".mrk.json")]

    if len(seg_files) != 1:
        print(f"  Skipping {folder_path}: need exactly 1 NRRD, found {len(seg_files)}")
        return None, None
    if len(mrk_files) != 1:
        print(f"  Skipping {folder_path}: need exactly 1 .mrk.json, found {len(mrk_files)}")
        return None, None

    seg_nrrd = os.path.join(folder_path, seg_files[0])
    mrk_json = os.path.join(folder_path, mrk_files[0])

    return seg_nrrd, mrk_json


def create_dummy_image_from_label(label_image):
    """Create a dummy CT image (zeros) with same geometry as the label."""
    arr = np.zeros(sitk.GetArrayFromImage(label_image).shape, dtype=np.float32)
    dummy = sitk.GetImageFromArray(arr)
    dummy.CopyInformation(label_image)
    return dummy


def convert_subjects_to_nnunet(input_root):
    ensure_dirs()
    case_counter = 1

    for entry in os.scandir(input_root):
        if not entry.is_dir():
            continue

        subject_folder = entry.path
        print(f"Checking folder: {subject_folder}")

        seg_nrrd, mrk_json = find_files_in_subject_folder(subject_folder)
        if seg_nrrd is None or mrk_json is None:
            continue

        # ID: 0001, 0002, ...
        case_id = f"{case_counter:04d}"
        case_counter += 1

        # nnU-Net filenames
        image_filename = f"{DATASET_NAME}_{case_id}_0000.nii"
        label_filename = f"{DATASET_NAME}_{case_id}.nii"

        image_out_path = os.path.join(IMAGES_TR, image_filename)
        label_out_path = os.path.join(LABELS_TR, label_filename)

        # Write label (segmentation) to labelsTr
        try:
            print(f"  Reading segmentation: {seg_nrrd}")
            seg_img = sitk.ReadImage(seg_nrrd)
            sitk.WriteImage(seg_img, label_out_path)
            print(f"  Wrote label: {label_out_path}")
        except Exception as e:
            print(f"  ERROR reading/writing segmentation for {subject_folder}: {e}")
            continue

        # Create dummy image (zeros) with same geometry for imagesTr
        try:
            dummy_img = create_dummy_image_from_label(seg_img)
            sitk.WriteImage(dummy_img, image_out_path)
            print(f"  Wrote dummy image (from geometry, mrk: {mrk_json}): {image_out_path}")
        except Exception as e:
            print(f"  ERROR creating/writing image for {subject_folder}: {e}")
            continue

    print("Done converting all subjects.")


if __name__ == "__main__":
    convert_subjects_to_nnunet(INPUT_ROOT)