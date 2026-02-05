import os
import shutil
import SimpleITK as sitk

# Folder with the new data to add.
# Structure: Formated_for DL/<subject_folder>/<files>
FORMATED_FOR_DL = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//Formated_for DL"

# Existing nnU-Net dataset root
NNUNET_RAW_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw"
DATASET_NAME = "Dataset001_InfantCSpineBin"



DATASET_ROOT = os.path.join(NNUNET_RAW_ROOT, DATASET_NAME)
IMAGES_TR = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_TR = os.path.join(DATASET_ROOT, "labelsTr")


def ensure_dataset_dirs():
    os.makedirs(IMAGES_TR, exist_ok=True)
    os.makedirs(LABELS_TR, exist_ok=True)


def get_next_case_id():
    """
    Scan existing imagesTr and find the next available numeric case ID.
    Assumes files are named like Dataset001_InfantCSpineBin_XXXX_0000.nii.
    """
    existing_ids = set()

    for fname in os.listdir(IMAGES_TR):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        if not fname.startswith(DATASET_NAME + "_"):
            continue
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        case_id = parts[-2]  # the XXXX part
        if case_id.isdigit():
            existing_ids.add(int(case_id))

    if not existing_ids:
        return 1
    return max(existing_ids) + 1


def add_formated_nifti_to_dataset():
    ensure_dataset_dirs()

    case_counter = get_next_case_id()
    print(f"Starting from case ID: {case_counter:04d}")

    for entry in os.scandir(FORMATED_FOR_DL):
        if not entry.is_dir():
            continue

        subject_folder = entry.path
        print(f"Processing folder: {subject_folder}")

        nii_files = [
            f for f in os.listdir(subject_folder)
            if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
        ]

        if len(nii_files) < 2:
            print(f"  Skipping (needs at least 2 NIfTI files, found {len(nii_files)})")
            continue

        nii_files.sort()

        image_name = nii_files[0]  # first = image
        label_name = nii_files[1]  # second = seg (.seg.nii or .nii)

        image_src = os.path.join(subject_folder, image_name)
        label_src = os.path.join(subject_folder, label_name)

        case_id = f"{case_counter:04d}"

        image_filename = f"{DATASET_NAME}_{case_id}_0000.nii"
        label_filename = f"{DATASET_NAME}_{case_id}.nii"

        image_dst = os.path.join(IMAGES_TR, image_filename)
        label_dst = os.path.join(LABELS_TR, label_filename)

        # If either target already exists, skip this subject to avoid overwriting
        if os.path.exists(image_dst) or os.path.exists(label_dst):
            print(f"  Skipping (target already exists): {image_dst} or {label_dst}")
            continue

        # Now we know this ID is unused -> increment for next subject
        case_counter += 1

        # Copy image as-is
        try:
            print(f"  Copying image: {image_src} -> {image_dst}")
            shutil.copy2(image_src, image_dst)
        except Exception as e:
            print(f"  ERROR copying image for {subject_folder}: {e}")
            continue

        # Convert label into .nii in labelsTr
        try:
            print(f"  Reading label: {label_src}")
            seg_img = sitk.ReadImage(label_src)
            sitk.WriteImage(seg_img, label_dst)
            print(f"  Wrote label: {label_dst}")
        except Exception as e:
            print(f"  ERROR reading/writing label for {subject_folder}: {e}")
            # Optional: remove image_dst if label write fails
            try:
                if os.path.exists(image_dst):
                    os.remove(image_dst)
            except OSError:
                pass
            continue

    print("Finished adding all new cases.")


if __name__ == "__main__":
    add_formated_nifti_to_dataset()


# import os
# import SimpleITK as sitk
# import shutil
# # Folder with the new data to add.
# # Structure: Formated_for DL/<subject_folder>/<two .nrrd files>
# FORMATED_FOR_DL = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//Formated_for DL"

# # Existing nnU-Net dataset root
# NNUNET_RAW_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw"
# DATASET_NAME = "Dataset001_InfantCSpineBin"

# DATASET_ROOT = os.path.join(NNUNET_RAW_ROOT, DATASET_NAME)
# IMAGES_TR = os.path.join(DATASET_ROOT, "imagesTr")
# LABELS_TR = os.path.join(DATASET_ROOT, "labelsTr")


# def ensure_dataset_dirs():
#     os.makedirs(IMAGES_TR, exist_ok=True)
#     os.makedirs(LABELS_TR, exist_ok=True)


# def get_next_case_id():
#     """
#     Scan existing imagesTr / labelsTr and find the next available numeric case ID.
#     Assumes files are named like Dataset001_InfantCSpineBin_XXXX_0000.nii.
#     """
#     existing_ids = set()

#     for fname in os.listdir(IMAGES_TR):
#         if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
#             continue
#         if not fname.startswith(DATASET_NAME + "_"):
#             continue
#         stem = os.path.splitext(fname)[0]
#         parts = stem.split("_")
#         if len(parts) < 3:
#             continue
#         case_id = parts[-2]  # the XXXX part
#         if case_id.isdigit():
#             existing_ids.add(int(case_id))

#     if not existing_ids:
#         return 1
#     return max(existing_ids) + 1


# def add_formated_nifti_to_dataset():
#     ensure_dataset_dirs()

#     case_counter = get_next_case_id()
#     print(f"Starting from case ID: {case_counter:04d}")

#     for entry in os.scandir(FORMATED_FOR_DL):
#         if not entry.is_dir():
#             continue

#         subject_folder = entry.path
#         print(f"Processing folder: {subject_folder}")

#         nii_files = [f for f in os.listdir(subject_folder)
#                     if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")]
#         if len(nii_files) != 2:
#             print(f"  Skipping (needs exactly 2 NIfTI files, found {len(nii_files)})")
#             continue

#         nii_files.sort()
#         image_src = os.path.join(subject_folder, nii_files[0])  # first = image
#         label_src = os.path.join(subject_folder, nii_files[1])  # second = label

#         case_id = f"{case_counter:04d}"
#         case_counter += 1

#         image_filename = f"{DATASET_NAME}_{case_id}_0000.nii"
#         label_filename = f"{DATASET_NAME}_{case_id}.nii"

#         image_dst = os.path.join(IMAGES_TR, image_filename)
#         label_dst = os.path.join(LABELS_TR, label_filename)

#         try:
#             print(f"  Copying image: {image_src} -> {image_dst}")
#             shutil.copy2(image_src, image_dst)
#         except Exception as e:
#             print(f"  ERROR copying image for {subject_folder}: {e}")
#             continue

#         try:
#             print(f"  Copying label: {label_src} -> {label_dst}")
#             shutil.copy2(label_src, label_dst)
#         except Exception as e:
#             print(f"  ERROR copying label for {subject_folder}: {e}")
#             continue

#     print("Finished adding all new cases.")


# if __name__ == "__main__":
#     add_formated_nifti_to_dataset()