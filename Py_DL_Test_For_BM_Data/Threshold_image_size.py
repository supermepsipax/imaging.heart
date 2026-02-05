import os
import re
import shutil

# CHANGE THESE PATHS AND DATASET NAME
NNUNET_RAW_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//nnUNet_raw"

DATASET_NAME = "Dataset001_InfantCSpineBin"

DATASET_ROOT = os.path.join(NNUNET_RAW_ROOT, DATASET_NAME)
IMAGES_TR = os.path.join(DATASET_ROOT, "imagesTr")
LABELS_TR = os.path.join(DATASET_ROOT, "labelsTr")

IMAGES_TR_BIG = os.path.join(DATASET_ROOT, "imagesTr_big")
LABELS_TR_BIG = os.path.join(DATASET_ROOT, "labelsTr_big")

SIZE_THRESHOLD_KB = 20000  # 20 000 KB


def parse_case_id_from_image(fname):
    """
    From 'Dataset001_InfantCSpineBin_0038_0000.nii' -> '0038'
    """
    if not fname.startswith(DATASET_NAME + "_"):
        return None
    stem = os.path.splitext(fname)[0]
    m = re.match(rf"{DATASET_NAME}_(\d+)_0000$", stem)
    if not m:
        return None
    return m.group(1)


def main():
    os.makedirs(IMAGES_TR_BIG, exist_ok=True)
    os.makedirs(LABELS_TR_BIG, exist_ok=True)

    # 1) Identify cases to move (large images)
    move_ids = set()

    for fname in os.listdir(IMAGES_TR):
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            continue
        case_id = parse_case_id_from_image(fname)
        if case_id is None:
            continue
        fpath = os.path.join(IMAGES_TR, fname)
        size_kb = os.path.getsize(fpath) / 1024.0
        if size_kb > SIZE_THRESHOLD_KB:
            print(f"Marking for move (size {size_kb:.1f} KB): {fname}")
            move_ids.add(case_id)

    # 2) Move those images + corresponding labels to *_big folders
    for case_id in move_ids:
        img_fname = f"{DATASET_NAME}_{case_id}_0000.nii"
        img_path = os.path.join(IMAGES_TR, img_fname)
        img_dst = os.path.join(IMAGES_TR_BIG, img_fname)

        if os.path.exists(img_path):
            print(f"Moving image: {img_path} -> {img_dst}")
            shutil.move(img_path, img_dst)

        lbl_fname = f"{DATASET_NAME}_{case_id}.nii"
        lbl_path = os.path.join(LABELS_TR, lbl_fname)
        lbl_dst = os.path.join(LABELS_TR_BIG, lbl_fname)

        if os.path.exists(lbl_path):
            print(f"Moving label: {lbl_path} -> {lbl_dst}")
            shutil.move(lbl_path, lbl_dst)

    # 3) Renumber remaining cases consecutively
    remaining = []
    for fname in os.listdir(IMAGES_TR):
        if not fname.endswith(".nii") and not fname.endswith(".nii.gz"):
            continue
        case_id = parse_case_id_from_image(fname)
        if case_id is None:
            continue
        remaining.append((int(case_id), fname))

    if not remaining:
        print("No remaining cases to renumber.")
        return

    remaining.sort()  # sort by old numeric ID

    # Map old_id -> new_id ("0001", "0002", ...)
    id_map = {}
    for idx, (old_id_num, fname) in enumerate(remaining, start=1):
        old_id_str = f"{old_id_num:04d}"
        new_id_str = f"{idx:04d}"
        id_map[old_id_str] = new_id_str

    for old_id_str, new_id_str in id_map.items():
        old_img = os.path.join(IMAGES_TR, f"{DATASET_NAME}_{old_id_str}_0000.nii")
        new_img = os.path.join(IMAGES_TR, f"{DATASET_NAME}_{new_id_str}_0000.nii")
        if os.path.exists(old_img):
            print(f"Renaming image: {old_img} -> {new_img}")
            os.rename(old_img, new_img)

        old_lbl = os.path.join(LABELS_TR, f"{DATASET_NAME}_{old_id_str}.nii")
        new_lbl = os.path.join(LABELS_TR, f"{DATASET_NAME}_{new_id_str}.nii")
        if os.path.exists(old_lbl):
            print(f"Renaming label: {old_lbl} -> {new_lbl}")
            os.rename(old_lbl, new_lbl)

    print("Done: large cases moved to *_big and remaining IDs renumbered consecutively.")


if __name__ == "__main__":
    main()
