import os
import SimpleITK as sitk

# INPUT ROOT: contains many subfolders, each with 2 NRRD files
INPUT_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//Maddalena-20251216T123826Z-3-001//Maddalena"

# OUTPUT ROOT: new structure will be created here
OUTPUT_ROOT = r"D://Studium//M.Sc. Medical Engineering//Lab Course//BM Child Project//Data C Project//DL_Try//Formated_for DL"

def convert_nrrd_folder_to_nifti(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    for entry in os.scandir(input_root):
        if not entry.is_dir():
            continue

        subfolder_name = entry.name
        in_subfolder = entry.path

        # Find all .nrrd files in this subfolder
        nrrd_files = [f for f in os.listdir(in_subfolder) if f.lower().endswith(".nrrd")]

        # Only process folders with exactly two NRRD files
        if len(nrrd_files) != 2:
            print(f"Skipping folder (needs exactly 2 NRRD files): {in_subfolder} (found {len(nrrd_files)})")
            continue

        print(f"Processing folder: {in_subfolder} (found 2 NRRD files)")

        out_subfolder = os.path.join(output_root, subfolder_name)
        os.makedirs(out_subfolder, exist_ok=True)

        for nrrd_filename in nrrd_files:
            in_path = os.path.join(in_subfolder, nrrd_filename)

            try:
                image = sitk.ReadImage(in_path)
            except Exception as e:
                print(f"  ERROR reading {in_path}: {e}")
                continue

            base_name = os.path.splitext(nrrd_filename)[0]
            if base_name.lower().endswith(".nrrd"):
                base_name = base_name[:-5]

            # Write as .nii (change to .nii.gz if you want compressed NIfTI)
            out_filename = base_name + ".nii"
            out_path = os.path.join(out_subfolder, out_filename)

            try:
                sitk.WriteImage(image, out_path)
                print(f"  Converted: {nrrd_filename} -> {out_filename}")
            except Exception as e:
                print(f"  ERROR writing {out_path}: {e}")

if __name__ == "__main__":
    convert_nrrd_folder_to_nifti(INPUT_ROOT, OUTPUT_ROOT)
    print("Done.")


