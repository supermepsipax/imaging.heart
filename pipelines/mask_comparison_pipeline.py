import os
import itertools
import random
import numpy as np
from pathlib import Path
from segmentation import compare_masks
from utilities import ensure_continuous_body, load_config, load_nrrd_mask, sort_labelled_bodies_by_size
from utilities.input_output import strip_filenames
from visualizations import visualize_3d_graph, visualize_mask_overlap


def compare_mask_batch(mask_folders = None, output_folder = None, config=None, config_path=None, visualize=None):
    """
    Compare segmentation masks across multiple folders and report mean overlap scores.

    For each folder, masks are keyed by a stripped filename and then matched across
    folders using those keys. For each unique folder pair, the function loads each
    matched mask pair, combines the two largest continuous bodies into a single binary
    mask for both inputs, and computes the Dice score. The mean Dice score per folder
    pair is returned in a triangular (unique pairwise) comparison dict.

    Args:
        input_folder (str, optional): Path to folder containing .nrrd mask files (can be set in config)
        output_folder (str, optional): Path to folder where CSV results will be saved (can be set in config)
        config (dict, optional): Configuration dictionary with pipeline parameters
        config_path (str, optional): Path to JSON config file (if config not provided directly)
        visualize (bool, optional): Whether to create 3D visualizations for each artery (can be set in config)

    Returns:
        dict: Mapping of "label_a<->label_b" to mean Dice score across matched files

    Raises:
        ValueError: If input_folder or output_folder are not provided either as parameters or in config
    """
    if config is None and config_path is not None:
        config = load_config(config_path)

    if config is None:
        config = {}

    if mask_folders is None:
        mask_folders = config.get('mask_folders')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if visualize is None:
        visualize = config.get('visualize', False)
    visualize_overlap = config.get('visualize_overlap', False)
    visualize_overlap_percent = config.get('visualize_overlap_percent', 0)

    if mask_folders is None:
        raise ValueError("mask_folders must be provided either as a parameter or in the config file")
    if output_folder is None:
        raise ValueError("output_folder must be provided either as a parameter or in the config file")

    os.makedirs(output_folder, exist_ok=True)

    max_nrrd_files = 0
    for label, path in mask_folders.items():
        nrrd_files = list(Path(path).glob('*.nrrd'))
        stripped_nrrd_files = strip_filenames(nrrd_files)

        file_map = {}
        for original_path, stripped_path in zip(nrrd_files, stripped_nrrd_files):
            key = Path(stripped_path).name
            file_map[key] = str(original_path)

        if len(file_map) > max_nrrd_files:
            max_nrrd_files = len(file_map)

        mask_folders[label] = file_map
    output_path = Path(output_folder)

    comparison_matrix = {}
    labels = list(mask_folders.keys())
    for label_a, label_b in itertools.combinations(labels, 2):
        files_a = mask_folders[label_a]
        files_b = mask_folders[label_b]

        common_keys = set(files_a.keys()) & set(files_b.keys())
        visualize_keys = set()
        if visualize_overlap and common_keys:
            percent = float(visualize_overlap_percent)
            percent = max(0.0, min(100.0, percent))
            if percent > 0:
                sample_count = int(round(len(common_keys) * (percent / 100.0)))
                sample_count = max(1, min(len(common_keys), sample_count))
                visualize_keys = set(random.sample(list(common_keys), sample_count))
        dice_score_array = []
        for key in common_keys:
            mask_1, _ = load_nrrd_mask(files_a[key])
            is_continous, labelled_bodies = ensure_continuous_body(mask_1)
            sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
            if len(sorted_bodies) > 1:
                mask_1 = np.logical_or(sorted_bodies[0], sorted_bodies[1]).astype(np.uint8)
            elif len(sorted_bodies) == 1:
                mask_1 = sorted_bodies[0].astype(np.uint8)

            mask_2, _ = load_nrrd_mask(files_b[key])
            is_continous, labelled_bodies = ensure_continuous_body(mask_2)
            sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
            if len(sorted_bodies) > 1:
                mask_2 = np.logical_or(sorted_bodies[0], sorted_bodies[1]).astype(np.uint8)
            elif len(sorted_bodies) == 1:
                mask_2 = sorted_bodies[0].astype(np.uint8)

            dice_score_array.append(compare_masks(mask_1, mask_2))

            if visualize_overlap and key in visualize_keys:
                title = f"Overlap: {label_a} vs {label_b} ({key})"
                visualize_mask_overlap(mask_1, mask_2, title=title, label_1=label_a, label_2=label_b)

        if dice_score_array:
            mean_dice = float(np.mean(dice_score_array))
        else:
            mean_dice = np.nan
        comparison_matrix[f"{label_a}<->{label_b}"] = mean_dice
    print(comparison_matrix)



    
