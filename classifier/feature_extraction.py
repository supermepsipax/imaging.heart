"""
Feature extraction module for coronary artery disease classification.

Extracts structural features from pre-processed artery analysis data
to create fixed-length feature vectors for each patient.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from analysis import (
    extract_main_branch_statistics,
    extract_all_branch_statistics,
    extract_bifurcation_statistics,
    extract_trifurcation_statistics,
    compute_branch_tapering
)


def get_feature_names() -> List[str]:
    """
    Return the list of feature names in order.

    Returns:
        List of feature column names
    """
    return [
        # =======================================================================
        # PRIMARY BRANCH DIMENSIONAL FEATURES
        # =======================================================================
        'lad_length',
        'lad_diameter',
        'lad_tortuosity',
        'lcx_length',
        'lcx_diameter',
        'lcx_tortuosity',
        'rca_length',
        'rca_diameter',
        'rca_tortuosity',

        # =======================================================================
        # TAPERING FEATURES (disease indicator - abnormal tapering)
        # =======================================================================
        'lad_taper_absolute',      # Diameter change from proximal to distal
        'lad_taper_relative',      # Relative taper (%)
        'lad_taper_rate',          # Slope of diameter change
        'lcx_taper_absolute',
        'lcx_taper_relative',
        'lcx_taper_rate',
        'rca_taper_absolute',
        'rca_taper_relative',
        'rca_taper_rate',

        # =======================================================================
        # RAMUS FEATURES (optional branch - trifurcation indicator)
        # =======================================================================
        'ramus_present',
        'ramus_length',
        'ramus_diameter',

        # =======================================================================
        # TRIFURCATION ANGLES (if present - hemodynamically important)
        # =======================================================================
        'trifurc_angle_A_main',    # Parent-LCx angle
        'trifurc_angle_B_main',    # LAD-LCx angle
        'trifurc_angle_C_main',    # Parent-LAD angle
        'trifurc_inflow_angle',    # Inflow angle at trifurcation
        'trifurc_angle_B1',        # LCx-Ramus angle
        'trifurc_angle_B2',        # LAD-Ramus angle

        # =======================================================================
        # LAD-LCx BIFURCATION (primary bifurcation - key disease site)
        # =======================================================================
        'lad_lcx_angle_A',
        'lad_lcx_angle_B',
        'lad_lcx_angle_C',
        'lad_lcx_inflow_angle',
        'lad_lcx_pmv_diameter',
        'lad_lcx_dmv_diameter',
        'lad_lcx_side_diameter',

        # Murray's law ratio: (D_child1^3 + D_child2^3) / D_parent^3 ≈ 1 for healthy
        'lad_lcx_murray_ratio',

        # =======================================================================
        # LAD-D1 BIFURCATION (first diagonal - common disease site)
        # =======================================================================
        'lad_d1_angle_A',
        'lad_d1_angle_B',
        'lad_d1_angle_C',
        'lad_d1_inflow_angle',
        'lad_d1_pmv_diameter',
        'lad_d1_dmv_diameter',
        'lad_d1_side_diameter',
        'lad_d1_murray_ratio',

        # =======================================================================
        # SIDE BRANCH AGGREGATE FEATURES
        # =======================================================================
        'diagonal_count',          # Number of diagonal branches (D1, D2, etc.)
        'diagonal_total_length',   # Total length of all diagonals
        'diagonal_mean_diameter',  # Mean diameter of diagonals
        'om_count',                # Number of obtuse marginal branches
        'om_total_length',         # Total length of all OMs
        'om_mean_diameter',        # Mean diameter of OMs

        # =======================================================================
        # GLOBAL STRUCTURAL FEATURES
        # =======================================================================
        'total_vessel_length',
        'total_branch_count',
        'lca_total_length',
        'rca_total_length',
        'lca_rca_length_ratio',
        'lca_branch_count',
        'rca_branch_count',
        'bifurcation_count',       # Total number of bifurcations

        # =======================================================================
        # DIAMETER STATISTICS
        # =======================================================================
        'mean_diameter_global',
        'diameter_std_global',
        'diameter_cv',             # Coefficient of variation (std/mean)
        'max_diameter',
        'min_diameter',
        'diameter_range',          # max - min
    ]


def extract_patient_features(
    lca_data: Optional[Dict[str, Any]],
    rca_data: Optional[Dict[str, Any]],
    diameter_method: str = 'slicing'
) -> Dict[str, float]:
    """
    Extract features for a single patient from LCA and RCA analysis data.

    Args:
        lca_data: Dictionary containing LCA analysis results with keys:
            - 'final_graph': NetworkX DiGraph
            - 'spacing_info': Voxel spacing tuple
            - 'metadata': Metadata dictionary
        rca_data: Dictionary containing RCA analysis results (same structure)
        diameter_method: 'slicing' or 'edt' for diameter measurements

    Returns:
        Dictionary of feature name -> value pairs
    """
    features = {}

    for name in get_feature_names():
        features[name] = np.nan

    all_branch_lengths = []
    all_branch_diameters = []
    lca_total_length = 0.0
    rca_total_length = 0.0
    lca_branch_count = 0
    rca_branch_count = 0
    total_bifurcation_count = 0

    # =========================================================================
    # LCA Features
    # =========================================================================
    if lca_data is not None:
        lca_graph = lca_data.get('final_graph')
        lca_spacing = lca_data.get('spacing_info')

        if lca_graph is not None and lca_spacing is not None:
            # -----------------------------------------------------------------
            # Main branch statistics
            # -----------------------------------------------------------------
            try:
                lca_main_branches = extract_main_branch_statistics(
                    lca_graph, lca_spacing, artery_type='LCA',
                    diameter_method=diameter_method
                )

                if 'LAD' in lca_main_branches:
                    lad = lca_main_branches['LAD']
                    features['lad_length'] = lad['total_path_length']
                    features['lad_diameter'] = lad['mean_diameter']
                    features['lad_tortuosity'] = lad['tortuosity']
                    lca_total_length += lad['total_path_length']
                    all_branch_lengths.append(lad['total_path_length'])
                    all_branch_diameters.append(lad['mean_diameter'])

                    if lad.get('diameter_profile') and len(lad['diameter_profile']) > 1:
                        taper = compute_branch_tapering(lad['diameter_profile'])
                        features['lad_taper_absolute'] = taper['absolute_change']
                        features['lad_taper_relative'] = taper['relative_change']
                        features['lad_taper_rate'] = taper['tapering_rate']

                if 'LCx' in lca_main_branches:
                    lcx = lca_main_branches['LCx']
                    features['lcx_length'] = lcx['total_path_length']
                    features['lcx_diameter'] = lcx['mean_diameter']
                    features['lcx_tortuosity'] = lcx['tortuosity']
                    lca_total_length += lcx['total_path_length']
                    all_branch_lengths.append(lcx['total_path_length'])
                    all_branch_diameters.append(lcx['mean_diameter'])

                    if lcx.get('diameter_profile') and len(lcx['diameter_profile']) > 1:
                        taper = compute_branch_tapering(lcx['diameter_profile'])
                        features['lcx_taper_absolute'] = taper['absolute_change']
                        features['lcx_taper_relative'] = taper['relative_change']
                        features['lcx_taper_rate'] = taper['tapering_rate']

                if 'Ramus' in lca_main_branches:
                    ramus = lca_main_branches['Ramus']
                    features['ramus_present'] = 1
                    features['ramus_length'] = ramus['total_path_length']
                    features['ramus_diameter'] = ramus['mean_diameter']
                    lca_total_length += ramus['total_path_length']
                    all_branch_lengths.append(ramus['total_path_length'])
                    all_branch_diameters.append(ramus['mean_diameter'])
                else:
                    features['ramus_present'] = 0
                    features['ramus_length'] = 0
                    features['ramus_diameter'] = 0

            except Exception as e:
                print(f"    [WARNING] Failed to extract LCA main branch stats: {e}")

            # -----------------------------------------------------------------
            # Trifurcation statistics
            # -----------------------------------------------------------------
            try:
                trifurcations = extract_trifurcation_statistics(
                    lca_graph, lca_spacing, diameter_method=diameter_method
                )

                if 'LCA_TRIFURCATION' in trifurcations:
                    trifurc = trifurcations['LCA_TRIFURCATION']

                    main_angles = trifurc.get('main_plane_angles', {})
                    features['trifurc_angle_A_main'] = main_angles.get('averaged_angle_A_main')
                    features['trifurc_angle_B_main'] = main_angles.get('averaged_angle_B_main')
                    features['trifurc_angle_C_main'] = main_angles.get('averaged_angle_C_main')
                    features['trifurc_inflow_angle'] = main_angles.get('averaged_inflow_angle')

                    add_angles = trifurc.get('additional_angles', {})
                    features['trifurc_angle_B1'] = add_angles.get('averaged_angle_B1')
                    features['trifurc_angle_B2'] = add_angles.get('averaged_angle_B2')

            except Exception as e:
                print(f"    [WARNING] Failed to extract trifurcation stats: {e}")

            # -----------------------------------------------------------------
            # Bifurcation statistics
            # -----------------------------------------------------------------
            try:
                lca_bifurcations = extract_bifurcation_statistics(
                    lca_graph, lca_spacing, diameter_method=diameter_method
                )

                total_bifurcation_count += len(lca_bifurcations)

                if 'LAD_LCx' in lca_bifurcations:
                    bif = lca_bifurcations['LAD_LCx']
                    angles = bif['angles']
                    diameters = bif['diameters']

                    features['lad_lcx_angle_A'] = angles.get('averaged_angle_A')
                    features['lad_lcx_angle_B'] = angles.get('averaged_angle_B')
                    features['lad_lcx_angle_C'] = angles.get('averaged_angle_C')
                    features['lad_lcx_inflow_angle'] = angles.get('averaged_inflow_angle')

                    features['lad_lcx_pmv_diameter'] = diameters.get('PMV')
                    features['lad_lcx_dmv_diameter'] = diameters.get('DMV')
                    features['lad_lcx_side_diameter'] = diameters.get('side_branch')

                    # Murray's law ratio
                    pmv = diameters.get('PMV')
                    dmv = diameters.get('DMV')
                    side = diameters.get('side_branch')
                    if pmv and dmv and side and pmv > 0:
                        murray = (dmv**3 + side**3) / (pmv**3)
                        features['lad_lcx_murray_ratio'] = murray

                if 'LAD_D1' in lca_bifurcations:
                    bif = lca_bifurcations['LAD_D1']
                    angles = bif['angles']
                    diameters = bif['diameters']

                    features['lad_d1_angle_A'] = angles.get('averaged_angle_A')
                    features['lad_d1_angle_B'] = angles.get('averaged_angle_B')
                    features['lad_d1_angle_C'] = angles.get('averaged_angle_C')
                    features['lad_d1_inflow_angle'] = angles.get('averaged_inflow_angle')

                    features['lad_d1_pmv_diameter'] = diameters.get('PMV')
                    features['lad_d1_dmv_diameter'] = diameters.get('DMV')
                    features['lad_d1_side_diameter'] = diameters.get('side_branch')

                    pmv = diameters.get('PMV')
                    dmv = diameters.get('DMV')
                    side = diameters.get('side_branch')
                    if pmv and dmv and side and pmv > 0:
                        murray = (dmv**3 + side**3) / (pmv**3)
                        features['lad_d1_murray_ratio'] = murray

            except Exception as e:
                print(f"    [WARNING] Failed to extract LCA bifurcation stats: {e}")

            # -----------------------------------------------------------------
            # All branches statistics (for side branch aggregates)
            # -----------------------------------------------------------------
            try:
                lca_all_branches = extract_all_branch_statistics(
                    lca_graph, lca_spacing, diameter_method=diameter_method
                )
                lca_branch_count = len(lca_all_branches)

                diagonal_lengths = []
                diagonal_diameters = []
                om_lengths = []
                om_diameters = []

                for branch in lca_all_branches:
                    label = branch.get('branch_label', '')
                    length = branch.get('length', 0)
                    diameter = branch.get('mean_diameter', 0)

                    if length > 0:
                        all_branch_lengths.append(length)
                    if diameter > 0:
                        all_branch_diameters.append(diameter)

                    if label.startswith('D') and label[1:].isdigit():
                        # Diagonal branch (D1, D2, etc.)
                        if length > 0:
                            diagonal_lengths.append(length)
                        if diameter > 0:
                            diagonal_diameters.append(diameter)
                    elif label.startswith('OM') or (label.startswith('O') and len(label) > 1):
                        # Obtuse marginal branch
                        if length > 0:
                            om_lengths.append(length)
                        if diameter > 0:
                            om_diameters.append(diameter)

                features['diagonal_count'] = len(diagonal_lengths)
                features['diagonal_total_length'] = sum(diagonal_lengths) if diagonal_lengths else 0
                features['diagonal_mean_diameter'] = np.mean(diagonal_diameters) if diagonal_diameters else 0

                features['om_count'] = len(om_lengths)
                features['om_total_length'] = sum(om_lengths) if om_lengths else 0
                features['om_mean_diameter'] = np.mean(om_diameters) if om_diameters else 0

            except Exception as e:
                print(f"    [WARNING] Failed to extract LCA all branch stats: {e}")

    # =========================================================================
    # RCA Features
    # =========================================================================
    if rca_data is not None:
        rca_graph = rca_data.get('final_graph')
        rca_spacing = rca_data.get('spacing_info')

        if rca_graph is not None and rca_spacing is not None:
            # -----------------------------------------------------------------
            # Main branch statistics
            # -----------------------------------------------------------------
            try:
                rca_main_branches = extract_main_branch_statistics(
                    rca_graph, rca_spacing, artery_type='RCA',
                    diameter_method=diameter_method
                )

                if 'RCA' in rca_main_branches:
                    rca = rca_main_branches['RCA']
                    features['rca_length'] = rca['total_path_length']
                    features['rca_diameter'] = rca['mean_diameter']
                    features['rca_tortuosity'] = rca['tortuosity']
                    rca_total_length += rca['total_path_length']
                    all_branch_lengths.append(rca['total_path_length'])
                    all_branch_diameters.append(rca['mean_diameter'])

                    if rca.get('diameter_profile') and len(rca['diameter_profile']) > 1:
                        taper = compute_branch_tapering(rca['diameter_profile'])
                        features['rca_taper_absolute'] = taper['absolute_change']
                        features['rca_taper_relative'] = taper['relative_change']
                        features['rca_taper_rate'] = taper['tapering_rate']

            except Exception as e:
                print(f"    [WARNING] Failed to extract RCA main branch stats: {e}")

            # -----------------------------------------------------------------
            # Bifurcation statistics
            # -----------------------------------------------------------------
            try:
                rca_bifurcations = extract_bifurcation_statistics(
                    rca_graph, rca_spacing, diameter_method=diameter_method
                )
                total_bifurcation_count += len(rca_bifurcations)

            except Exception as e:
                print(f"    [WARNING] Failed to extract RCA bifurcation stats: {e}")

            # -----------------------------------------------------------------
            # All branches statistics
            # -----------------------------------------------------------------
            try:
                rca_all_branches = extract_all_branch_statistics(
                    rca_graph, rca_spacing, diameter_method=diameter_method
                )
                rca_branch_count = len(rca_all_branches)

                for branch in rca_all_branches:
                    length = branch.get('length', 0)
                    diameter = branch.get('mean_diameter', 0)

                    if length > 0:
                        all_branch_lengths.append(length)
                        rca_total_length += length
                    if diameter > 0:
                        all_branch_diameters.append(diameter)

            except Exception as e:
                print(f"    [WARNING] Failed to extract RCA all branch stats: {e}")

    # =========================================================================
    # Global Features
    # =========================================================================
    features['lca_total_length'] = lca_total_length
    features['rca_total_length'] = rca_total_length
    features['lca_branch_count'] = lca_branch_count
    features['rca_branch_count'] = rca_branch_count
    features['total_branch_count'] = lca_branch_count + rca_branch_count
    features['bifurcation_count'] = total_bifurcation_count

    if all_branch_lengths:
        features['total_vessel_length'] = sum(all_branch_lengths)

    if lca_total_length > 0 and rca_total_length > 0:
        features['lca_rca_length_ratio'] = lca_total_length / rca_total_length

    if all_branch_diameters:
        features['mean_diameter_global'] = np.mean(all_branch_diameters)
        features['diameter_std_global'] = np.std(all_branch_diameters) if len(all_branch_diameters) > 1 else 0
        features['max_diameter'] = np.max(all_branch_diameters)
        features['min_diameter'] = np.min(all_branch_diameters)
        features['diameter_range'] = features['max_diameter'] - features['min_diameter']

        if features['mean_diameter_global'] > 0:
            features['diameter_cv'] = features['diameter_std_global'] / features['mean_diameter_global']

    return features


def build_feature_matrix(
    patients_data: List[Dict[str, Any]],
    diameter_method: str = 'slicing'
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Build feature matrix from list of patient data.

    Args:
        patients_data: List of patient dictionaries, each containing:
            - 'patient_id': Patient identifier string
            - 'label': Binary label (0=healthy, 1=diseased)
            - 'lca_data': LCA analysis dictionary (or None)
            - 'rca_data': RCA analysis dictionary (or None)
        diameter_method: 'slicing' or 'edt'

    Returns:
        Tuple of:
            - X: DataFrame with features (one row per patient)
            - y: numpy array of labels
            - patient_ids: List of patient ID strings
    """
    feature_dicts = []
    labels = []
    patient_ids = []

    for patient in patients_data:
        patient_id = patient.get('patient_id', 'unknown')
        label = patient.get('label')
        lca_data = patient.get('lca_data')
        rca_data = patient.get('rca_data')

        # Extract features
        features = extract_patient_features(
            lca_data, rca_data, diameter_method=diameter_method
        )

        feature_dicts.append(features)
        labels.append(label)
        patient_ids.append(patient_id)

    # Create DataFrame
    X = pd.DataFrame(feature_dicts)
    y = np.array(labels)

    return X, y, patient_ids


def impute_missing_values(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in feature matrix.

    Args:
        X: Feature DataFrame with potential NaN values
        strategy: 'median', 'mean', or 'zero'

    Returns:
        DataFrame with imputed values
    """
    X_imputed = X.copy()

    if strategy == 'median':
        X_imputed = X_imputed.fillna(X_imputed.median())
    elif strategy == 'mean':
        X_imputed = X_imputed.fillna(X_imputed.mean())
    elif strategy == 'zero':
        X_imputed = X_imputed.fillna(0)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    # If any columns are still NaN (all values were NaN), fill with 0
    X_imputed = X_imputed.fillna(0)

    return X_imputed


def select_features(
    X: pd.DataFrame,
    feature_subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select a subset of features from the feature matrix.

    Args:
        X: Full feature DataFrame
        feature_subset: List of feature names to keep (None = all features)

    Returns:
        DataFrame with selected features only
    """
    if feature_subset is None:
        return X

    # Filter to only existing columns
    available = [f for f in feature_subset if f in X.columns]
    missing = [f for f in feature_subset if f not in X.columns]

    if missing:
        print(f"    [WARNING] Requested features not found: {missing}")

    return X[available]


# =============================================================================
# FEATURE SUBSETS
# =============================================================================

# Priority features - most reliable and clinically relevant
PRIORITY_FEATURES = [
    # Main branch dimensions
    'lad_length',
    'lad_diameter',
    'lcx_length',
    'lcx_diameter',
    'rca_length',
    'rca_diameter',

    # Key bifurcation angle
    'lad_lcx_angle_B',

    # Global
    'lca_rca_length_ratio',
]

# Extended features - adds tapering and more angles
EXTENDED_FEATURES = PRIORITY_FEATURES + [
    'lad_tortuosity',
    'lcx_tortuosity',
    'rca_tortuosity',
    'lad_lcx_inflow_angle',
    'ramus_present',
    'total_vessel_length',
    'mean_diameter_global',
    'lad_taper_relative',
    'lcx_taper_relative',
    'rca_taper_relative',
    'lad_lcx_murray_ratio',
]

# Disease-focused features - emphasizing potential disease markers
DISEASE_FEATURES = [
    # Tapering (abnormal in disease)
    'lad_taper_relative',
    'lcx_taper_relative',
    'rca_taper_relative',

    # Murray's law deviation (indicates remodeling)
    'lad_lcx_murray_ratio',
    'lad_d1_murray_ratio',

    # Bifurcation angles (remodeling indicator)
    'lad_lcx_angle_B',
    'lad_d1_angle_B',

    # Diameters at key sites
    'lad_lcx_pmv_diameter',
    'lad_lcx_dmv_diameter',
    'lad_d1_pmv_diameter',

    # Tortuosity (increases with disease)
    'lad_tortuosity',
    'lcx_tortuosity',
    'rca_tortuosity',

    # Diameter variability
    'diameter_cv',
    'diameter_range',
]

ALL_FEATURES = get_feature_names()
