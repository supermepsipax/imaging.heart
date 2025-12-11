"""
Main pipeline for coronary artery disease classification.

Orchestrates data loading, feature extraction, model training, and evaluation.
"""

import os
import io
import pickle
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from utilities import load_artery_analysis, load_config

from .feature_extraction import (
    extract_patient_features,
    build_feature_matrix,
    impute_missing_values,
    select_features,
    get_feature_names,
    PRIORITY_FEATURES,
    EXTENDED_FEATURES,
    DISEASE_FEATURES
)

from .model_training import (
    create_model,
    evaluate_loocv,
    evaluate_repeated_cv,
    train_final_model,
    get_feature_importance,
    save_model,
    compare_models
)


def load_patient_data(
    input_folder: Optional[str] = None,
    input_tar_file: Optional[str] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Load artery analysis data and organize by patient.

    Each patient has LCA and RCA data. Labels are determined from filename
    (contains 'Normal' = healthy, otherwise = diseased).

    Args:
        input_folder: Path to folder containing .pkl analysis files
        input_tar_file: Path to tar archive containing .pkl files
        verbose: Print loading progress

    Returns:
        List of patient dictionaries with keys:
            - patient_id: Patient identifier
            - label: 0 (healthy) or 1 (diseased)
            - lca_data: LCA analysis dict or None
            - rca_data: RCA analysis dict or None
    """
    if input_folder is None and input_tar_file is None:
        raise ValueError("Must provide either input_folder or input_tar_file")

    if input_folder is not None and input_tar_file is not None:
        raise ValueError("Cannot specify both input_folder and input_tar_file")

    # Collect all analysis files
    analysis_files = {}  # {identifier: filepath or TarInfo}

    if input_tar_file is not None:
        if verbose:
            print(f"Loading from tar archive: {input_tar_file}")

        with tarfile.open(input_tar_file, 'r:*') as tar:
            pkl_members = [m for m in tar.getmembers()
                          if m.name.endswith('.pkl') and m.isfile()]
            for m in pkl_members:
                identifier = Path(m.name).stem
                analysis_files[identifier] = ('tar', m)

    else:
        if verbose:
            print(f"Loading from folder: {input_folder}")

        input_path = Path(input_folder)
        pkl_files = list(input_path.glob('*_analysis.pkl'))
        for f in pkl_files:
            identifier = f.stem
            analysis_files[identifier] = ('file', str(f))

    if verbose:
        print(f"Found {len(analysis_files)} analysis files")

    # Group files by patient
    # Expected naming: PatientID_LCA_analysis.pkl, PatientID_RCA_analysis.pkl
    # Or: PatientID_artery_1_analysis.pkl, PatientID_artery_2_analysis.pkl
    patients_data = defaultdict(lambda: {
        'patient_id': None,
        'label': None,
        'lca_data': None,
        'rca_data': None
    })

    # Open tar file if needed (keep open for all reads)
    tar_file = None
    if input_tar_file is not None:
        tar_file = tarfile.open(input_tar_file, 'r:*')

    try:
        for identifier, (source_type, source) in analysis_files.items():
            # Load the data
            if source_type == 'tar':
                file_obj = tar_file.extractfile(source)
                if file_obj is None:
                    continue
                data = pickle.load(file_obj)
            else:
                data = load_artery_analysis(source)

            # Extract patient ID and artery type
            metadata = data.get('metadata', {})
            artery_type = metadata.get('artery', 'unknown').upper()
            file_basename = metadata.get('file_basename', identifier)

            # Determine patient ID (remove artery suffix)
            # Handle various naming patterns
            patient_id = file_basename
            for suffix in ['_LCA', '_RCA', '_lca', '_rca', '_artery_1', '_artery_2']:
                if patient_id.endswith(suffix):
                    patient_id = patient_id[:-len(suffix)]
                    break

            # Determine label from filename
            if 'Normal' in file_basename or 'normal' in file_basename:
                label = 0  # Healthy
            else:
                label = 1  # Diseased

            # Store data
            patients_data[patient_id]['patient_id'] = patient_id
            patients_data[patient_id]['label'] = label

            if artery_type == 'LCA':
                patients_data[patient_id]['lca_data'] = data
            elif artery_type == 'RCA':
                patients_data[patient_id]['rca_data'] = data

    finally:
        if tar_file is not None:
            tar_file.close()

    # Convert to list and filter out incomplete patients
    patients_list = []
    for patient_id, pdata in patients_data.items():
        # Require at least one artery
        if pdata['lca_data'] is not None or pdata['rca_data'] is not None:
            patients_list.append(pdata)

    if verbose:
        print(f"Organized into {len(patients_list)} patients")

        # Count labels
        n_healthy = sum(1 for p in patients_list if p['label'] == 0)
        n_diseased = sum(1 for p in patients_list if p['label'] == 1)
        print(f"  Healthy: {n_healthy}, Diseased: {n_diseased}")

        # Count complete patients (both LCA and RCA)
        n_complete = sum(1 for p in patients_list
                        if p['lca_data'] is not None and p['rca_data'] is not None)
        print(f"  Complete (LCA + RCA): {n_complete}")

    return patients_list


def run_classification_pipeline(
    input_folder: Optional[str] = None,
    input_tar_file: Optional[str] = None,
    output_folder: Optional[str] = None,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    model_type: str = 'logistic_regression',
    feature_set: str = 'priority',
    diameter_method: str = 'slicing',
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete classification pipeline.

    Args:
        input_folder: Path to folder with analysis .pkl files
        input_tar_file: Path to tar archive with analysis files
        output_folder: Path to save results (default: creates 'classifier_results')
        config: Configuration dictionary
        config_path: Path to config file
        model_type: 'logistic_regression', 'random_forest', or 'svm'
        feature_set: 'priority' (8 features), 'extended' (12 features), or 'all'
        diameter_method: 'slicing' or 'edt'
        save_results: Whether to save model and results
        verbose: Print detailed output

    Returns:
        Dictionary with pipeline results:
            - model: Trained model
            - evaluation: LOOCV evaluation results
            - feature_importance: Feature importance series
            - X: Feature matrix
            - y: Labels
            - patient_ids: Patient identifiers
            - model_path: Path to saved model (if save_results=True)
    """
    # Load config if provided
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        config = {}

    # Get parameters from config or use defaults
    if input_folder is None:
        input_folder = config.get('input_folder')
    if input_tar_file is None:
        input_tar_file = config.get('input_tar_file')
    if output_folder is None:
        output_folder = config.get('output_folder', 'classifier_results')

    # Read model/feature parameters from config (use function defaults if not in config)
    model_type = config.get('model_type', model_type)
    feature_set = config.get('feature_set', feature_set)
    diameter_method = config.get('diameter_method', diameter_method)
    save_results = config.get('save_results', save_results)
    verbose = config.get('verbose', verbose)

    # Print header
    if verbose:
        print("=" * 80)
        print("CORONARY ARTERY DISEASE CLASSIFIER")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Model type:      {model_type}")
        print(f"  Feature set:     {feature_set}")
        print(f"  Diameter method: {diameter_method}")
        if output_folder:
            print(f"  Output folder:   {output_folder}")
        print("=" * 80)

    # =========================================================================
    # Step 1: Load patient data
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading patient data...")

    patients_data = load_patient_data(
        input_folder=input_folder,
        input_tar_file=input_tar_file,
        verbose=verbose
    )

    if len(patients_data) == 0:
        raise ValueError("No patient data found")

    # =========================================================================
    # Step 2: Extract features
    # =========================================================================
    if verbose:
        print("\n[Step 2] Extracting features...")

    X, y, patient_ids = build_feature_matrix(
        patients_data,
        diameter_method=diameter_method
    )

    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Labels: {len(y)} ({sum(y == 0)} healthy, {sum(y == 1)} diseased)")

    # =========================================================================
    # Step 3: Select features
    # =========================================================================
    if verbose:
        print("\n[Step 3] Selecting features...")

    if feature_set == 'priority':
        feature_subset = PRIORITY_FEATURES
    elif feature_set == 'extended':
        feature_subset = EXTENDED_FEATURES
    elif feature_set == 'disease':
        feature_subset = DISEASE_FEATURES
    elif feature_set == 'all':
        feature_subset = None  # Use all features
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X_selected = select_features(X, feature_subset)

    if verbose:
        print(f"  Selected {X_selected.shape[1]} features")
        if feature_subset:
            print(f"  Features: {list(X_selected.columns)}")

    # =========================================================================
    # Step 4: Impute missing values
    # =========================================================================
    if verbose:
        print("\n[Step 4] Handling missing values...")

    n_missing = X_selected.isna().sum().sum()
    if verbose:
        print(f"  Missing values: {n_missing}")

    X_imputed = impute_missing_values(X_selected, strategy='median')

    # =========================================================================
    # Step 5: Evaluate with LOOCV
    # =========================================================================
    if verbose:
        print("\n[Step 5] Evaluating model with LOOCV...")

    model = create_model(model_type)
    evaluation = evaluate_loocv(model, X_imputed, y, verbose=verbose)

    # =========================================================================
    # Step 6: Train final model
    # =========================================================================
    if verbose:
        print("\n[Step 6] Training final model on all data...")

    final_model = train_final_model(
        X_imputed, y, model_type=model_type, verbose=verbose
    )

    # Get feature importance
    feature_importance = get_feature_importance(
        final_model,
        X_imputed.columns.tolist()
    )

    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    model_path = None
    if save_results and output_folder:
        if verbose:
            print("\n[Step 7] Saving results...")

        os.makedirs(output_folder, exist_ok=True)

        # Save model
        metadata = {
            'model_type': model_type,
            'feature_set': feature_set,
            'feature_names': X_imputed.columns.tolist(),
            'n_samples': len(y),
            'n_healthy': int(sum(y == 0)),
            'n_diseased': int(sum(y == 1)),
            'evaluation': {
                'accuracy': evaluation['accuracy'],
                'auc_roc': evaluation['auc_roc'],
                'sensitivity': evaluation['sensitivity'],
                'specificity': evaluation['specificity']
            }
        }

        model_path = save_model(
            final_model,
            output_folder,
            feature_names=X_imputed.columns.tolist(),
            metadata=metadata
        )

        # Save feature importance
        importance_path = Path(output_folder) / 'feature_importance.csv'
        feature_importance.to_csv(importance_path)
        if verbose:
            print(f"  Feature importance saved to: {importance_path}")

        # Save predictions
        predictions_df = pd.DataFrame({
            'patient_id': patient_ids,
            'true_label': y,
            'predicted_label': evaluation['predictions'],
            'probability_diseased': evaluation['probabilities'] if evaluation['probabilities'] is not None else np.nan
        })
        predictions_path = Path(output_folder) / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        if verbose:
            print(f"  Predictions saved to: {predictions_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("CLASSIFICATION COMPLETE")
        print("=" * 80)
        print(f"\n  Model: {model_type}")
        print(f"  Features: {X_imputed.shape[1]}")
        print(f"  Samples: {len(y)} ({sum(y == 0)} healthy, {sum(y == 1)} diseased)")
        print(f"\n  LOOCV Performance:")
        print(f"    Accuracy:    {evaluation['accuracy']:.3f}")
        if evaluation['auc_roc'] is not None:
            print(f"    AUC-ROC:     {evaluation['auc_roc']:.3f}")
        print(f"    Sensitivity: {evaluation['sensitivity']:.3f}")
        print(f"    Specificity: {evaluation['specificity']:.3f}")
        print(f"\n  Top 5 Important Features:")
        for feat, imp in feature_importance.head(5).items():
            print(f"    {feat}: {imp:+.3f}")
        print("=" * 80)

    return {
        'model': final_model,
        'evaluation': evaluation,
        'feature_importance': feature_importance,
        'X': X_imputed,
        'y': y,
        'patient_ids': patient_ids,
        'model_path': model_path
    }


def run_model_comparison(
    input_folder: Optional[str] = None,
    input_tar_file: Optional[str] = None,
    feature_set: str = 'priority',
    diameter_method: str = 'slicing',
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple model types on the dataset.

    Args:
        input_folder: Path to folder with analysis files
        input_tar_file: Path to tar archive
        feature_set: 'priority', 'extended', or 'all'
        diameter_method: 'slicing' or 'edt'
        verbose: Print comparison results

    Returns:
        Dictionary with results for each model type
    """
    if verbose:
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

    # Load and prepare data
    patients_data = load_patient_data(
        input_folder=input_folder,
        input_tar_file=input_tar_file,
        verbose=verbose
    )

    X, y, _ = build_feature_matrix(patients_data, diameter_method=diameter_method)

    if feature_set == 'priority':
        X = select_features(X, PRIORITY_FEATURES)
    elif feature_set == 'extended':
        X = select_features(X, EXTENDED_FEATURES)
    elif feature_set == 'disease':
        X = select_features(X, DISEASE_FEATURES)
    # 'all' uses all features (no selection)

    X = impute_missing_values(X, strategy='median')

    # Compare models
    results = compare_models(X, y, verbose=verbose)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier_pipeline.py <input_tar_file_or_folder>")
        print("       python classifier_pipeline.py <input> --compare")
        sys.exit(1)

    input_path = sys.argv[1]
    do_compare = '--compare' in sys.argv

    # Determine input type
    if input_path.endswith('.tar.gz') or input_path.endswith('.tar'):
        input_kwargs = {'input_tar_file': input_path}
    else:
        input_kwargs = {'input_folder': input_path}

    if do_compare:
        # Run model comparison
        results = run_model_comparison(**input_kwargs, verbose=True)
    else:
        # Run full pipeline
        results = run_classification_pipeline(
            **input_kwargs,
            model_type='logistic_regression',
            feature_set='priority',
            output_folder='classifier_results',
            verbose=True
        )
