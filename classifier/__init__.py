"""
Classifier package for coronary artery disease classification.

This package provides tools to classify healthy vs diseased patients
based on structural features extracted from coronary artery analysis.

Modules:
    feature_extraction: Extract features from artery analysis data
    model_training: Train and evaluate classification models
    classifier_pipeline: Main pipeline orchestration

Usage:
    from classifier import run_classification_pipeline

    results = run_classification_pipeline(
        input_tar_file='results/artery_analyses.tar.gz',
        output_folder='classifier_results/',
        model_type='logistic_regression',
        feature_set='priority'
    )
"""

from .feature_extraction import (
    extract_patient_features,
    build_feature_matrix,
    impute_missing_values,
    select_features,
    get_feature_names,
    PRIORITY_FEATURES,
    EXTENDED_FEATURES,
    DISEASE_FEATURES,
    ALL_FEATURES
)

from .model_training import (
    create_model,
    evaluate_loocv,
    evaluate_repeated_cv,
    train_final_model,
    get_feature_importance,
    save_model,
    load_model,
    predict_patient,
    compare_models
)

from .classifier_pipeline import (
    load_patient_data,
    run_classification_pipeline,
    run_model_comparison
)

__all__ = [
    # Feature extraction
    'extract_patient_features',
    'build_feature_matrix',
    'impute_missing_values',
    'select_features',
    'get_feature_names',
    'PRIORITY_FEATURES',
    'EXTENDED_FEATURES',
    'ALL_FEATURES',

    # Model training
    'create_model',
    'evaluate_loocv',
    'evaluate_repeated_cv',
    'train_final_model',
    'get_feature_importance',
    'save_model',
    'load_model',
    'predict_patient',
    'compare_models',

    # Pipeline
    'load_patient_data',
    'run_classification_pipeline',
    'run_model_comparison',
]
