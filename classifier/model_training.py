"""
Model training and evaluation module for coronary artery disease classification.

Implements logistic regression with cross-validation optimized for small datasets.
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneOut,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def create_model(model_type: str = 'logistic_regression') -> Pipeline:
    """
    Create a classification model with appropriate preprocessing.

    All models are configured with strong regularization for small datasets.

    Args:
        model_type: One of 'logistic_regression', 'random_forest', 'svm'

    Returns:
        sklearn Pipeline with scaler (if needed) and classifier
    """
    if model_type == 'logistic_regression':
        # Logistic Regression with L2 regularization
        # C=0.1 provides strong regularization for small datasets
        # l1_ratio=0 is equivalent to L2 penalty (new sklearn API)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=0.1,
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])

    elif model_type == 'random_forest':
        # Random Forest with shallow trees to prevent overfitting
        return Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ))
        ])

    elif model_type == 'svm':
        # SVM with RBF kernel and strong regularization
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=0.1,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])

    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Expected 'logistic_regression', 'random_forest', or 'svm'")


def evaluate_loocv(
    model: Pipeline,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model using Leave-One-Out Cross-Validation.

    LOOCV is recommended for very small datasets (~40 samples) as it:
    - Uses maximum data for training in each fold
    - Provides unbiased estimate of generalization
    - Every sample gets to be the test set exactly once

    Args:
        model: sklearn Pipeline or estimator
        X: Feature matrix
        y: Labels (0 or 1)
        verbose: Print detailed results

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - auc_roc: Area under ROC curve
            - sensitivity: True positive rate (recall for positive class)
            - specificity: True negative rate
            - confusion_matrix: 2x2 confusion matrix
            - predictions: Array of predictions for each sample
            - probabilities: Array of predicted probabilities
            - classification_report: Detailed classification metrics
    """
    loo = LeaveOneOut()

    # Get predictions for each held-out sample
    y_pred = cross_val_predict(model, X, y, cv=loo)

    # Get probability predictions
    try:
        y_prob = cross_val_predict(model, X, y, cv=loo, method='predict_proba')[:, 1]
        auc = roc_auc_score(y, y_prob)
    except Exception:
        y_prob = None
        auc = None

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results = {
        'accuracy': acc,
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_prob,
        'classification_report': classification_report(y, y_pred, output_dict=True),
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

    if verbose:
        print(f"\n  LOOCV Results:")
        print(f"    Accuracy:    {acc:.3f}")
        if auc is not None:
            print(f"    AUC-ROC:     {auc:.3f}")
        print(f"    Sensitivity: {sensitivity:.3f} (true positive rate)")
        print(f"    Specificity: {specificity:.3f} (true negative rate)")
        print(f"\n    Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Healthy  Diseased")
        print(f"    Actual Healthy    {tn:3d}      {fp:3d}")
        print(f"           Diseased   {fn:3d}      {tp:3d}")

    return results


def evaluate_repeated_cv(
    model: Pipeline,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model using Repeated Stratified K-Fold Cross-Validation.

    Provides more stable estimates than single CV run by averaging
    across multiple random splits.

    Args:
        model: sklearn Pipeline or estimator
        X: Feature matrix
        y: Labels
        n_splits: Number of folds per repeat
        n_repeats: Number of times to repeat CV
        verbose: Print detailed results

    Returns:
        Dictionary with mean and std of metrics across all folds
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )

    # Get scores across all folds and repeats
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    results = {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'auc_scores': auc_scores,
        'accuracy_mean': np.mean(acc_scores),
        'accuracy_std': np.std(acc_scores),
        'accuracy_scores': acc_scores,
        'n_splits': n_splits,
        'n_repeats': n_repeats
    }

    if verbose:
        print(f"\n  Repeated {n_splits}-Fold CV ({n_repeats} repeats):")
        print(f"    AUC-ROC:  {results['auc_mean']:.3f} (+/- {results['auc_std']:.3f})")
        print(f"    Accuracy: {results['accuracy_mean']:.3f} (+/- {results['accuracy_std']:.3f})")

    return results


def train_final_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    model_type: str = 'logistic_regression',
    verbose: bool = True
) -> Pipeline:
    """
    Train final model on all available data.

    This should be called after validation to produce the deployment model.

    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model to train
        verbose: Print training summary

    Returns:
        Trained sklearn Pipeline
    """
    model = create_model(model_type)
    model.fit(X, y)

    if verbose:
        print(f"\n  Final model trained on {len(y)} samples")
        print(f"    Model type: {model_type}")
        print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        if model_type == 'logistic_regression':
            clf = model.named_steps['clf']
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
                importance = pd.Series(
                    clf.coef_[0],
                    index=feature_names
                ).sort_values(key=abs, ascending=False)
                print(f"\n    Top feature coefficients (absolute):")
                for feat, coef in importance.head(5).items():
                    print(f"      {feat}: {coef:+.3f}")

        elif model_type == 'random_forest':
            clf = model.named_steps['clf']
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
                importance = pd.Series(
                    clf.feature_importances_,
                    index=feature_names
                ).sort_values(ascending=False)
                print(f"\n    Top feature importances:")
                for feat, imp in importance.head(5).items():
                    print(f"      {feat}: {imp:.3f}")

    return model


def get_feature_importance(
    model: Pipeline,
    feature_names: List[str]
) -> pd.Series:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained sklearn Pipeline
        feature_names: List of feature names

    Returns:
        Series with feature importances sorted by absolute value
    """
    clf = model.named_steps.get('clf')

    if clf is None:
        raise ValueError("Could not find classifier in pipeline")

    if hasattr(clf, 'coef_'):
        importance = pd.Series(clf.coef_[0], index=feature_names)
        importance = importance.reindex(importance.abs().sort_values(ascending=False).index)

    elif hasattr(clf, 'feature_importances_'):
        importance = pd.Series(clf.feature_importances_, index=feature_names)
        importance = importance.sort_values(ascending=False)

    else:
        raise ValueError(f"Model type {type(clf)} does not support feature importance")

    return importance


def save_model(
    model: Pipeline,
    save_path: str,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save trained model to disk.

    Args:
        model: Trained sklearn Pipeline
        save_path: Directory to save model
        feature_names: List of feature names (saved alongside model)
        metadata: Additional metadata to save

    Returns:
        Path to saved model file
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_path = save_dir / f'coronary_classifier_{timestamp}.joblib'
    joblib.dump(model, model_path)

    # Save feature names if provided
    if feature_names is not None:
        feature_path = save_dir / f'feature_names_{timestamp}.txt'
        with open(feature_path, 'w') as f:
            for feat in feature_names:
                f.write(f"{feat}\n")

    # Save metadata if provided
    if metadata is not None:
        metadata_path = save_dir / f'metadata_{timestamp}.joblib'
        joblib.dump(metadata, metadata_path)

    print(f"  Model saved to: {model_path}")
    return str(model_path)


def load_model(model_path: str) -> Pipeline:
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model file

    Returns:
        Trained sklearn Pipeline
    """
    return joblib.load(model_path)


def predict_patient(
    model: Pipeline,
    features: Dict[str, float],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Make prediction for a single patient.

    Args:
        model: Trained sklearn Pipeline
        features: Dictionary of feature values
        feature_names: List of feature names in correct order

    Returns:
        Dictionary with prediction and probability
    """
    # Build feature vector in correct order
    X = pd.DataFrame([{name: features.get(name, np.nan) for name in feature_names}])

    # Handle missing values
    X = X.fillna(X.median())

    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    return {
        'prediction': 'diseased' if prediction == 1 else 'healthy',
        'prediction_label': int(prediction),
        'probability_healthy': probability[0],
        'probability_diseased': probability[1],
        'confidence': max(probability)
    }


def compare_models(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple model types using LOOCV.

    Args:
        X: Feature matrix
        y: Labels
        verbose: Print comparison summary

    Returns:
        Dictionary with results for each model type
    """
    model_types = ['logistic_regression', 'random_forest', 'svm']
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (LOOCV)")
        print("=" * 60)

    for model_type in model_types:
        if verbose:
            print(f"\n  {model_type.upper().replace('_', ' ')}:")

        model = create_model(model_type)
        model_results = evaluate_loocv(model, X, y, verbose=verbose)
        results[model_type] = model_results

    if verbose:
        print("\n" + "-" * 60)
        print("  SUMMARY:")
        print("-" * 60)
        print(f"  {'Model':<25} {'Accuracy':<12} {'AUC-ROC':<12}")
        print("-" * 60)
        for model_type, res in results.items():
            acc = res['accuracy']
            auc = res['auc_roc'] if res['auc_roc'] is not None else 'N/A'
            auc_str = f"{auc:.3f}" if isinstance(auc, float) else auc
            print(f"  {model_type:<25} {acc:.3f}        {auc_str}")

        # Find best model
        best_model = max(results.items(),
                        key=lambda x: x[1]['auc_roc'] if x[1]['auc_roc'] else 0)
        print("-" * 60)
        print(f"  Best model: {best_model[0]} (AUC-ROC: {best_model[1]['auc_roc']:.3f})")

    return results
