import os
import csv
import numpy as np
from collections import defaultdict
from utilities import load_config
from pipelines.node_clustering_pipeline import _load_and_group_archives
from comparison import compare_graphs_by_traversal


def traversal_comparison(model_archives=None, ground_truth_label=None,
                         output_folder=None, config=None, config_path=None,
                         branch_match_distance_mm=None,
                         origin_max_distance_mm=None,
                         lca_first_bif_max_distance_mm=None):
    """
    Compare artery graphs to ground truth using structure-aware traversal.

    For each (patient, artery) combination, traverses the GT graph and matches
    junctions to the model graph by nearest-neighbor matching with parent
    verification. Reports bifurcation-level metrics.

    Origin validation:
      - LCA: valid if origin distance < threshold OR first bifurcation matches
      - RCA: valid if origin distance < threshold only
      - Cases with invalid origins are excluded from aggregate metrics.

    Args:
        model_archives: dict of {model_label: archive_path}
        ground_truth_label: key in model_archives for the ground truth model
        output_folder: path for output CSVs
        config: configuration dict
        config_path: path to YAML/JSON config file
        branch_match_distance_mm: max distance to match junctions (default 5.0)
        origin_max_distance_mm: max origin distance for valid case (default 10.0)
        lca_first_bif_max_distance_mm: LCA fallback — if origin is off but first
            bifurcation is within this distance, case is still valid (default 5.0)
    """
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        config = {}

    if model_archives is None:
        model_archives = config.get('model_archives')
    if ground_truth_label is None:
        ground_truth_label = config.get('ground_truth_label')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if branch_match_distance_mm is None:
        branch_match_distance_mm = config.get('branch_match_distance_mm', 5.0)
    if origin_max_distance_mm is None:
        origin_max_distance_mm = config.get('origin_max_distance_mm', 10.0)
    if lca_first_bif_max_distance_mm is None:
        lca_first_bif_max_distance_mm = config.get('lca_first_bif_max_distance_mm', 5.0)

    if model_archives is None:
        raise ValueError("model_archives must be provided")
    if ground_truth_label is None:
        raise ValueError("ground_truth_label is required for traversal comparison")
    if ground_truth_label not in model_archives:
        raise ValueError(
            f"ground_truth_label '{ground_truth_label}' not found in "
            f"model_archives keys: {list(model_archives.keys())}"
        )
    if output_folder is None:
        raise ValueError("output_folder must be provided")

    os.makedirs(output_folder, exist_ok=True)
    model_labels = [m for m in model_archives if m != ground_truth_label]

    print("=" * 80)
    print("GRAPH TRAVERSAL COMPARISON PIPELINE")
    print("=" * 80)
    print(f"Models: {model_labels}")
    print(f"Ground truth: {ground_truth_label}")
    print(f"Output folder: {output_folder}")
    print(f"Branch match distance: {branch_match_distance_mm} mm")
    print(f"Origin validation: max {origin_max_distance_mm} mm")
    print(f"LCA first-bifurcation fallback: max {lca_first_bif_max_distance_mm} mm")
    print("=" * 80)

    # =========================================================================
    # Step 1: Load archives
    # =========================================================================
    patient_data = _load_and_group_archives(model_archives)

    if not patient_data:
        print("No patient/artery combinations found.")
        return {'per_patient_results': {}, 'aggregate': {}}

    patient_data = {
        k: v for k, v in patient_data.items()
        if ground_truth_label in v and len(v) > 1
    }
    print(f"\nFound {len(patient_data)} patient/artery combinations with GT + model(s)")

    # =========================================================================
    # Step 2: Run traversal comparison
    # =========================================================================
    per_patient_results = {}

    for (patient_id, artery_type), model_data in sorted(patient_data.items()):
        gt_data = model_data[ground_truth_label]
        comparisons = {}

        for model_label in model_labels:
            if model_label not in model_data:
                continue

            m_data = model_data[model_label]
            try:
                result = compare_graphs_by_traversal(
                    gt_graph=gt_data['graph'],
                    model_graph=m_data['graph'],
                    spacing_info_gt=gt_data['spacing_info'],
                    spacing_info_model=m_data['spacing_info'],
                    branch_match_distance_mm=branch_match_distance_mm,
                )

                # Origin validation
                origin_valid = _validate_origin(
                    result, artery_type,
                    origin_max_distance_mm, lca_first_bif_max_distance_mm,
                )
                result['origin_valid'] = origin_valid

                comparisons[model_label] = result
                s = result['summary']
                valid_str = "VALID" if origin_valid else "EXCLUDED"
                print(f"  {patient_id}/{artery_type} vs {model_label}: "
                      f"bif_acc={s['bifurcation_accuracy']:.0%} "
                      f"bif_rec={s['bifurcation_recall']:.0%} "
                      f"junc_err={s['mean_junction_error_mm']:.1f}mm "
                      f"origin={s['origin_offset_mm']:.1f}mm [{valid_str}]")
            except Exception as e:
                print(f"  {patient_id}/{artery_type} vs {model_label}: FAILED — {e}")

        if comparisons:
            per_patient_results[(patient_id, artery_type)] = comparisons

    # =========================================================================
    # Step 3: Aggregate and save
    # =========================================================================
    aggregate = _compute_aggregate(per_patient_results, model_labels)

    _save_summary_csv(per_patient_results, output_folder)
    _save_aggregate_csv(aggregate, output_folder)
    _print_summary(aggregate, model_labels)

    # =========================================================================
    # Step 4: Figures
    # =========================================================================
    fig_dir = os.path.join(output_folder, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    figure_format = config.get('figure_format', 'png')

    from visualizations.visualize_traversal import plot_traversal_summary
    plot_traversal_summary(
        aggregate, per_patient_results,
        output_dir=fig_dir, figure_format=figure_format,
    )
    print(f"Figures saved to {fig_dir}/")

    return {
        'per_patient_results': per_patient_results,
        'aggregate': aggregate,
    }


def _validate_origin(result, artery_type, origin_max_distance_mm,
                      lca_first_bif_max_distance_mm):
    """
    Determine if a comparison has a valid origin for inclusion in aggregates.

    LCA: valid if origin distance < threshold OR first bifurcation error < threshold.
    RCA: valid if origin distance < threshold only.
    """
    origin_ok = result['origin_offset_mm'] <= origin_max_distance_mm

    if artery_type == 'LCA' and not origin_ok:
        first_bif_err = result.get('first_bifurcation_error_mm')
        if first_bif_err is not None and first_bif_err <= lca_first_bif_max_distance_mm:
            return True

    return origin_ok


def _save_summary_csv(per_patient_results, output_folder):
    """Save per-patient per-model summary metrics."""
    path = os.path.join(output_folder, 'traversal_summary.csv')
    rows = []

    for (patient_id, artery_type), comparisons in sorted(per_patient_results.items()):
        for model_label, result in sorted(comparisons.items()):
            s = result['summary']
            rows.append({
                'patient_id': patient_id,
                'artery_type': artery_type,
                'model': model_label,
                'origin_valid': result['origin_valid'],
                'origin_offset_mm': f"{s['origin_offset_mm']:.3f}",
                'first_bif_error_mm': f"{s['first_bifurcation_error_mm']:.3f}" if s['first_bifurcation_error_mm'] is not None else '',
                'bifurcation_accuracy': f"{s['bifurcation_accuracy']:.4f}",
                'bifurcation_recall': f"{s['bifurcation_recall']:.4f}",
                'num_matched_bifs': s['num_matched_bifurcations'],
                'total_gt_bifs': s['total_gt_bifurcations'],
                'total_model_bifs': s['total_model_bifurcations'],
                'mean_junction_error_mm': f"{s['mean_junction_error_mm']:.3f}",
                'median_junction_error_mm': f"{s['median_junction_error_mm']:.3f}",
            })

    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved to {path}")


def _cv(values):
    """Coefficient of variation (std/mean). Returns 0.0 if mean is 0."""
    m = float(np.mean(values))
    return float(np.std(values) / m) if m != 0 else 0.0


def _compute_aggregate(per_patient_results, model_labels):
    """Aggregate results by artery type and model, only including valid-origin cases."""
    aggregate = {}

    for artery_type in ['LCA', 'RCA']:
        relevant = {
            k: v for k, v in per_patient_results.items()
            if k[1] == artery_type
        }
        if not relevant:
            continue

        per_model = {}
        for model_label in model_labels:
            # Only include valid-origin cases
            valid_results = []
            excluded_count = 0
            for comparisons in relevant.values():
                if model_label not in comparisons:
                    continue
                r = comparisons[model_label]
                if r['origin_valid']:
                    valid_results.append(r)
                else:
                    excluded_count += 1

            if not valid_results:
                continue

            summaries = [r['summary'] for r in valid_results]
            origins = [s['origin_offset_mm'] for s in summaries]
            accs = [s['bifurcation_accuracy'] for s in summaries]
            recs = [s['bifurcation_recall'] for s in summaries]
            junc_errs = [s['mean_junction_error_mm'] for s in summaries
                         if s['mean_junction_error_mm'] > 0]
            # Collect all individual junction errors for box plots
            all_junc_errors = []
            for s in summaries:
                all_junc_errors.extend(s.get('junction_errors', []))

            per_model[model_label] = {
                'num_valid': len(valid_results),
                'num_excluded': excluded_count,
                'mean_origin_offset_mm': float(np.mean(origins)),
                'std_origin_offset_mm': float(np.std(origins)),
                'mean_bifurcation_accuracy': float(np.mean(accs)),
                'std_bifurcation_accuracy': float(np.std(accs)),
                'mean_bifurcation_recall': float(np.mean(recs)),
                'std_bifurcation_recall': float(np.std(recs)),
                'mean_junction_error_mm': float(np.mean(junc_errs)) if junc_errs else 0.0,
                'std_junction_error_mm': float(np.std(junc_errs)) if junc_errs else 0.0,
                'all_junction_errors': all_junc_errors,
                'total_gt_bifs': sum(s['total_gt_bifurcations'] for s in summaries),
                'total_model_bifs': sum(s['total_model_bifurcations'] for s in summaries),
                'total_matched_bifs': sum(s['num_matched_bifurcations'] for s in summaries),
            }

        # CV across models
        cv_across_models = {}
        if len(per_model) >= 2:
            for metric in ['mean_bifurcation_accuracy', 'mean_bifurcation_recall',
                           'mean_origin_offset_mm', 'mean_junction_error_mm']:
                values = [s[metric] for s in per_model.values()]
                cv_across_models[metric] = _cv(values)

        aggregate[artery_type] = {
            'per_model': per_model,
            'cv_across_models': cv_across_models,
        }

    return aggregate


def _save_aggregate_csv(aggregate, output_folder):
    """Save aggregate results to CSV."""
    path = os.path.join(output_folder, 'traversal_aggregate.csv')
    rows = []

    for artery_type in ['LCA', 'RCA']:
        if artery_type not in aggregate:
            continue
        per_model = aggregate[artery_type]['per_model']
        cv = aggregate[artery_type]['cv_across_models']

        for model_label, stats in sorted(per_model.items()):
            rows.append({
                'artery_type': artery_type,
                'model': model_label,
                'num_valid': stats['num_valid'],
                'num_excluded': stats['num_excluded'],
                'mean_origin_offset_mm': f"{stats['mean_origin_offset_mm']:.3f}",
                'mean_bifurcation_accuracy': f"{stats['mean_bifurcation_accuracy']:.4f}",
                'std_bifurcation_accuracy': f"{stats['std_bifurcation_accuracy']:.4f}",
                'mean_bifurcation_recall': f"{stats['mean_bifurcation_recall']:.4f}",
                'std_bifurcation_recall': f"{stats['std_bifurcation_recall']:.4f}",
                'mean_junction_error_mm': f"{stats['mean_junction_error_mm']:.3f}",
                'total_matched_bifs': stats['total_matched_bifs'],
                'total_gt_bifs': stats['total_gt_bifs'],
                'total_model_bifs': stats['total_model_bifs'],
            })

        if cv:
            cv_row = {
                'artery_type': artery_type,
                'model': 'CV_across_models',
                'mean_origin_offset_mm': f"{cv.get('mean_origin_offset_mm', 0):.4f}",
                'mean_bifurcation_accuracy': f"{cv.get('mean_bifurcation_accuracy', 0):.4f}",
                'mean_bifurcation_recall': f"{cv.get('mean_bifurcation_recall', 0):.4f}",
                'mean_junction_error_mm': f"{cv.get('mean_junction_error_mm', 0):.4f}",
            }
            rows.append(cv_row)

    if rows:
        fieldnames = list(rows[0].keys())
        # Ensure all rows have all fields
        for row in rows:
            for fn in fieldnames:
                row.setdefault(fn, '')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Aggregate results saved to {path}")


def _print_summary(aggregate, model_labels):
    """Print aggregate summary to console."""
    print("\n" + "=" * 80)
    print("TRAVERSAL COMPARISON — AGGREGATE RESULTS")
    print("  (only valid-origin cases included)")
    print("=" * 80)

    for artery_type in ['LCA', 'RCA']:
        if artery_type not in aggregate:
            continue

        per_model = aggregate[artery_type]['per_model']
        cv = aggregate[artery_type]['cv_across_models']

        print(f"\n{artery_type}:")
        print(f"  {'Model':<20s} {'BifAcc':>8s} {'BifRec':>8s} "
              f"{'JuncErr':>9s} {'Origin':>9s} "
              f"{'Valid':>6s} {'Excl':>6s}")
        print(f"  {'-'*19} {'-'*8} {'-'*8} "
              f"{'-'*9} {'-'*9} "
              f"{'-'*6} {'-'*6}")

        for model_label in model_labels:
            if model_label not in per_model:
                continue
            s = per_model[model_label]
            print(f"  {model_label:<20s} "
                  f"{s['mean_bifurcation_accuracy']:>7.1%}  "
                  f"{s['mean_bifurcation_recall']:>7.1%}  "
                  f"{s['mean_junction_error_mm']:>6.1f}mm  "
                  f"{s['mean_origin_offset_mm']:>6.1f}mm  "
                  f"{s['num_valid']:>5d} "
                  f"{s['num_excluded']:>5d}")

        if cv:
            print(f"\n  CV across models (higher = more inter-model variability):")
            display = {
                'mean_bifurcation_accuracy': 'Bifurcation Accuracy',
                'mean_bifurcation_recall': 'Bifurcation Recall',
                'mean_origin_offset_mm': 'Origin Offset',
                'mean_junction_error_mm': 'Junction Error',
            }
            for key, label in display.items():
                if key in cv:
                    print(f"    {label:<30s} {cv[key]:.3f}")

    print("\n" + "=" * 80)
