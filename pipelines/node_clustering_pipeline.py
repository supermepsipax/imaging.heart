import os
import pickle
import tarfile
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from utilities import load_config, load_artery_analysis
from utilities.input_output import strip_filenames
from clustering import (
    classify_and_extract_nodes,
    compute_origin_spread,
    cluster_nodes,
    compute_ground_truth_distances,
    compute_pairwise_model_distances,
)
import csv


def cluster_artery_nodes(model_archives=None, ground_truth_label=None,
                         output_folder=None, config=None, config_path=None,
                         dbscan_eps_mm=None, max_match_distance_mm=None,
                         visualize_3d=None, visualize_3d_percent=None,
                         save_2d_figures=None, figure_format=None):
    """
    Perform node clustering sensitivity analysis across multiple segmentation models.

    Loads processed artery archives from multiple models, clusters corresponding nodes
    (origin, bifurcation, endpoint) using DBSCAN, and reports matching statistics.
    Optionally computes reference-based distances when a ground truth model is designated.

    Args:
        model_archives (dict, optional): {model_label: tar_path} mapping model names to archive paths
        ground_truth_label (str, optional): Key in model_archives for the ground truth model, or None
        output_folder (str, optional): Path for output CSVs and figures
        config (dict, optional): Configuration dictionary
        config_path (str, optional): Path to YAML/JSON config file
        dbscan_eps_mm (float, optional): DBSCAN epsilon parameter in mm (default: 5.0)
        max_match_distance_mm (float, optional): Max distance to consider a match in mm (default: 15.0)
        visualize_3d (bool, optional): Whether to show interactive 3D plots
        visualize_3d_percent (float, optional): Percentage of patient/artery combos to visualize
        save_2d_figures (bool, optional): Whether to save 2D report figures
        figure_format (str, optional): Figure format, 'png' or 'pdf'

    Returns:
        dict: Summary of clustering results including per-patient and aggregate statistics
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
    if dbscan_eps_mm is None:
        dbscan_eps_mm = config.get('dbscan_eps_mm', 5.0)
    if max_match_distance_mm is None:
        max_match_distance_mm = config.get('max_match_distance_mm', 8.0)
    if visualize_3d is None:
        visualize_3d = config.get('visualize_3d', False)
    if visualize_3d_percent is None:
        visualize_3d_percent = config.get('visualize_3d_percent', 10)
    if save_2d_figures is None:
        save_2d_figures = config.get('save_2d_figures', True)
    if figure_format is None:
        figure_format = config.get('figure_format', 'png')

    if model_archives is None:
        raise ValueError("model_archives must be provided either as a parameter or in the config file")
    if output_folder is None:
        raise ValueError("output_folder must be provided either as a parameter or in the config file")
    if ground_truth_label is not None and ground_truth_label not in model_archives:
        raise ValueError(f"ground_truth_label '{ground_truth_label}' not found in model_archives keys: {list(model_archives.keys())}")

    os.makedirs(output_folder, exist_ok=True)

    print("=" * 80)
    print("NODE CLUSTERING SENSITIVITY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Models: {list(model_archives.keys())}")
    print(f"Ground truth: {ground_truth_label or 'None (symmetric mode)'}")
    print(f"Output folder: {output_folder}")
    print(f"DBSCAN eps: {dbscan_eps_mm} mm")
    print(f"Max match distance: {max_match_distance_mm} mm")
    print("=" * 80)

    # =========================================================================
    # Step 1: Load all archives and group by (patient_id, artery_type)
    # =========================================================================
    patient_data = _load_and_group_archives(model_archives)

    if not patient_data:
        print("No patient/artery combinations found with data from 2+ models.")
        return {'per_patient_results': {}, 'aggregate': {}}

    print(f"\nFound {len(patient_data)} patient/artery combinations with 2+ models")

    # =========================================================================
    # Step 2: Run clustering for each (patient_id, artery_type)
    # =========================================================================
    per_patient_results = {}
    all_pairwise = defaultdict(lambda: defaultdict(list))  # {node_type: {(m_a, m_b): [distances]}}

    for (patient_id, artery_type), model_data in sorted(patient_data.items()):
        print(f"\n--- {patient_id} / {artery_type} ({len(model_data)} models) ---")

        result = _analyze_single_combination(
            model_data, dbscan_eps_mm, max_match_distance_mm, ground_truth_label
        )
        per_patient_results[(patient_id, artery_type)] = result

        # Collect pairwise distances for heatmaps
        for node_type in ['bifurcation', 'endpoint']:
            for pair, dist in result.get('pairwise_distances', {}).get(node_type, {}).items():
                all_pairwise[node_type][pair].append(dist)

    # =========================================================================
    # Step 3: Aggregate results by artery type
    # =========================================================================
    aggregate = _aggregate_results(per_patient_results, list(model_archives.keys()))

    # =========================================================================
    # Step 4: Save CSV output
    # =========================================================================
    _save_per_patient_csv(per_patient_results, output_folder, list(model_archives.keys()))
    _save_aggregate_csv(aggregate, output_folder)
    _save_pairwise_heatmap_csv(all_pairwise, output_folder)

    if ground_truth_label:
        _save_ground_truth_csv(per_patient_results, output_folder, ground_truth_label)

    # =========================================================================
    # Step 5: Visualization
    # =========================================================================
    if visualize_3d or save_2d_figures:
        from visualizations.visualize_clusters import (
            plot_cluster_3d,
            plot_cluster_projections_2d,
            plot_summary_bar_charts,
            plot_distance_box_plots,
            plot_pairwise_heatmaps,
            plot_ground_truth_comparison,
        )

    if save_2d_figures:
        fig_dir = os.path.join(output_folder, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Per-patient 2D projection plots
        keys = list(per_patient_results.keys())
        for key in keys:
            patient_id, artery_type = key
            result = per_patient_results[key]
            plot_cluster_projections_2d(
                result['node_sets_by_model'],
                result['clustering'],
                patient_id=patient_id,
                artery_type=artery_type,
                output_dir=fig_dir,
                figure_format=figure_format,
            )

        # Summary plots
        plot_summary_bar_charts(aggregate, output_dir=fig_dir, figure_format=figure_format)
        plot_distance_box_plots(per_patient_results, output_dir=fig_dir, figure_format=figure_format)
        plot_pairwise_heatmaps(all_pairwise, output_dir=fig_dir, figure_format=figure_format)

        if ground_truth_label:
            plot_ground_truth_comparison(
                aggregate, ground_truth_label,
                output_dir=fig_dir, figure_format=figure_format,
            )

        print(f"\n2D figures saved to {fig_dir}/")

    if visualize_3d:
        keys = list(per_patient_results.keys())
        sample_count = max(1, int(len(keys) * visualize_3d_percent / 100))
        sample_count = min(sample_count, len(keys))
        sample_keys = random.sample(keys, sample_count)

        for key in sample_keys:
            patient_id, artery_type = key
            result = per_patient_results[key]
            plot_cluster_3d(
                result['node_sets_by_model'],
                result['clustering'],
                title=f"Node Clustering: {patient_id} {artery_type}",
            )

    # =========================================================================
    # Print summary
    # =========================================================================
    _print_summary(aggregate)

    return {
        'per_patient_results': per_patient_results,
        'aggregate': aggregate,
        'pairwise_distances': dict(all_pairwise),
    }


def _load_and_group_archives(model_archives):
    """
    Load all model archives and group analyses by (patient_id, artery_type).

    Returns:
        dict: {(patient_id, artery_type): {model_label: {'graph': DiGraph, 'spacing_info': tuple}}}
    """
    patient_data = defaultdict(dict)

    for model_label, archive_path in model_archives.items():
        print(f"\nLoading archive: {model_label} ({archive_path})")

        try:
            with tarfile.open(archive_path, 'r:*') as tar:
                pkl_members = [m for m in tar.getmembers() if m.name.endswith('.pkl') and m.isfile()]
                pkl_members.sort(key=lambda m: m.name)

                # Normalize filenames for patient matching
                member_names = [m.name for m in pkl_members]
                stripped_names = strip_filenames([Path(n).stem for n in member_names])

                print(f"  Found {len(pkl_members)} pickle files")

                for member, stripped in zip(pkl_members, stripped_names):
                    try:
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            continue
                        data = pickle.load(file_obj)

                        graph = data.get('final_graph')
                        spacing_info = data.get('spacing_info')
                        metadata = data.get('metadata', {})
                        artery_type = metadata.get('artery', 'UNKNOWN')

                        if graph is None or spacing_info is None:
                            print(f"  Skipping {member.name}: missing graph or spacing_info")
                            continue

                        # Extract patient ID from stripped filename
                        # stripped is like "Normal_1" or "Diseased_2"
                        patient_id = Path(stripped).stem

                        key = (patient_id, artery_type)
                        patient_data[key][model_label] = {
                            'graph': graph,
                            'spacing_info': spacing_info,
                        }

                    except Exception as e:
                        print(f"  Failed to load {member.name}: {e}")

        except Exception as e:
            print(f"  Failed to open archive {archive_path}: {e}")

    # Filter to combinations with 2+ models
    filtered = {k: v for k, v in patient_data.items() if len(v) >= 2}
    return filtered


def _analyze_single_combination(model_data, dbscan_eps_mm, max_match_distance_mm,
                                ground_truth_label):
    """
    Run node clustering analysis for a single (patient_id, artery_type) combination.

    Returns dict with clustering results for each node type plus metadata.
    """
    # Extract and classify nodes for each model
    node_sets_by_model = {}
    for model_label, data in model_data.items():
        nodes = classify_and_extract_nodes(data['graph'], data['spacing_info'])
        node_sets_by_model[model_label] = nodes

    # Origin analysis
    origin_coords = {m: nodes['origin'] for m, nodes in node_sets_by_model.items()}
    origin_result = compute_origin_spread(origin_coords)

    # Bifurcation clustering
    bif_sets = {m: nodes['bifurcation'] for m, nodes in node_sets_by_model.items()}
    bif_clustering = cluster_nodes(bif_sets, dbscan_eps_mm, max_match_distance_mm)

    # Endpoint clustering
    ep_sets = {m: nodes['endpoint'] for m, nodes in node_sets_by_model.items()}
    ep_clustering = cluster_nodes(ep_sets, dbscan_eps_mm, max_match_distance_mm)

    # Pairwise model distances (for heatmaps)
    pairwise = {
        'bifurcation': compute_pairwise_model_distances(bif_sets),
        'endpoint': compute_pairwise_model_distances(ep_sets),
    }

    result = {
        'node_sets_by_model': node_sets_by_model,
        'origin': origin_result,
        'clustering': {
            'bifurcation': bif_clustering,
            'endpoint': ep_clustering,
        },
        'pairwise_distances': pairwise,
        'models': list(model_data.keys()),
    }

    # Ground truth distances if applicable
    if ground_truth_label and ground_truth_label in model_data:
        gt_results = {}
        for node_type in ['bifurcation', 'endpoint']:
            sets = {m: nodes[node_type] for m, nodes in node_sets_by_model.items()}
            gt_results[node_type] = compute_ground_truth_distances(
                sets, ground_truth_label, max_match_distance_mm
            )

        # Origin GT distance
        gt_origin = origin_result['per_model_distance']
        gt_results['origin'] = {
            m: {'distance_to_gt': gt_origin.get(m, 0.0)}
            for m in model_data if m != ground_truth_label
        }

        result['ground_truth'] = gt_results

    return result


def _aggregate_results(per_patient_results, model_labels):
    """Aggregate per-patient results by artery type (LCA vs RCA)."""
    aggregate = {}

    for artery_type in ['LCA', 'RCA']:
        relevant = {k: v for k, v in per_patient_results.items() if k[1] == artery_type}
        if not relevant:
            continue

        origin_spreads = [r['origin']['max_spread_radius_mm'] for r in relevant.values()]
        origin_means = [r['origin']['mean_spread_radius_mm'] for r in relevant.values()]

        bif_matched_dists = [
            r['clustering']['bifurcation']['stats']['mean_matched_distance_mm']
            for r in relevant.values()
            if r['clustering']['bifurcation']['stats']['num_matched_clusters'] > 0
        ]
        ep_matched_dists = [
            r['clustering']['endpoint']['stats']['mean_matched_distance_mm']
            for r in relevant.values()
            if r['clustering']['endpoint']['stats']['num_matched_clusters'] > 0
        ]

        # Per-model totals: {model: {total, matched, full, partial, unmatched}} for bif and ep
        accum = {}
        for node_type in ['bifurcation', 'endpoint']:
            accum[node_type] = {
                'total': defaultdict(int),
                'matched': defaultdict(int),
                'full': defaultdict(int),
                'partial': defaultdict(int),
                'unmatched': defaultdict(int),
                'matched_clusters': 0,
                'full_clusters': 0,
                'partial_clusters': 0,
            }

        model_case_count = defaultdict(int)

        for r in relevant.values():
            for node_type in ['bifurcation', 'endpoint']:
                stats = r['clustering'][node_type]['stats']
                a = accum[node_type]
                for m, count in stats.get('total_per_model', {}).items():
                    a['total'][m] += count
                for m, count in stats.get('matched_per_model', {}).items():
                    a['matched'][m] += count
                for m, count in stats.get('full_match_per_model', {}).items():
                    a['full'][m] += count
                for m, count in stats.get('partial_match_per_model', {}).items():
                    a['partial'][m] += count
                for m, count in stats.get('unmatched_per_model', {}).items():
                    a['unmatched'][m] += count
                a['matched_clusters'] += stats['num_matched_clusters']
                a['full_clusters'] += stats.get('num_full_clusters', 0)
                a['partial_clusters'] += stats.get('num_partial_clusters', 0)

            for m in r['models']:
                model_case_count[m] += 1

        def _build_node_type_stats(a, matched_dists):
            return {
                'mean_matched_distance_mm': float(np.mean(matched_dists)) if matched_dists else 0.0,
                'std_matched_distance_mm': float(np.std(matched_dists)) if matched_dists else 0.0,
                'total_matched_clusters': a['matched_clusters'],
                'full_agreement_clusters': a['full_clusters'],
                'partial_agreement_clusters': a['partial_clusters'],
                'total_per_model': dict(a['total']),
                'matched_per_model': dict(a['matched']),
                'full_match_per_model': dict(a['full']),
                'partial_match_per_model': dict(a['partial']),
                'total_unmatched_per_model': dict(a['unmatched']),
            }

        # Aggregate ground truth stats if available
        gt_aggregate = None
        has_gt = any('ground_truth' in r for r in relevant.values())
        if has_gt:
            gt_aggregate = {}
            for node_type in ['bifurcation', 'endpoint']:
                gt_total = 0
                gt_matched = 0
                gt_unmatched = 0
                # Per-model: matched_gt, matched_other_only, unmatched
                model_gt_matched = defaultdict(int)
                model_other_only = defaultdict(int)
                model_unmatched_gt = defaultdict(int)
                # For F1: accumulate raw counts to compute aggregate precision/recall
                model_total_nodes = defaultdict(int)
                model_gt_recalled = defaultdict(int)

                for r in relevant.values():
                    gt_data = r.get('ground_truth', {}).get(node_type, {})
                    coverage = gt_data.get('_gt_coverage', {})
                    gt_total += coverage.get('total', 0)
                    gt_matched += coverage.get('matched', 0)
                    gt_unmatched += coverage.get('unmatched', 0)

                    clustering_stats = r['clustering'][node_type]['stats']

                    for m, m_gt in gt_data.items():
                        if m.startswith('_'):
                            continue
                        num_matched_gt = m_gt.get('num_matched', 0)
                        num_unmatched_gt = m_gt.get('num_unmatched', 0)
                        total_dbscan_matched = clustering_stats.get('matched_per_model', {}).get(m, 0)
                        other_only = max(0, total_dbscan_matched - num_matched_gt)

                        truly_unmatched = max(0, num_unmatched_gt - other_only)

                        model_gt_matched[m] += num_matched_gt
                        model_other_only[m] += other_only
                        model_unmatched_gt[m] += truly_unmatched
                        model_total_nodes[m] += num_matched_gt + num_unmatched_gt
                        model_gt_recalled[m] += m_gt.get('num_gt_recalled', 0)

                # Compute aggregate Dice per model: 2*overlap / (GT_nodes + model_nodes)
                model_dice = {}
                for m in model_gt_matched:
                    total_m = model_total_nodes[m]
                    overlap = model_gt_matched[m]
                    dice = (2 * overlap) / (gt_total + total_m) if (gt_total + total_m) > 0 else 0.0
                    model_dice[m] = float(dice)

                gt_aggregate[node_type] = {
                    'gt_total': gt_total,
                    'gt_matched': gt_matched,
                    'gt_unmatched': gt_unmatched,
                    'model_gt_matched': dict(model_gt_matched),
                    'model_other_only': dict(model_other_only),
                    'model_unmatched': dict(model_unmatched_gt),
                    'model_dice': model_dice,
                }

            # Origin GT scores: 1 / (1 + distance) per model, averaged across patients
            origin_gt_scores = defaultdict(list)
            for r in relevant.values():
                gt_origin = r.get('ground_truth', {}).get('origin', {})
                for m, m_data in gt_origin.items():
                    if m.startswith('_'):
                        continue
                    dist = m_data.get('distance_to_gt', 0.0)
                    origin_gt_scores[m].append(2.0 / (1.0 + np.exp(dist / 2.0)))

            gt_aggregate['origin'] = {
                m: {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                }
                for m, scores in origin_gt_scores.items()
            }

        aggregate[artery_type] = {
            'num_patients': len(relevant),
            'origin': {
                'mean_max_spread_mm': float(np.mean(origin_spreads)) if origin_spreads else 0.0,
                'std_max_spread_mm': float(np.std(origin_spreads)) if origin_spreads else 0.0,
                'mean_avg_spread_mm': float(np.mean(origin_means)) if origin_means else 0.0,
                'std_avg_spread_mm': float(np.std(origin_means)) if origin_means else 0.0,
            },
            'bifurcation': _build_node_type_stats(accum['bifurcation'], bif_matched_dists),
            'endpoint': _build_node_type_stats(accum['endpoint'], ep_matched_dists),
            'model_case_count': dict(model_case_count),
            'ground_truth': gt_aggregate,
        }

    return aggregate


def _save_per_patient_csv(per_patient_results, output_folder, model_labels):
    """Save per-patient clustering results to CSV."""
    path = os.path.join(output_folder, 'per_patient_clustering.csv')
    rows = []

    for (patient_id, artery_type), result in sorted(per_patient_results.items()):
        row = {
            'patient_id': patient_id,
            'artery_type': artery_type,
            'num_models': len(result['models']),
            'origin_max_spread_mm': result['origin']['max_spread_radius_mm'],
            'origin_mean_spread_mm': result['origin']['mean_spread_radius_mm'],
            'bif_matched_clusters': result['clustering']['bifurcation']['stats']['num_matched_clusters'],
            'bif_mean_matched_dist_mm': result['clustering']['bifurcation']['stats']['mean_matched_distance_mm'],
            'bif_total_unmatched': result['clustering']['bifurcation']['stats']['total_unmatched'],
            'ep_matched_clusters': result['clustering']['endpoint']['stats']['num_matched_clusters'],
            'ep_mean_matched_dist_mm': result['clustering']['endpoint']['stats']['mean_matched_distance_mm'],
            'ep_total_unmatched': result['clustering']['endpoint']['stats']['total_unmatched'],
        }

        # Add per-model unmatched counts
        for m in model_labels:
            row[f'bif_unmatched_{m}'] = result['clustering']['bifurcation']['stats']['unmatched_per_model'].get(m, 0)
            row[f'ep_unmatched_{m}'] = result['clustering']['endpoint']['stats']['unmatched_per_model'].get(m, 0)

        rows.append(row)

    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-patient results saved to {path}")


def _save_aggregate_csv(aggregate, output_folder):
    """Save aggregate results to CSV."""
    path = os.path.join(output_folder, 'aggregate_clustering.csv')
    rows = []

    for artery_type, stats in sorted(aggregate.items()):
        row = {
            'artery_type': artery_type,
            'num_patients': stats['num_patients'],
            'origin_mean_max_spread_mm': stats['origin']['mean_max_spread_mm'],
            'origin_std_max_spread_mm': stats['origin']['std_max_spread_mm'],
            'bif_mean_matched_dist_mm': stats['bifurcation']['mean_matched_distance_mm'],
            'bif_std_matched_dist_mm': stats['bifurcation']['std_matched_distance_mm'],
            'bif_total_matched_clusters': stats['bifurcation']['total_matched_clusters'],
            'ep_mean_matched_dist_mm': stats['endpoint']['mean_matched_distance_mm'],
            'ep_std_matched_dist_mm': stats['endpoint']['std_matched_distance_mm'],
            'ep_total_matched_clusters': stats['endpoint']['total_matched_clusters'],
        }
        rows.append(row)

    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Aggregate results saved to {path}")


def _save_pairwise_heatmap_csv(all_pairwise, output_folder):
    """Save pairwise model distances to CSV for heatmap generation."""
    for node_type, pair_dists in all_pairwise.items():
        path = os.path.join(output_folder, f'pairwise_distances_{node_type}.csv')
        rows = []
        for (model_a, model_b), distances in sorted(pair_dists.items()):
            rows.append({
                'model_a': model_a,
                'model_b': model_b,
                'mean_distance_mm': float(np.mean(distances)),
                'std_distance_mm': float(np.std(distances)),
                'num_patients': len(distances),
            })

        if rows:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Pairwise distances ({node_type}) saved to {path}")


def _save_ground_truth_csv(per_patient_results, output_folder, ground_truth_label):
    """Save ground truth reference distance results to CSV."""
    path = os.path.join(output_folder, 'ground_truth_distances.csv')
    rows = []

    for (patient_id, artery_type), result in sorted(per_patient_results.items()):
        gt = result.get('ground_truth')
        if gt is None:
            continue

        for node_type in ['origin', 'bifurcation', 'endpoint']:
            gt_node = gt.get(node_type, {})
            for model_label, model_gt in gt_node.items():
                if model_label.startswith('_'):
                    continue
                if node_type == 'origin':
                    row = {
                        'patient_id': patient_id,
                        'artery_type': artery_type,
                        'node_type': node_type,
                        'model': model_label,
                        'mean_distance_mm': model_gt['distance_to_gt'],
                        'median_distance_mm': model_gt['distance_to_gt'],
                        'max_distance_mm': model_gt['distance_to_gt'],
                        'num_matched': 1,
                        'num_unmatched': 0,
                        'dice': '',
                    }
                else:
                    row = {
                        'patient_id': patient_id,
                        'artery_type': artery_type,
                        'node_type': node_type,
                        'model': model_label,
                        'mean_distance_mm': model_gt['mean_distance'],
                        'median_distance_mm': model_gt['median_distance'],
                        'max_distance_mm': model_gt['max_distance'],
                        'num_matched': model_gt['num_matched'],
                        'num_unmatched': model_gt['num_unmatched'],
                        'dice': model_gt.get('dice', ''),
                    }
                rows.append(row)

    if rows:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ground truth distances saved to {path}")


def _print_summary(aggregate):
    """Print a summary of aggregate results with per-model match percentages."""
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS SUMMARY")
    print("=" * 80)

    for artery_type in ['LCA', 'RCA']:
        if artery_type not in aggregate:
            continue
        stats = aggregate[artery_type]
        n = stats['num_patients']
        case_counts = stats.get('model_case_count', {})

        print(f"\n{artery_type} ({n} patients):")
        print(f"  Origin spread:       {stats['origin']['mean_max_spread_mm']:.2f} +/- {stats['origin']['std_max_spread_mm']:.2f} mm (max radius)")

        for node_type in ['bifurcation', 'endpoint']:
            ns = stats[node_type]
            total_clusters = ns['total_matched_clusters']
            full_clusters = ns.get('full_agreement_clusters', 0)
            partial_clusters = ns.get('partial_agreement_clusters', 0)
            full_pct = (full_clusters / total_clusters * 100) if total_clusters > 0 else 0.0
            partial_pct = (partial_clusters / total_clusters * 100) if total_clusters > 0 else 0.0

            label = node_type.title()
            print(f"\n  {label} clusters: {total_clusters} matched across {n} patients")
            print(f"    Mean distance:     {ns['mean_matched_distance_mm']:.2f} +/- {ns['std_matched_distance_mm']:.2f} mm")
            print(f"    Full agreement:    {full_clusters}/{total_clusters} clusters ({full_pct:.1f}%) — all models agree")
            print(f"    Partial agreement: {partial_clusters}/{total_clusters} clusters ({partial_pct:.1f}%) — 2+ but not all models")

        # Per-model match rate breakdown
        models = sorted(set(
            list(stats['bifurcation'].get('total_per_model', {}).keys()) +
            list(stats['endpoint'].get('total_per_model', {}).keys())
        ))

        if models:
            print(f"\n  Per-model match rates:")
            for model in models:
                cases = case_counts.get(model, 0)
                print(f"    {model} ({cases} cases):")

                for node_type in ['bifurcation', 'endpoint']:
                    ns = stats[node_type]
                    total = ns['total_per_model'].get(model, 0)
                    matched = ns['matched_per_model'].get(model, 0)
                    full = ns.get('full_match_per_model', {}).get(model, 0)
                    partial = ns.get('partial_match_per_model', {}).get(model, 0)
                    unmatched = ns['total_unmatched_per_model'].get(model, 0)

                    match_pct = (matched / total * 100) if total > 0 else 0.0
                    full_pct = (full / total * 100) if total > 0 else 0.0
                    partial_pct = (partial / total * 100) if total > 0 else 0.0

                    label = node_type.title() + 's'
                    print(f"      {label:15s} {matched}/{total} matched ({match_pct:.1f}%) — "
                          f"full: {full} ({full_pct:.1f}%), partial: {partial} ({partial_pct:.1f}%), "
                          f"unmatched: {unmatched}")

        # Ground truth scorecard
        gt_agg = stats.get('ground_truth')
        if gt_agg:
            print(f"\n  Ground Truth Scorecard (Dice = 2*overlap / (GT + model)):")
            gt_models = set()
            for nt in ['bifurcation', 'endpoint']:
                gt_models.update(gt_agg.get(nt, {}).get('model_dice', {}).keys())
            gt_models = sorted(gt_models)

            for model in gt_models:
                origin_score = gt_agg.get('origin', {}).get(model, {}).get('mean_score', 0.0)
                bif_dice = gt_agg.get('bifurcation', {}).get('model_dice', {}).get(model, 0.0)
                ep_dice = gt_agg.get('endpoint', {}).get('model_dice', {}).get(model, 0.0)

                print(f"    {model}:")
                print(f"      Origin score:      {origin_score:.3f}")
                print(f"      Bifurcation Dice:  {bif_dice:.3f}")
                print(f"      Endpoint Dice:     {ep_dice:.3f}")

    print("\n" + "=" * 80)
