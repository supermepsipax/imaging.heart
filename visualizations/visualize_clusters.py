import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# Consistent model color palette
MODEL_COLORS_PLOTLY = [
    'blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow',
    'darkblue', 'darkred', 'darkgreen', 'darkorange',
]

MODEL_COLORS_MPL = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf',
    '#e377c2', '#bcbd22', '#8c564b', '#7f7f7f',
]

NODE_MARKERS_PLOTLY = {
    'origin': 'diamond',
    'bifurcation': 'circle',
    'endpoint': 'x',
}

NODE_MARKERS_MPL = {
    'origin': 'D',
    'bifurcation': 'o',
    'endpoint': '^',
}


def plot_cluster_3d(node_sets_by_model, clustering, title="Node Clustering"):
    """
    Interactive 3D plotly scatter plot showing nodes from all models.

    Color by model, shape by node type. Lines connect matched cluster members.

    Args:
        node_sets_by_model: dict of {model_label: {'origin': arr, 'bifurcation': arr, 'endpoint': arr}}
        clustering: dict with 'bifurcation' and 'endpoint' clustering results
        title: str
    """
    fig = go.Figure()
    model_labels = list(node_sets_by_model.keys())

    for i, model in enumerate(model_labels):
        color = MODEL_COLORS_PLOTLY[i % len(MODEL_COLORS_PLOTLY)]
        nodes = node_sets_by_model[model]

        for node_type in ['origin', 'bifurcation', 'endpoint']:
            coords = nodes[node_type]
            if len(coords) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                x=coords[:, 2], y=coords[:, 1], z=coords[:, 0],
                mode='markers',
                marker=dict(
                    size=8 if node_type == 'origin' else 5,
                    symbol=NODE_MARKERS_PLOTLY[node_type],
                    color=color,
                ),
                name=f"{model} - {node_type}",
                legendgroup=model,
            ))

    # Draw lines for matched clusters
    for node_type in ['bifurcation', 'endpoint']:
        clusters = clustering.get(node_type, {}).get('clusters', [])
        for cluster in clusters:
            if not cluster['matched']:
                continue
            all_pts = []
            for model, coords in cluster['models'].items():
                for c in coords:
                    all_pts.append(c)

            if len(all_pts) < 2:
                continue

            centroid = cluster['centroid']
            for pt in all_pts:
                fig.add_trace(go.Scatter3d(
                    x=[pt[2], centroid[2]],
                    y=[pt[1], centroid[1]],
                    z=[pt[0], centroid[0]],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=False,
                ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (dim2)',
            yaxis_title='Y (dim1)',
            zaxis_title='Z (dim0)',
            aspectmode='data',
        ),
        template='plotly_dark',
    )
    fig.show()


def plot_cluster_projections_2d(node_sets_by_model, clustering,
                                patient_id="", artery_type="",
                                output_dir=None, figure_format='png'):
    """
    2D projection plots of node clusters onto two orthogonal planes.

    Creates side-by-side subplots for dim1-dim2 (XY) and dim0-dim2 (XZ) projections.

    Args:
        node_sets_by_model: dict of {model_label: {'origin': arr, 'bifurcation': arr, 'endpoint': arr}}
        clustering: dict with 'bifurcation' and 'endpoint' clustering results
        patient_id: str
        artery_type: str
        output_dir: str or None — if set, saves figure to this directory
        figure_format: 'png' or 'pdf'
    """
    model_labels = list(node_sets_by_model.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Projection definitions: (x_idx, y_idx, xlabel, ylabel)
    projections = [
        (2, 1, 'dim2 (X)', 'dim1 (Y)'),
        (2, 0, 'dim2 (X)', 'dim0 (Z)'),
    ]

    for ax_idx, (xi, yi, xlabel, ylabel) in enumerate(projections):
        ax = axes[ax_idx]

        for i, model in enumerate(model_labels):
            color = MODEL_COLORS_MPL[i % len(MODEL_COLORS_MPL)]
            nodes = node_sets_by_model[model]

            for node_type in ['origin', 'bifurcation', 'endpoint']:
                coords = nodes[node_type]
                if len(coords) == 0:
                    continue

                marker = NODE_MARKERS_MPL[node_type]
                size = 80 if node_type == 'origin' else 30
                label = f"{model} - {node_type}" if ax_idx == 0 else None

                ax.scatter(
                    coords[:, xi], coords[:, yi],
                    c=color, marker=marker, s=size,
                    label=label, alpha=0.7, edgecolors='black', linewidths=0.5,
                )

        # Draw cluster connections
        for node_type in ['bifurcation', 'endpoint']:
            clusters = clustering.get(node_type, {}).get('clusters', [])
            for cluster in clusters:
                if not cluster['matched']:
                    continue
                centroid = cluster['centroid']
                for model, coords in cluster['models'].items():
                    for c in coords:
                        ax.plot(
                            [c[xi], centroid[xi]],
                            [c[yi], centroid[yi]],
                            color='gray', linewidth=0.5, linestyle='--', alpha=0.5,
                        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{patient_id} {artery_type} — {xlabel} vs {ylabel}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    axes[0].legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=3, fontsize=8)
    fig.suptitle(f"Node Clustering: {patient_id} {artery_type}", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_dir:
        filename = f"clusters_{patient_id}_{artery_type}.{figure_format}"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_summary_bar_charts(aggregate, output_dir=None, figure_format='png'):
    """
    Grouped bar charts for mean matched distance and unmatched counts by artery type.

    Args:
        aggregate: dict from pipeline aggregation
        output_dir: str or None
        figure_format: 'png' or 'pdf'
    """
    artery_types = [a for a in ['LCA', 'RCA'] if a in aggregate]
    if not artery_types:
        return

    # --- Mean matched distance by node type ---
    fig, ax = plt.subplots(figsize=(8, 5))
    node_types = ['origin', 'bifurcation', 'endpoint']
    x = np.arange(len(node_types))
    width = 0.8 / len(artery_types)

    for i, artery in enumerate(artery_types):
        stats = aggregate[artery]
        means = [
            stats['origin']['mean_avg_spread_mm'],
            stats['bifurcation']['mean_matched_distance_mm'],
            stats['endpoint']['mean_matched_distance_mm'],
        ]
        stds = [
            stats['origin']['std_avg_spread_mm'],
            stats['bifurcation']['std_matched_distance_mm'],
            stats['endpoint']['std_matched_distance_mm'],
        ]
        ax.bar(x + i * width, means, width, yerr=stds, label=artery, capsize=3)

    ax.set_xticks(x + width * (len(artery_types) - 1) / 2)
    ax.set_xticklabels(node_types)
    ax.set_ylabel('Distance (mm)')
    ax.set_title('Mean Node Distance by Type and Artery')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir, f'summary_distances.{figure_format}'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # --- Match rate breakdown by model (stacked count bars) ---
    for artery in artery_types:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_idx, node_type in enumerate(['bifurcation', 'endpoint']):
            ax = axes[ax_idx]
            ns = aggregate[artery][node_type]

            all_models = sorted(ns.get('total_per_model', {}).keys())
            if not all_models:
                continue

            full_counts = []
            partial_counts = []
            unmatched_counts = []
            totals = []

            for m in all_models:
                total = ns['total_per_model'].get(m, 0)
                full = ns.get('full_match_per_model', {}).get(m, 0)
                partial = ns.get('partial_match_per_model', {}).get(m, 0)
                unmatched = ns['total_unmatched_per_model'].get(m, 0)
                full_counts.append(full)
                partial_counts.append(partial)
                unmatched_counts.append(unmatched)
                totals.append(total)

            x = np.arange(len(all_models))
            width = 0.6

            ax.bar(x, full_counts, width, label='Full agreement', color='#2ca02c')
            ax.bar(x, partial_counts, width, bottom=full_counts,
                   label='Partial agreement', color='#ff7f0e')
            ax.bar(x, unmatched_counts, width,
                   bottom=[f + p for f, p in zip(full_counts, partial_counts)],
                   label='Unmatched', color='#d62728', alpha=0.7)

            # Annotate: percentage on the largest segment, n= above bar
            for i, m in enumerate(all_models):
                total = totals[i]
                if total == 0:
                    continue

                full = full_counts[i]
                partial = partial_counts[i]
                unmatched = unmatched_counts[i]

                # Find largest segment and label it with its percentage
                segments = [
                    ('full', full, 0, full),
                    ('partial', partial, full, full + partial),
                    ('unmatched', unmatched, full + partial, total),
                ]
                largest = max(segments, key=lambda s: s[1])
                if largest[1] > 0:
                    mid_y = (largest[2] + largest[3]) / 2
                    pct = largest[1] / total * 100
                    ax.text(i, mid_y, f'{pct:.0f}%',
                            ha='center', va='center', fontsize=9, fontweight='bold')

                # Total count above bar
                ax.text(i, total * 1.02, f'n={total}',
                        ha='center', va='bottom', fontsize=8, color='gray')

            ax.set_xticks(x)
            ax.set_xticklabels(all_models, rotation=30, ha='right')
            ax.set_ylabel('Number of nodes')
            ax.set_title(f'{node_type.title()} Match Rate — {artery}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'match_rates_{artery}.{figure_format}'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_distance_box_plots(per_patient_results, output_dir=None, figure_format='png'):
    """
    Box plots of per-patient matched distances by artery type and node type.

    Args:
        per_patient_results: dict from pipeline
        output_dir: str or None
        figure_format: 'png' or 'pdf'
    """
    data_by_group = {}
    for (patient_id, artery_type), result in per_patient_results.items():
        for node_type in ['bifurcation', 'endpoint']:
            stats = result['clustering'][node_type]['stats']
            if stats['num_matched_clusters'] > 0:
                key = (artery_type, node_type)
                if key not in data_by_group:
                    data_by_group[key] = []
                data_by_group[key].append(stats['mean_matched_distance_mm'])

        # Origin spread
        key = (artery_type, 'origin')
        if key not in data_by_group:
            data_by_group[key] = []
        data_by_group[key].append(result['origin']['mean_spread_radius_mm'])

    if not data_by_group:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    box_data = []

    for artery in ['LCA', 'RCA']:
        for node_type in ['origin', 'bifurcation', 'endpoint']:
            key = (artery, node_type)
            if key in data_by_group:
                labels.append(f"{artery}\n{node_type}")
                box_data.append(data_by_group[key])

    if box_data:
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        colors = []
        for label in labels:
            if 'LCA' in label:
                colors.append('#1f77b4')
            else:
                colors.append('#d62728')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.set_ylabel('Distance (mm)')
    ax.set_title('Per-Patient Node Distance Distribution')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        fig.savefig(os.path.join(output_dir, f'distance_boxplots.{figure_format}'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_pairwise_heatmaps(all_pairwise, output_dir=None, figure_format='png'):
    """
    NxN heatmaps of mean nearest-neighbor distance between each pair of models.

    Args:
        all_pairwise: dict of {node_type: {(model_a, model_b): [distances]}}
        output_dir: str or None
        figure_format: 'png' or 'pdf'
    """
    for node_type, pair_dists in all_pairwise.items():
        if not pair_dists:
            continue

        # Collect all model labels
        models = set()
        for (a, b) in pair_dists.keys():
            models.add(a)
            models.add(b)
        models = sorted(models)
        n = len(models)
        model_idx = {m: i for i, m in enumerate(models)}

        matrix = np.full((n, n), np.nan)
        np.fill_diagonal(matrix, 0.0)

        for (a, b), distances in pair_dists.items():
            mean_d = float(np.mean(distances))
            i, j = model_idx[a], model_idx[b]
            matrix[i, j] = mean_d
            matrix[j, i] = mean_d

        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=9, color='black' if val < np.nanmax(matrix) * 0.6 else 'white')

        plt.colorbar(im, ax=ax, label='Mean NN Distance (mm)')
        ax.set_title(f'Pairwise Model Distance — {node_type.title()} Nodes')
        plt.tight_layout()

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'heatmap_{node_type}.{figure_format}'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_ground_truth_comparison(aggregate, ground_truth_label,
                                 output_dir=None, figure_format='png'):
    """
    Stacked bar chart comparing each model's nodes against the ground truth.

    For the ground truth bar: green = matched by at least one model, red = unmatched.
    For each model bar: green = matched GT node, orange = matched another model but
    not GT, red = unmatched by anything.

    Bar heights reflect actual node counts so differences in total nodes are visible.

    Args:
        aggregate: dict from pipeline aggregation (must contain 'ground_truth' key per artery)
        ground_truth_label: str — the ground truth model label
        output_dir: str or None
        figure_format: 'png' or 'pdf'
    """
    for artery in ['LCA', 'RCA']:
        if artery not in aggregate:
            continue
        gt_agg = aggregate[artery].get('ground_truth')
        if gt_agg is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, node_type in enumerate(['bifurcation', 'endpoint']):
            ax = axes[ax_idx]
            gt_data = gt_agg.get(node_type)
            if gt_data is None:
                continue

            gt_total = gt_data['gt_total']
            gt_matched = gt_data['gt_matched']
            gt_unmatched = gt_data['gt_unmatched']

            # Build bars: GT first, then each model sorted alphabetically
            models = sorted(gt_data['model_gt_matched'].keys())
            bar_labels = [ground_truth_label] + models
            n_bars = len(bar_labels)
            x = np.arange(n_bars)
            width = 0.6

            # GT bar segments
            gt_green = [gt_matched]
            gt_red = [gt_unmatched]

            # Model bar segments
            model_green = []  # matched GT
            model_orange = []  # matched other model only
            model_red = []  # unmatched
            model_totals = []

            for m in models:
                mg = gt_data['model_gt_matched'].get(m, 0)
                mo = gt_data['model_other_only'].get(m, 0)
                mu = gt_data['model_unmatched'].get(m, 0)
                model_green.append(mg)
                model_orange.append(mo)
                model_red.append(mu)
                model_totals.append(mg + mo + mu)

            # Combined arrays for all bars
            greens = gt_green + model_green
            oranges = [0] + model_orange  # GT has no orange segment
            reds = gt_red + model_red
            totals = [gt_total] + model_totals

            ax.bar(x, greens, width, label='Matched GT', color='#2ca02c')
            ax.bar(x, oranges, width, bottom=greens,
                   label='Matched other model (not GT)', color='#ff7f0e')
            ax.bar(x, reds, width,
                   bottom=[g + o for g, o in zip(greens, oranges)],
                   label='Unmatched', color='#d62728', alpha=0.7)

            # Annotate: percentage of largest segment + n= above
            for i in range(n_bars):
                total = totals[i]
                if total == 0:
                    continue

                segments = [
                    (greens[i], 0),
                    (oranges[i], greens[i]),
                    (reds[i], greens[i] + oranges[i]),
                ]
                largest_val, largest_bottom = max(segments, key=lambda s: s[0])
                if largest_val > 0:
                    mid_y = largest_bottom + largest_val / 2
                    pct = largest_val / total * 100
                    ax.text(i, mid_y, f'{pct:.0f}%',
                            ha='center', va='center', fontsize=9, fontweight='bold')

                ax.text(i, total * 1.02, f'n={total}',
                        ha='center', va='bottom', fontsize=8, color='gray')

            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=30, ha='right')
            ax.set_ylabel('Number of nodes')
            ax.set_title(f'{node_type.title()} — GT Comparison ({artery})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        fig.suptitle(f'Ground Truth Node Comparison — {artery}', fontsize=13)
        plt.tight_layout()

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'gt_comparison_{artery}.{figure_format}'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    # --- Dice scorecard heatmap ---
    for artery in ['LCA', 'RCA']:
        if artery not in aggregate:
            continue
        gt_agg = aggregate[artery].get('ground_truth')
        if gt_agg is None:
            continue

        # Collect all models
        models = set()
        for nt in ['bifurcation', 'endpoint']:
            models.update(gt_agg.get(nt, {}).get('model_dice', {}).keys())
        models = sorted(models)
        if not models:
            continue

        col_labels = ['Origin\nHit Rate', 'Bifurcation\nDice', 'Endpoint\nDice']
        matrix = np.zeros((len(models), len(col_labels)))

        for i, m in enumerate(models):
            origin_data = gt_agg.get('origin', {}).get(m, {})
            matrix[i, 0] = origin_data.get('hit_rate', 0.0)
            matrix[i, 1] = gt_agg.get('bifurcation', {}).get('model_dice', {}).get(m, 0.0)
            matrix[i, 2] = gt_agg.get('endpoint', {}).get('model_dice', {}).get(m, 0.0)

        fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 2), max(3, len(models) * 0.8 + 1)))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(models)))
        ax.set_xticklabels(col_labels, fontsize=10)
        ax.set_yticklabels(models, fontsize=10)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(col_labels)):
                val = matrix[i, j]
                text_color = 'white' if val < 0.4 or val > 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_color)

        plt.colorbar(im, ax=ax, label='Score (0-1)', shrink=0.8)
        ax.set_title(f'Ground Truth Scorecard — {artery}', fontsize=13, pad=10)
        plt.tight_layout()

        if output_dir:
            fig.savefig(os.path.join(output_dir, f'gt_scorecard_{artery}.{figure_format}'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
