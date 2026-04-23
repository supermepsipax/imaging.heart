import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

MODEL_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#17becf',
    '#e377c2', '#bcbd22', '#8c564b', '#7f7f7f',
]


def plot_traversal_summary(aggregate, per_patient_results,
                           output_dir=None, figure_format='png'):
    """
    Generate summary figures for the traversal comparison.

    Per artery type, produces a 3-panel figure:
      1. Bifurcation accuracy and recall per model (grouped bars)
      2. Junction position error per model (box plot of individual errors)
      3. Origin offset per model (bar chart)

    Plus a separate CV chart if there are 2+ models.

    Args:
        aggregate: {artery_type: {'per_model': {...}, 'cv_across_models': {...}}}
        per_patient_results: {(patient, artery): {model: comparison_result}}
        output_dir: save path (None = plt.show())
        figure_format: 'png' or 'pdf'
    """
    for artery_type in ['LCA', 'RCA']:
        if artery_type not in aggregate:
            continue

        per_model = aggregate[artery_type]['per_model']
        cv_across = aggregate[artery_type].get('cv_across_models', {})
        model_labels = sorted(per_model.keys())
        if not model_labels:
            continue

        model_colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)]
                        for i, m in enumerate(model_labels)}

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'Traversal Comparison — {artery_type}  '
                     f'(valid-origin cases only)',
                     fontsize=13, fontweight='bold')

        # ==============================================================
        # Panel 1: Bifurcation Accuracy & Recall
        # ==============================================================
        ax = axes[0]
        x = np.arange(len(model_labels))
        width = 0.35

        accs = [per_model[m]['mean_bifurcation_accuracy'] for m in model_labels]
        acc_stds = [per_model[m]['std_bifurcation_accuracy'] for m in model_labels]
        recs = [per_model[m]['mean_bifurcation_recall'] for m in model_labels]
        rec_stds = [per_model[m]['std_bifurcation_recall'] for m in model_labels]

        bars_acc = ax.bar(x - width / 2, accs, width, yerr=acc_stds,
                          label='Accuracy (matched/GT)', color='#1f77b4',
                          capsize=3, alpha=0.85)
        bars_rec = ax.bar(x + width / 2, recs, width, yerr=rec_stds,
                          label='Recall (matched/model)', color='#ff7f0e',
                          capsize=3, alpha=0.85)

        for bar, val in zip(bars_acc, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
        for bar, val in zip(bars_rec, recs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha='right')
        ax.set_ylabel('Rate')
        ax.set_title('Bifurcation Detection')
        ax.set_ylim(0, 1.2)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        # ==============================================================
        # Panel 2: Junction Position Error (box plot)
        # ==============================================================
        ax = axes[1]
        error_data = []
        for m in model_labels:
            errors = per_model[m].get('all_junction_errors', [])
            error_data.append(errors if errors else [0.0])

        bp = ax.boxplot(error_data, labels=model_labels, patch_artist=True,
                        showfliers=True,
                        flierprops=dict(markersize=3, alpha=0.5),
                        medianprops=dict(color='black', linewidth=1.5))

        for patch, m in zip(bp['boxes'], model_labels):
            patch.set_facecolor(model_colors[m])
            patch.set_alpha(0.6)

        # Annotate median
        for i, m in enumerate(model_labels):
            errors = per_model[m].get('all_junction_errors', [])
            if errors:
                med = np.median(errors)
                ax.text(i + 1, med, f' {med:.1f}', ha='left', va='center',
                        fontsize=7, color='gray')

        ax.set_ylabel('Position Error (mm)')
        ax.set_title('Junction Position Error')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)

        # ==============================================================
        # Panel 3: Origin Offset
        # ==============================================================
        ax = axes[2]
        origin_means = [per_model[m]['mean_origin_offset_mm'] for m in model_labels]
        origin_stds = [per_model[m]['std_origin_offset_mm'] for m in model_labels]

        bars = ax.bar(x, origin_means, 0.6, yerr=origin_stds,
                      color=[model_colors[m] for m in model_labels],
                      capsize=3, alpha=0.85, edgecolor='gray', linewidth=0.5)

        for bar, val in zip(bars, origin_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=20, ha='right')
        ax.set_ylabel('Distance (mm)')
        ax.set_title('Origin Offset from GT')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if output_dir:
            path = os.path.join(output_dir,
                                f'traversal_summary_{artery_type}.{figure_format}')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        # ==============================================================
        # CV chart (separate figure)
        # ==============================================================
        if cv_across:
            _plot_cv_chart(cv_across, artery_type, output_dir, figure_format)


def _plot_cv_chart(cv_across_models, artery_type, output_dir=None,
                   figure_format='png'):
    """
    Horizontal bar chart of CV across models per metric.
    Higher CV = more disagreement between models on that metric.
    """
    display_names = {
        'mean_bifurcation_accuracy': 'Bifurcation Accuracy',
        'mean_bifurcation_recall': 'Bifurcation Recall',
        'mean_origin_offset_mm': 'Origin Offset',
        'mean_junction_error_mm': 'Junction Error',
    }

    metrics = [m for m in display_names if m in cv_across_models]
    if not metrics:
        return

    labels = [display_names[m] for m in metrics]
    values = [cv_across_models[m] for m in metrics]

    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    max_cv = max(values) if max(values) > 0 else 1.0
    colors = [plt.cm.RdYlGn_r(v / max_cv * 0.8 + 0.1) for v in values]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='gray',
                   linewidth=0.5, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', ha='left', va='center', fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Coefficient of Variation (std / mean)')
    ax.set_title(f'Inter-Model Variability — {artery_type}\n'
                 f'(higher = more disagreement between models)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if output_dir:
        path = os.path.join(output_dir,
                            f'traversal_cv_{artery_type}.{figure_format}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
