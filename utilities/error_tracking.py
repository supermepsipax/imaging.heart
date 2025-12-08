"""
Error tracking and validation for coronary artery analysis pipeline.

This module provides error tracking, validation, and logging functionality to ensure
only high-quality vessel analyses are saved to output files.
"""

import os
import networkx as nx
from datetime import datetime


class VesselErrorTracker:
    """
    Tracks errors, warnings, and validation results for a single vessel during processing.

    Supports three error levels:
    - CRITICAL: Prevents saving results (e.g., graph fragmentation, too many unlabeled edges)
    - ERROR: Serious issue logged but allows saving (e.g., no branch pattern detected)
    - WARNING: Informational, logged for review (e.g., low bifurcation count)
    """

    def __init__(self, filename, vessel_label=None):
        """
        Initialize error tracker for a vessel.

        Args:
            filename (str): Name of the NRRD file being processed
            vessel_label (str, optional): 'LCA' or 'RCA' or None if not yet classified
        """
        self.filename = filename
        self.vessel_label = vessel_label
        self.errors = {
            'CRITICAL': [],
            'ERROR': [],
            'WARNING': []
        }
        self.metrics = {}
        self.validation_results = {}

    def log_critical(self, step, message):
        """
        Log a critical error that prevents saving results.

        Args:
            step (str): Pipeline step where error occurred (e.g., "Bypass Removal")
            message (str): Detailed error message
        """
        self.errors['CRITICAL'].append({'step': step, 'message': message})

    def log_error(self, step, message):
        """
        Log an error that allows saving but needs attention.

        Args:
            step (str): Pipeline step where error occurred
            message (str): Detailed error message
        """
        self.errors['ERROR'].append({'step': step, 'message': message})

    def log_warning(self, step, message):
        """
        Log a warning for informational purposes.

        Args:
            step (str): Pipeline step where warning occurred
            message (str): Detailed warning message
        """
        self.errors['WARNING'].append({'step': step, 'message': message})

    def add_metric(self, key, value):
        """Add a metric for validation summary."""
        self.metrics[key] = value

    def add_validation_result(self, key, passed, details=None):
        """
        Add a validation result.

        Args:
            key (str): Validation check name
            passed (bool): Whether validation passed
            details (str, optional): Additional details about the check
        """
        self.validation_results[key] = {
            'passed': passed,
            'details': details
        }

    def has_critical_errors(self):
        """Returns True if any critical errors have been logged."""
        return len(self.errors['CRITICAL']) > 0

    def should_save(self):
        """Returns True if vessel should be saved (no critical errors)."""
        return not self.has_critical_errors()

    def get_status(self):
        """
        Get overall status of vessel processing.

        Returns:
            str: 'SUCCESS', 'WARNING', or 'FAILED'
        """
        if self.has_critical_errors():
            return 'FAILED'
        elif len(self.errors['ERROR']) > 0 or len(self.errors['WARNING']) > 0:
            return 'WARNING'
        else:
            return 'SUCCESS'

    def get_summary(self):
        """
        Generate a formatted summary of all errors, warnings, and validation results.

        Returns:
            str: Multi-line formatted summary
        """
        lines = []

        vessel_name = self.vessel_label if self.vessel_label else "Vessel"
        lines.append(f"\nVESSEL: {vessel_name}")
        lines.append("")

        # Log critical errors
        if self.errors['CRITICAL']:
            for error in self.errors['CRITICAL']:
                lines.append(f"  [CRITICAL] {error['step']}:")
                for line in error['message'].split('\n'):
                    lines.append(f"    → {line}")
                lines.append("")

        # Log errors
        if self.errors['ERROR']:
            for error in self.errors['ERROR']:
                lines.append(f"  [ERROR] {error['step']}:")
                for line in error['message'].split('\n'):
                    lines.append(f"    → {line}")
                lines.append("")

        # Log warnings
        if self.errors['WARNING']:
            for warning in self.errors['WARNING']:
                lines.append(f"  [WARNING] {warning['step']}:")
                for line in warning['message'].split('\n'):
                    lines.append(f"    → {line}")
                lines.append("")

        # Success message if no errors/warnings
        if not self.errors['CRITICAL'] and not self.errors['ERROR'] and not self.errors['WARNING']:
            lines.append("  [SUCCESS] All checks passed")
            lines.append("")

        # Validation summary
        if self.validation_results:
            lines.append("  VALIDATION SUMMARY:")
            for key, result in self.validation_results.items():
                status = "✓" if result['passed'] else "✗"
                line = f"    {status} {key}: {'PASS' if result['passed'] else 'FAIL'}"
                if result['details']:
                    line += f" ({result['details']})"
                lines.append(line)
            lines.append("")

        # Action taken
        if self.should_save():
            lines.append(f"  ACTION: ✅ {vessel_name} results saved successfully")
        else:
            lines.append(f"  ACTION: ❌ {vessel_name} results NOT SAVED due to critical errors")

        return '\n'.join(lines)


class BatchErrorLogger:
    """
    Manages error logging for entire batch processing run.

    Accumulates vessel error trackers and writes comprehensive error log at end.
    """

    def __init__(self, output_folder, config):
        """
        Initialize batch error logger.

        Args:
            output_folder (str): Directory where log file will be written
            config (dict): Pipeline configuration dictionary
        """
        self.output_folder = output_folder
        self.config = config
        self.start_time = datetime.now()
        self.vessel_trackers = []
        self.file_results = {}  # filename -> {'status': ..., 'trackers': [...]}

        # Create log filename with timestamp
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"batch_error_log_{timestamp}.txt"
        self.log_path = os.path.join(output_folder, self.log_filename)

    def add_file_result(self, filename, status, reason=None, trackers=None):
        """
        Add result for a file.

        Args:
            filename (str): Name of the NRRD file
            status (str): 'success', 'partial', 'failed'
            reason (str, optional): Failure reason if status is 'failed'
            trackers (list, optional): List of VesselErrorTrackers for this file
        """
        self.file_results[filename] = {
            'status': status,
            'reason': reason,
            'trackers': trackers or []
        }

        if trackers:
            self.vessel_trackers.extend(trackers)

    def get_summary_stats(self):
        """
        Calculate batch-level summary statistics.

        Returns:
            dict: Summary statistics including counts, percentages, etc.
        """
        total_files = len(self.file_results)
        success_files = sum(1 for r in self.file_results.values() if r['status'] == 'success')
        partial_files = sum(1 for r in self.file_results.values() if r['status'] == 'partial')
        failed_files = sum(1 for r in self.file_results.values() if r['status'] == 'failed')

        total_vessels = len(self.vessel_trackers)
        saved_vessels = sum(1 for t in self.vessel_trackers if t.should_save())

        critical_errors = sum(len(t.errors['CRITICAL']) for t in self.vessel_trackers)
        errors = sum(len(t.errors['ERROR']) for t in self.vessel_trackers)
        warnings = sum(len(t.errors['WARNING']) for t in self.vessel_trackers)

        return {
            'total_files': total_files,
            'success_files': success_files,
            'partial_files': partial_files,
            'failed_files': failed_files,
            'total_vessels': total_vessels,
            'saved_vessels': saved_vessels,
            'critical_errors': critical_errors,
            'errors': errors,
            'warnings': warnings
        }

    def write_log(self):
        """Write comprehensive error log to file."""
        with open(self.log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BATCH PROCESSING ERROR LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Folder: {self.output_folder}\n")
            f.write(f"Total Files: {len(self.file_results)}\n")
            f.write("\nConfiguration:\n")

            # Write key config parameters
            config_keys = [
                'bypass_threshold', 'max_y_branch_length_voxels',
                'lca_trifurcation_threshold_mm', 'max_unlabeled_edges'
            ]
            for key in config_keys:
                if key in self.config:
                    f.write(f"  - {key}: {self.config[key]}\n")

            f.write("=" * 80 + "\n\n")

            # Write file-by-file results
            for filename, result in self.file_results.items():
                f.write("━" * 80 + "\n")
                f.write(f"FILE: {filename}\n")

                status_symbol = {
                    'success': '✅ SUCCESS',
                    'partial': '⚠️  PARTIAL SUCCESS',
                    'failed': '❌ FAILED (CRITICAL ERRORS)'
                }
                f.write(f"STATUS: {status_symbol.get(result['status'], result['status'])}\n")
                f.write("━" * 80 + "\n")

                if result['status'] == 'failed' and result['reason']:
                    f.write(f"\n[CRITICAL] {result['reason']}\n")
                    f.write("RESULT: No output files generated\n")
                elif result['trackers']:
                    # Write each vessel's error summary
                    for tracker in result['trackers']:
                        f.write(tracker.get_summary())
                        f.write("\n")

                    # Overall result for this file
                    saved_count = sum(1 for t in result['trackers'] if t.should_save())
                    total_count = len(result['trackers'])

                    if saved_count == total_count:
                        f.write(f"\nRESULT: {saved_count}/{total_count} vessels saved successfully\n")
                    elif saved_count == 0:
                        f.write(f"\nRESULT: No vessels saved (all failed validation)\n")
                    else:
                        saved_labels = [t.vessel_label for t in result['trackers'] if t.should_save()]
                        f.write(f"\nRESULT: {saved_count}/{total_count} vessels saved ({', '.join(saved_labels)})\n")

                f.write("\n" + "─" * 80 + "\n\n")

            # Write batch summary
            end_time = datetime.now()
            duration = end_time - self.start_time
            stats = self.get_summary_stats()

            f.write("=" * 80 + "\n")
            f.write("BATCH SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {duration.total_seconds():.1f}s ({duration.total_seconds()/60:.1f}m)\n\n")

            f.write(f"Files Processed: {stats['total_files']}\n")
            f.write(f"  ✅ Success: {stats['success_files']}\n")
            f.write(f"  ⚠️  Partial: {stats['partial_files']}\n")
            f.write(f"  ❌ Failed: {stats['failed_files']}\n\n")

            if stats['total_vessels'] > 0:
                save_pct = (stats['saved_vessels'] / stats['total_vessels']) * 100
                f.write(f"Vessels Saved: {stats['saved_vessels']}/{stats['total_vessels']} ({save_pct:.1f}%)\n\n")

            f.write(f"Critical Errors Encountered: {stats['critical_errors']}\n")
            f.write(f"Errors (Non-Critical): {stats['errors']}\n")
            f.write(f"Warnings: {stats['warnings']}\n")

            f.write("=" * 80 + "\n")

        print(f"\n[Error Log] Written to: {self.log_path}")
        return self.log_path


def validate_graph_structure(graph, tracker, step_name):
    """
    Validate that graph has proper structure (connected, non-empty).

    Args:
        graph (networkx.Graph): Graph to validate
        tracker (VesselErrorTracker): Error tracker to log issues
        step_name (str): Name of pipeline step for error messages

    Returns:
        tuple: (is_valid, should_continue)
            is_valid: True if graph passes basic structure checks
            should_continue: True if processing can continue
    """
    is_valid = True
    should_continue = True

    # Check if graph is empty
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    if num_edges == 0:
        tracker.log_critical(step_name, f"Graph has 0 edges (empty graph)")
        tracker.add_validation_result("Graph non-empty", False, "0 edges")
        is_valid = False
        should_continue = False
        return is_valid, should_continue

    # Check connectivity
    num_components = nx.number_connected_components(graph)
    if num_components > 1:
        components = list(nx.connected_components(graph))
        component_sizes = [len(c) for c in components]

        message = f"Graph has {num_components} disconnected components\n"
        for i, size in enumerate(component_sizes):
            message += f"Component {i+1}: {size} nodes"
            if i < len(component_sizes) - 1:
                message += "\n"

        tracker.log_critical(step_name, message)
        tracker.add_validation_result("Graph connectivity", False, f"{num_components} components")
        is_valid = False
        should_continue = False
    else:
        tracker.add_validation_result("Graph connectivity", True, "1 component")

    # Add metrics
    tracker.add_metric('num_nodes', num_nodes)
    tracker.add_metric('num_edges', num_edges)
    tracker.add_metric('num_components', num_components)

    return is_valid, should_continue


def validate_anatomical_labels(graph, vessel_type, tracker, max_unlabeled=1):
    """
    Validate anatomical labeling coverage (count-based, not percentage).

    For LCA: Checks 'lca_branch' labels
    For RCA: Checks 'rca_branch' labels

    Args:
        graph (networkx.DiGraph): Directed graph with anatomical labels
        vessel_type (str): 'LCA' or 'RCA'
        tracker (VesselErrorTracker): Error tracker to log issues
        max_unlabeled (int): Maximum number of unlabeled edges allowed (default: 1)

    Returns:
        tuple: (is_valid, unlabeled_edges)
            is_valid: True if unlabeled edge count is within threshold
            unlabeled_edges: List of edge tuples without anatomical labels
    """
    label_attr = 'lca_branch' if vessel_type == 'LCA' else 'rca_branch'

    unlabeled_edges = []
    main_branches_found = set()

    for edge in graph.edges():
        label = graph.edges[edge].get(label_attr)

        if label is None or label == '':
            unlabeled_edges.append(edge)
        else:
            # Track main branches found
            if vessel_type == 'LCA':
                if label in ['LAD', 'LCx', 'Ramus', 'Left_Main']:
                    main_branches_found.add(label)
            elif vessel_type == 'RCA':
                if label == 'RCA':
                    main_branches_found.add(label)

    unlabeled_count = len(unlabeled_edges)
    total_edges = graph.number_of_edges()

    # Check if unlabeled count exceeds threshold
    is_valid = unlabeled_count <= max_unlabeled

    if not is_valid:
        message = f"Too many unlabeled edges: {unlabeled_count} unlabeled (threshold: max {max_unlabeled})\n"
        message += f"Total edges: {total_edges}\n"
        message += "Unlabeled edges:"

        # Show up to 10 unlabeled edges
        for i, edge in enumerate(unlabeled_edges[:10]):
            message += f"\n   - {edge}"
        if len(unlabeled_edges) > 10:
            message += f"\n   ... ({len(unlabeled_edges) - 10} more)"

        tracker.log_critical("Anatomical Labeling", message)
        tracker.add_validation_result(
            "Anatomical labeling",
            False,
            f"{unlabeled_count} unlabeled edges, max allowed: {max_unlabeled}"
        )
    else:
        tracker.add_validation_result(
            "Anatomical labeling",
            True,
            f"{unlabeled_count} unlabeled edges" if unlabeled_count > 0 else "All edges labeled"
        )

    # Check for main branches
    if vessel_type == 'LCA':
        has_lad = 'LAD' in main_branches_found
        has_lcx = 'LCx' in main_branches_found

        if not has_lad or not has_lcx:
            missing = []
            if not has_lad:
                missing.append('LAD')
            if not has_lcx:
                missing.append('LCx')

            tracker.log_critical(
                "Anatomical Labeling",
                f"Missing main branches: {', '.join(missing)}"
            )
            tracker.add_validation_result("Main branches found", False, f"Missing: {', '.join(missing)}")
            is_valid = False
        else:
            branches_str = ', '.join(sorted(main_branches_found))
            tracker.add_validation_result("Main branches found", True, branches_str)

    elif vessel_type == 'RCA':
        if 'RCA' not in main_branches_found:
            tracker.log_critical("Anatomical Labeling", "No RCA trunk labeled")
            tracker.add_validation_result("Main branches found", False, "No RCA trunk")
            is_valid = False
        else:
            tracker.add_validation_result("Main branches found", True, "RCA trunk present")

    # Add metrics
    tracker.add_metric('unlabeled_edges', unlabeled_count)
    tracker.add_metric('total_edges', total_edges)
    tracker.add_metric('main_branches', list(main_branches_found))

    return is_valid, unlabeled_edges


def validate_lca_branch_length_ratio(lca_graph, spacing_info, tracker, min_ratio=0.1,
                                     log_critical=True):
    """
    Validate that LCA main branches (LAD and LCx) have reasonable length ratios.

    Checks if the ratio of the shorter branch to the longer branch meets the minimum threshold.
    This helps detect cases where LCA/RCA might be misidentified (e.g., if LCx is extremely short,
    it might actually be an RCA branch).

    Args:
        lca_graph (networkx.DiGraph): LCA graph with branch labels
        spacing_info (tuple): Voxel spacing (z, y, x) in mm
        tracker (VesselErrorTracker): Error tracker for logging issues
        min_ratio (float): Minimum acceptable ratio (shorter/longer), default 0.1 (10%)
        log_critical (bool): Whether to log critical errors on failure (default True).
                            Set to False when testing before attempting a swap.

    Returns:
        tuple: (is_valid, lad_length, lcx_length, ratio)
            - is_valid: True if ratio meets threshold
            - lad_length: LAD total path length in mm (or None if not found)
            - lcx_length: LCx total path length in mm (or None if not found)
            - ratio: min/max length ratio (or None if branches not found)
    """
    from analysis.branch_statistics import extract_main_branch_statistics

    # Extract main branch statistics for LCA
    try:
        branch_stats = extract_main_branch_statistics(lca_graph, spacing_info, artery_type='LCA')
    except Exception as e:
        tracker.log_critical(
            "Branch Length Validation",
            f"Failed to extract main branch statistics: {str(e)}"
        )
        return False, None, None, None

    # Check if LAD and LCx exist
    if 'LAD' not in branch_stats or 'LCx' not in branch_stats:
        missing = []
        if 'LAD' not in branch_stats:
            missing.append('LAD')
        if 'LCx' not in branch_stats:
            missing.append('LCx')

        tracker.log_critical(
            "Branch Length Validation",
            f"Cannot validate length ratio - missing branches: {', '.join(missing)}"
        )
        return False, None, None, None

    lad_length = branch_stats['LAD']['total_path_length']
    lcx_length = branch_stats['LCx']['total_path_length']

    # Compute ratio (shorter / longer)
    min_length = min(lad_length, lcx_length)
    max_length = max(lad_length, lcx_length)

    ratio = min_length / max_length if max_length > 0 else 0.0

    is_valid = ratio >= min_ratio

    if not is_valid:
        shorter_branch = 'LAD' if lad_length < lcx_length else 'LCx'

        # Only log critical error if requested (not during initial test before swap)
        if log_critical:
            tracker.log_critical(
                "Branch Length Validation",
                f"LCA branch length ratio failed: {shorter_branch} is too short "
                f"(LAD={lad_length:.1f}mm, LCx={lcx_length:.1f}mm, ratio={ratio:.3f}, threshold={min_ratio:.3f}). "
                f"Possible LCA/RCA misclassification."
            )

        tracker.add_validation_result(
            "LCA branch length ratio",
            False,
            f"{shorter_branch} too short: ratio={ratio:.3f} < {min_ratio:.3f}"
        )
    else:
        tracker.add_validation_result(
            "LCA branch length ratio",
            True,
            f"LAD={lad_length:.1f}mm, LCx={lcx_length:.1f}mm, ratio={ratio:.3f}"
        )

    # Add metrics
    tracker.add_metric('lad_length_mm', lad_length)
    tracker.add_metric('lcx_length_mm', lcx_length)
    tracker.add_metric('lad_lcx_ratio', ratio)

    return is_valid, lad_length, lcx_length, ratio
