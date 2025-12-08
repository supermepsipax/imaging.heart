#!/usr/bin/env python3
"""
Utility script to compress and merge artery analysis results.

Merges individual .pkl files into a single compressed file while preserving
all data exactly as in the originals. Compression provides 3-10x space savings
with no information loss.

Usage:
    # Merge all results from a folder (gzip compression, default)
    python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.gz

    # Use bz2 compression (better ratio, slower)
    python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.bz2 --compression bz2

    # Use lzma compression (best ratio, slowest)
    python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.xz --compression lzma

    # Custom file pattern
    python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.gz --pattern "*_analysis.pkl"
"""

import argparse
from pathlib import Path
from utilities import merge_artery_analyses, load_artery_analysis_compressed


def main():
    parser = argparse.ArgumentParser(
        description='Compress and merge artery analysis pickle files (preserves all data)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all .pkl files with gzip compression (fast, ~5-10x smaller)
  python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.gz

  # Use maximum compression (slowest, ~10-20x smaller)
  python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.xz --compression lzma

  # Merge only specific files
  python compress_and_merge_results.py results/batch_1 results/batch_1_merged.pkl.gz --pattern "Normal_*_analysis.pkl"

Compression methods:
  gzip  - Fast, good compression (~5-10x smaller) [default]
  bz2   - Slower, better compression (~8-15x smaller)
  lzma  - Slowest, best compression (~10-20x smaller)

Note: Merged files contain ALL original data - when extracted, they are
      identical to the individual .pkl files.
        """
    )

    parser.add_argument('input_folder', type=str,
                        help='Folder containing individual .pkl files')
    parser.add_argument('output_file', type=str,
                        help='Output path for merged file (e.g., results.pkl.gz)')
    parser.add_argument('--pattern', type=str, default='*.pkl',
                        help='Glob pattern to match files (default: *.pkl)')
    parser.add_argument('--compression', type=str, default='gzip',
                        choices=['gzip', 'bz2', 'lzma'],
                        help='Compression method (default: gzip)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify merged file after creation')

    args = parser.parse_args()

    # Validate input folder exists
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return 1

    if not input_path.is_dir():
        print(f"Error: Input path is not a folder: {args.input_folder}")
        return 1

    # Ensure output file has correct extension
    output_path = Path(args.output_file)
    if args.compression == 'gzip' and not str(output_path).endswith('.gz'):
        print(f"Warning: Output file should end with .gz for gzip compression")
    elif args.compression == 'bz2' and not str(output_path).endswith('.bz2'):
        print(f"Warning: Output file should end with .bz2 for bz2 compression")
    elif args.compression == 'lzma' and not str(output_path).endswith(('.xz', '.lzma')):
        print(f"Warning: Output file should end with .xz or .lzma for lzma compression")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COMPRESS AND MERGE ARTERY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Input folder:       {args.input_folder}")
    print(f"Output file:        {args.output_file}")
    print(f"File pattern:       {args.pattern}")
    print(f"Compression:        {args.compression}")
    print("=" * 80)
    print()

    # Merge files
    try:
        stats = merge_artery_analyses(
            input_folder=args.input_folder,
            output_file=args.output_file,
            pattern=args.pattern,
            compression=args.compression
        )

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Files merged:       {stats['files_merged']}")
        print(f"Total vessels:      {stats['total_vessels']}")
        print(f"Output size:        {stats['output_size_mb']:.2f} MB")

        # Optionally verify the merged file
        if args.verify:
            print()
            print("Verifying merged file...")
            try:
                merged_data = load_artery_analysis_compressed(args.output_file)
                num_cases = len(merged_data)
                total_vessels_verified = sum(len(vessels) for vessels in merged_data.values())

                print(f"✓ Verification passed")
                print(f"  Cases: {num_cases}")
                print(f"  Vessels: {total_vessels_verified}")

                # Show structure
                print()
                print("Structure preview:")
                for i, (case_name, vessels) in enumerate(list(merged_data.items())[:3]):
                    vessel_types = ', '.join(vessels.keys())
                    print(f"  {case_name}: {vessel_types}")
                    if i == 2 and len(merged_data) > 3:
                        print(f"  ... ({len(merged_data) - 3} more cases)")
                        break

            except Exception as e:
                print(f"✗ Verification failed: {e}")
                return 1

        print("=" * 80)
        print()

        return 0

    except Exception as e:
        print()
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
