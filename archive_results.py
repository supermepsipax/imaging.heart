#!/usr/bin/env python3
"""
Utility script to archive artery analysis results into a compressed tar file.

Creates a tar archive of individual .pkl files without loading all into memory.
Works cross-platform and supports streaming analysis without unpacking.

Usage:
    # Archive all results from a folder (gzip compression, default)
    python archive_results.py results/batch_1 results/batch_1_archive.tar.gz

    # Use bz2 compression (better ratio, slower)
    python archive_results.py results/batch_1 results/batch_1_archive.tar.bz2 --compression bz2

    # Use xz compression (best ratio, slowest)
    python archive_results.py results/batch_1 results/batch_1_archive.tar.xz --compression xz

    # Custom file pattern
    python archive_results.py results/batch_1 results/batch_1_archive.tar.gz --pattern "*_analysis.pkl"
"""

import argparse
import tarfile
from pathlib import Path
from utilities import archive_artery_analyses


def verify_tar_archive(tar_path):
    """Verify tar archive can be opened and list contents."""
    try:
        with tarfile.open(tar_path, 'r:*') as tar:
            members = tar.getmembers()
            pkl_files = [m for m in members if m.name.endswith('.pkl') and m.isfile()]
            return len(pkl_files), pkl_files
    except Exception as e:
        raise Exception(f"Failed to verify archive: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Archive artery analysis pickle files into compressed tar (memory-efficient)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Archive all .pkl files with gzip compression (fast, ~5-10x smaller)
  python archive_results.py results/batch_1 results/batch_1_archive.tar.gz

  # Use maximum compression (slowest, ~10-20x smaller)
  python archive_results.py results/batch_1 results/batch_1_archive.tar.xz --compression xz

  # Archive only specific files
  python archive_results.py results/batch_1 results/batch_1_archive.tar.gz --pattern "Normal_*_analysis.pkl"

Compression methods:
  gz   - gzip: Fast, good compression (~5-10x smaller) [default]
  bz2  - bzip2: Slower, better compression (~8-15x smaller)
  xz   - lzma: Slowest, best compression (~10-20x smaller)

Benefits:
  - Memory-efficient: Streams files without loading all into memory
  - Cross-platform: Works on Windows, Mac, and Linux
  - Streaming analysis: Use analyze_artery_batch(input_tar_file=...) to analyze
    without unpacking the archive
        """
    )

    parser.add_argument('input_folder', type=str,
                        help='Folder containing individual .pkl files')
    parser.add_argument('output_file', type=str,
                        help='Output path for tar archive (e.g., results.tar.gz)')
    parser.add_argument('--pattern', type=str, default='*.pkl',
                        help='Glob pattern to match files (default: *.pkl)')
    parser.add_argument('--compression', type=str, default='gz',
                        choices=['gz', 'bz2', 'xz'],
                        help='Compression method (default: gz)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify archive after creation')

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
    expected_ext = f'.tar.{args.compression}'
    if args.compression == 'xz':
        expected_ext = '.tar.xz'

    if not str(output_path).endswith(expected_ext):
        print(f"Warning: Output file should end with {expected_ext} for {args.compression} compression")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ARCHIVE ARTERY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Input folder:       {args.input_folder}")
    print(f"Output file:        {args.output_file}")
    print(f"File pattern:       {args.pattern}")
    print(f"Compression:        {args.compression}")
    print("=" * 80)
    print()

    # Archive files
    try:
        stats = archive_artery_analyses(
            input_folder=args.input_folder,
            output_file=args.output_file,
            pattern=args.pattern,
            compression=args.compression
        )

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Files archived:     {stats['files_archived']}")
        print(f"Output size:        {stats['output_size_mb']:.2f} MB")

        # Optionally verify the archive
        if args.verify:
            print()
            print("Verifying archive...")
            try:
                num_files, pkl_files = verify_tar_archive(args.output_file)

                print(f"✓ Verification passed")
                print(f"  Files in archive: {num_files}")

                # Show structure preview
                print()
                print("Archive contents preview:")
                for i, member in enumerate(pkl_files[:5]):
                    print(f"  {member.name} ({member.size / 1024 / 1024:.1f} MB)")
                if len(pkl_files) > 5:
                    print(f"  ... ({len(pkl_files) - 5} more files)")

            except Exception as e:
                print(f"✗ Verification failed: {e}")
                return 1

        print()
        print("To analyze this archive without unpacking:")
        print(f"  from pipelines import analyze_artery_batch")
        print(f"  analyze_artery_batch(input_tar_file='{args.output_file}')")
        print("=" * 80)
        print()

        return 0

    except Exception as e:
        print()
        print(f"Error during archive creation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
