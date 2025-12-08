"""
Example workflow showing how to create and use tar archives for memory-efficient batch processing.

This approach solves the problem of merging large .pkl files (e.g., 76 files × 700MB = ~53GB)
by creating a compressed tar archive that can be streamed during analysis without unpacking.

Works cross-platform (Windows/Mac/Linux) via Python's built-in tarfile module.
"""

from utilities import archive_artery_analyses
from pipelines import analyze_artery_batch

# ============================================================================
# Step 1: Create tar.gz archive from individual .pkl files
# ============================================================================
# This streams files into the archive WITHOUT loading them all into memory.
# Safe for datasets of any size.

print("Creating tar archive from batch results...")
stats = archive_artery_analyses(
    input_folder='results/batch_3',
    output_file='results/batch_3.tar.gz',
    pattern='*_analysis.pkl',
    compression='gz'  # Options: 'gz' (gzip), 'bz2', 'xz' (lzma)
)

print(f"\n✓ Archive created successfully!")
print(f"  Files: {stats['files_archived']}")
print(f"  Size: {stats['output_size_mb']:.1f} MB")
print(f"  Compression ratio: ~{(stats['files_archived'] * 700) / stats['output_size_mb']:.1f}x")

# ============================================================================
# Step 2: Run statistical analysis by streaming from archive
# ============================================================================
# This processes files ONE AT A TIME from the archive without unpacking.
# Only one file is in memory at a time - handles datasets of any size.

print("\n" + "=" * 80)
print("Running statistical analysis from tar archive...")
print("=" * 80)

results = analyze_artery_batch(
    input_tar_file='results/batch_3.tar.gz',  # Stream from archive
    diameter_method='slicing',
    verbose=True
)

print(f"\n✓ Analysis complete!")
print(f"  Processed: {results['processed_count']}/{results['total_files']}")
print(f"  Failed: {results['failed_count']}/{results['total_files']}")

# ============================================================================
# Alternative: Process with config file
# ============================================================================
# You can also specify input_tar_file in a YAML config:
#
# analysis_config.yaml:
#   input_tar_file: 'results/batch_3.tar.gz'
#   diameter_method: 'slicing'
#   verbose: true
#
# Then run:
# results = analyze_artery_batch(config_path='analysis_config.yaml')

# ============================================================================
# Notes on compression methods:
# ============================================================================
# - 'gz' (gzip):  Fast compression/decompression, good ratio (~5-10x)
# - 'bz2':        Slower, better compression (~8-15x)
# - 'xz' (lzma):  Slowest, best compression (~10-20x)
#
# Recommendation: Use 'gz' for most cases (best balance of speed and size)
