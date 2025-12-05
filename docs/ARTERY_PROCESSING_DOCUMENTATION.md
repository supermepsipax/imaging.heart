# Coronary Artery Processing Pipeline - Technical Documentation

## Overview

This system provides automated processing and analysis of coronary artery vasculature from medical imaging data. The pipeline transforms 3D binary masks in NRRD format into structured graph representations containing geometric and morphological measurements of the arterial tree. The processing includes centerline extraction, graph construction, diameter profiling, bifurcation angle computation, and anatomical branch labeling. Output data is provided in both human-readable CSV format and pickle format for subsequent statistical analysis.

## Environment Setup

### Creating a Development Environment

The codebase requires Python 3.8 or higher. Begin by creating an isolated virtual environment to manage dependencies:

```bash
cd /path/to/IMG-Heart
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Installing Dependencies

All required packages are specified in `requirements.txt`. The primary dependencies include NumPy for numerical operations, SciPy for scientific computing functions, scikit-image for image processing and skeletonization, NetworkX for graph data structures, pynrrd for NRRD file handling, and matplotlib for visualization. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Configuration File Structure

The pipeline reads parameters from a configuration file (`config.yaml` or `config.json`). This file controls preprocessing behavior, diameter computation parameters, and output settings. Both YAML and JSON formats are supported, with YAML recommended for its native comment support. A typical configuration contains the following parameters:

**Preprocessing Parameters**: The `upsample_factor` (default: 1) controls spatial upsampling of the input mask before processing. Higher values increase computational cost but may improve accuracy for low-resolution data.

**Diameter Computation Parameters**: The `min_depth_mm` and `max_depth_mm` (defaults: 2.0 and 7.0) define the range of distances from bifurcation points over which angles are computed and averaged. The `step_mm` parameter (default: 0.5) sets the increment for sampling along branches during angle computation.

**Graph Cleaning Parameters**: The `remove_bypass` flag (default: true) enables detection and removal of spurious bypass edges that can occur at high-degree nodes in the skeleton. The `bypass_threshold` (default: 2.0) sets the maximum Euclidean distance in voxels for identifying bypass edges.

**Input/Output Configuration**: The `input_folder` specifies the directory containing NRRD mask files, while `output_folder` determines where results are saved. The optional `visualize` flag controls whether 3D visualizations are displayed during processing.

## Technical Pipeline Architecture

### Input Data and Initial Loading

The pipeline processes NRRD files containing binary 3D masks where voxel intensity 1 represents vessel tissue and 0 represents background. Each file includes metadata in the NRRD header, most critically the `space directions` matrix that encodes voxel spacing in millimeters for each dimension. The loading process extracts both the volumetric data array and this spacing information, which is essential for all subsequent physical measurements.

### Spatial Preprocessing

After loading, the binary mask undergoes several preprocessing steps to prepare it for analysis. First, the mask is resampled to isotropic voxel spacing using trilinear interpolation. This ensures that measurements are consistent across all spatial dimensions and simplifies downstream geometric calculations. The resampling interpolates the binary mask to the smallest voxel dimension from the original spacing, effectively creating cubic voxels.

Following resampling, optional upsampling can be applied if specified in the configuration. This uses nearest-neighbor interpolation to increase the spatial resolution by the upsample factor, which can improve the quality of subsequent skeletonization for lower resolution input data.

The preprocessing system also includes body detection, which identifies separate connected components in the binary mask. This is critical for files containing multiple vessels, such as combined LCA and RCA masks. The connected component labeling uses 26-connectivity in 3D space, and components are sorted by voxel count. For combined coronary artery datasets, the two largest components are extracted and processed independently.

### Distance Transform Computation

Before centerline extraction, the pipeline computes a Euclidean distance transform of the entire binary mask. This operation assigns to each foreground voxel its distance to the nearest background voxel, effectively measuring the local radius at every point in the vessel. The distance transform is computed with the physical spacing information, yielding distances in millimeters rather than voxels. This single computation is later reused during diameter measurements, avoiding redundant calculations and significantly improving computational efficiency.

### Centerline Extraction via Skeletonization

The centerline extraction employs morphological skeletonization to reduce the 3D vessel mask to a one-voxel-thick medial axis representation. The skeletonization algorithm from scikit-image implements a topology-preserving thinning operation that iteratively removes boundary voxels while maintaining connectivity and the overall structure of the vascular tree. The resulting skeleton preserves critical topological features including endpoints (vessel terminations) and bifurcation points (vessel splits).

### Bifurcation and Endpoint Detection

Critical points in the skeleton are identified by analyzing local connectivity. For each voxel in the skeleton, the system convolves the binary skeleton with a 3x3x3 kernel of ones, effectively counting the number of connected neighbors in the 26-connected neighborhood. Voxels with exactly one neighbor are classified as endpoints, representing vessel terminations. Voxels with three or more neighbors are classified as bifurcation points, representing locations where vessels split.

This initial detection often produces redundant bifurcations due to noise or slight geometric irregularities in the skeleton. The redundant bifurcation removal process identifies clusters of nearby bifurcation points and consolidates them into single representative points. For each cluster, the system selects the bifurcation point with the highest connectivity score, determined by counting skeleton voxels within a local neighborhood. This ensures that the most central and structurally significant point is retained.

An additional refinement removes false bifurcations caused by sharp bends in vessels. Sharp bends can produce skeleton configurations with three connected segments even though no true vessel branching occurs. The system detects these by analyzing the angles between connected segments at each bifurcation candidate. If the angle between any pair of segments is very acute and the third segment forms a continuation, the point is reclassified as a path point rather than a bifurcation.

### Graph Construction

The processed skeleton with refined bifurcation and endpoint sets is converted into a graph data structure. Each bifurcation point and endpoint becomes a node in the graph. Edges are constructed by tracing paths through the skeleton between these critical points. The tracing algorithm performs a breadth-first search from each critical point, following skeleton voxels until reaching another critical point, thereby defining an edge.

Each edge stores several attributes: the complete list of voxel coordinates forming the path, the Euclidean length of the path in physical coordinates, and positional information about the edge's relationship to its parent in the tree structure. This sparse graph representation maintains the topology of the vascular tree while reducing the data volume by orders of magnitude compared to the original mask.

### Bypass Edge Detection and Removal

Skeleton artifacts can create spurious bypass edges at bifurcation points, where a direct connection appears between what should be sequential segments. The bypass edge removal algorithm identifies these by examining high-degree nodes in the graph. For each node with degree greater than three, the system compares the spatial distances between the node and its connected neighbors against the path lengths of the connecting edges. If an edge's Euclidean distance is much shorter than its path length through the skeleton, and removing it would not disconnect the graph, the edge is classified as a bypass and removed.

This process preserves true anatomical connections while eliminating structural artifacts. The algorithm processes each component of the graph independently, ensuring that bypass removal in one region does not affect disconnected structures. Special handling is applied to endpoints, which are preserved regardless of their involvement in potential bypass configurations.

### Diameter Measurement

The system implements two complementary methods for vessel diameter measurement. The first method leverages the pre-computed distance transform. For each edge in the graph, the algorithm samples the distance transform values at every voxel along the edge's path. Since the distance transform represents the radius at each point, doubling these values yields diameter measurements. The mean and median diameters for each edge are computed from these samples, and the complete diameter profile is stored as a list of values along the vessel segment.

The second method computes diameters using orthogonal slice extraction. For each voxel along a vessel segment, the algorithm computes a smoothed tangent vector by averaging multiple directional vectors around that point. An orthogonal plane is then constructed perpendicular to this tangent, and a 2D slice of the binary mask is extracted by sampling the 3D volume along this plane. The vessel cross-section in this slice is analyzed by fitting a circle to the contour using least-squares optimization. The fitted circle's diameter provides a measurement that accounts for the vessel's actual geometry in the slice plane rather than relying solely on distance to boundaries.

Both methods produce mean diameter, median diameter, and a complete diameter profile for each edge. The distance transform method is faster and provides good approximations for relatively straight segments. The slicing method is more accurate for curved vessels or irregular cross-sections but requires more computation. Having both measurements available allows for validation and method comparison in subsequent analysis.

### Branch Length Computation

Path lengths for each edge are computed by integrating Euclidean distances between consecutive voxels along the stored path. The algorithm accounts for diagonal connections in the 26-connected space, correctly measuring distances through the 3D lattice. All length measurements use the physical spacing information, yielding values in millimeters. These lengths represent the actual tortuous path of each vessel segment rather than straight-line distances between endpoints.

### Origin Node Identification and Graph Orientation

Coronary arteries exhibit a characteristic tapering pattern where diameter decreases from proximal to distal. The system exploits this property to identify the origin node (ostium) of the vessel tree. By examining all edges connected to endpoint nodes, the algorithm identifies the edge with the largest mean diameter. The endpoint connected to this edge is designated as the origin node, as it represents the entry point where blood flow enters the coronary circulation.

Once the origin is identified, the undirected graph is converted to a directed graph with edges oriented away from the origin. This transformation establishes a clear proximal-to-distal direction throughout the tree, which is essential for subsequent branch labeling and angle computations. The conversion uses breadth-first traversal from the origin, ensuring that each edge's direction points toward the periphery of the vascular tree.

### Bifurcation Angle Computation

At each bifurcation node in the directed graph, the system computes geometric angles characterizing the vessel split. The algorithm identifies the incoming edge (parent) and two outgoing edges (daughters) at each bifurcation. For each of these three edges, the system extracts a segment of the vessel path extending a configurable distance from the bifurcation point.

Multiple path segments are extracted at varying distances from the bifurcation, defined by the depth range specified in the configuration. At each distance, tangent vectors are computed for the three edges by fitting lines to the extracted path segments. The angles between these tangent vectors are then calculated using vector dot products. This process is repeated across the full range of depths, and the resulting angle measurements are averaged to produce robust estimates that are less sensitive to local geometric variations near the bifurcation point.

Three primary angles are stored at each bifurcation node: angle A (between the parent and first daughter), angle B (between the parent and second daughter), and angle C (between the two daughter branches). Additionally, an inflow angle characterizing the parent vessel's approach geometry may be computed. These averaged angles provide quantitative descriptors of bifurcation geometry that are critical for hemodynamic and morphological analysis.

### Artery Classification and Branch Labeling

For combined coronary artery datasets containing both left and right coronary arteries, the system must first classify which vessel is which. This classification exploits the morphological complexity difference between LCA and RCA. The left coronary artery typically exhibits greater branching complexity due to its bifurcation into LAD and LCx, each with multiple side branches. The system computes complexity metrics for each vessel graph, including endpoint count, bifurcation count, total path length, and a weighted composite complexity score. The vessel with higher complexity is classified as LCA, while the other is designated RCA.

For LCA vessels, the branch labeling algorithm first determines whether the vessel exhibits a bifurcation or trifurcation pattern. A trifurcation is detected when one of the two main branches emerging from the left main coronary artery is very short (less than 5mm by default) and quickly bifurcates. This creates three primary branches rather than the typical two. The detection examines the length of the first-level branches and their immediate children to identify this pattern.

In a standard bifurcation pattern, the two main branches are labeled as LAD (left anterior descending) and LCx (left circumflex). The labeling uses spatial validation by computing direction vectors from the bifurcation point to the distal endpoints of each branch. The LAD typically courses more anteriorly, characterized by decreasing values in the axis that corresponds to the anterior-posterior direction. The branch with greater anterior motion is assigned as LAD, while the other becomes LCx.

For trifurcations, the third branch is identified as the Ramus intermedius. The system determines which of the three branches is the Ramus by geometric centrality analysis. For each branch, the algorithm computes the angle formed by the other two branches. The Ramus is identified as the branch where the other two branches are most separated in space, indicating that the Ramus occupies a central geometric position between them.

Side branches are labeled sequentially as they are encountered in the graph traversal. Branches emerging from the LAD are numbered as diagonals (D1, D2, D3, etc.), while branches from the LCx are labeled as obtuse marginals (OM1, OM2, OM3, etc.). If a Ramus is present, its side branches receive R-numbered labels. The system distinguishes side branches from distal continuations by analyzing the edge position encoding, which captures the topological relationship of each edge to its parent.

For RCA vessels, all branches are labeled with RCA as the main trunk. Side branches receive specific anatomical labels where possible based on their position and characteristics, though the RCA labeling system is less complex than LCA due to the typically simpler branching pattern.

### Output Generation

The complete analysis results are saved in multiple formats. CSV files provide human-readable tabular data with separate files for nodes and edges. The node CSV contains coordinates and any computed node attributes such as bifurcation angles. The edge CSV includes start and end nodes, path length, diameter measurements, branch labels, and generation information.

A comprehensive pickle file is generated containing the complete analysis state. This file bundles the final directed graph with all computed attributes, the undirected sparse graph from an earlier processing stage, the binary vessel mask, the distance transform array, spacing information, the original NRRD header, processing time breakdowns, and metadata about the processing configuration. This pickle format preserves all Python object types exactly, including complex data structures like lists and dictionaries stored as graph attributes. The pickle file serves as a complete snapshot that can be loaded for statistical analysis without reprocessing.

## Batch Processing

The batch processing pipeline extends the single-artery processing to handle multiple NRRD files automatically. The batch system iterates through all NRRD files in the input directory, processing each sequentially. For files containing multiple vessel bodies, the system automatically extracts the two largest connected components and processes them as separate arteries.

A significant optimization in the batch pipeline is the single computation of the distance transform for the combined mask before vessel separation. Rather than computing separate distance transforms for each extracted vessel, the system computes one transform for the full mask and then masks it appropriately for each vessel. This reduces computational cost substantially for large datasets.

After both vessels are processed through the single-artery pipeline, the batch system applies vessel classification to determine which is LCA and which is RCA. Anatomical branch labeling is then applied with the appropriate algorithm for each artery type. The system saves results for each vessel with clear naming (filename_LCA_* and filename_RCA_*) and generates separate pickle files for each.

The batch processing includes memory monitoring and optional user prompts to manage memory usage when visualization is enabled. Processing times are tracked for each file and each pipeline stage, with summary statistics provided at completion. Error handling ensures that failures in individual files do not halt the entire batch, with failed files logged and reported in the final summary.

## Usage

### Single Artery Processing

To process a single artery mask, use the single artery pipeline with explicit parameters or a configuration file:

```python
from pipelines import process_single_artery
from utilities import load_nrrd_mask

# Load the binary mask
binary_mask, header = load_nrrd_mask('path/to/artery.nrrd')
spacing_info = tuple(np.diag(header['space directions']))

# Process with explicit parameters
result = process_single_artery(
    binary_mask=binary_mask,
    spacing_info=spacing_info,
    min_depth_mm=2.0,
    max_depth_mm=7.0,
    step_mm=0.5,
    remove_bypass=True,
    bypass_threshold=2.0,
    nodes_csv='output/nodes.csv',
    edges_csv='output/edges.csv'
)

# Access results
final_graph = result['final_graph']
processing_times = result['processing_times']
```

Alternatively, use a configuration file:

```python
result = process_single_artery(
    binary_mask=binary_mask,
    spacing_info=spacing_info,
    config_path='config.yaml',
    nodes_csv='output/nodes.csv',
    edges_csv='output/edges.csv'
)
```

### Batch Processing

For processing multiple files, use the batch pipeline:

```python
from pipelines import process_batch_arteries

results = process_batch_arteries(
    input_folder='data/masks',
    output_folder='results/processed',
    config_path='config.yaml',
    visualize=False
)

# Check processing summary
print(f"Successfully processed: {results['success_count']}/{results['total_files']}")
print(f"Total arteries: {results['total_arteries_processed']}")
```

The batch processing creates output files following the naming convention `{basename}_{LCA|RCA}_nodes.csv`, `{basename}_{LCA|RCA}_edges.csv`, and `{basename}_{LCA|RCA}_analysis.pkl` for each processed artery.

### Loading Processed Data

To load previously processed data for analysis:

```python
from utilities import load_artery_analysis

# Load complete analysis
data = load_artery_analysis('results/processed/Patient001_LCA_analysis.pkl')

# Access all components
graph = data['final_graph']
spacing = data['spacing_info']
binary_mask = data['binary_mask']
distance_array = data['distance_array']
metadata = data['metadata']

# Extract information from graph
for u, v in graph.edges():
    branch_label = graph[u][v]['branch_label']
    length = graph[u][v]['length']
    diameter = graph[u][v]['mean_diameter_slicing']
    profile = graph[u][v]['diameter_profile_slicing']
```

### Configuration Tuning

The configuration parameters can be adjusted based on data characteristics and analysis requirements. For low-resolution data, consider increasing the upsample factor to 2 or 3 to improve skeleton quality. For datasets with noisy bifurcations, reducing the bypass threshold can help eliminate spurious connections. The depth range for angle computation should be adjusted based on typical vessel caliber in the dataset; larger vessels may benefit from a wider range.

When processing large batches, disable visualization to avoid memory accumulation from multiple open plotting windows. Memory usage can be substantial for high-resolution volumes, particularly during distance transform computation and graph construction. Monitor system resources and consider processing large datasets in smaller batches if memory constraints are encountered.
