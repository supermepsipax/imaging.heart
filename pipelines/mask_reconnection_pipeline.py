import os
import time
import nrrd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import networkx as nx
from scipy import ndimage
from utilities import ensure_continuous_body, load_config, load_mask, glob_masks
from utilities import extract_centerline_skimage, skeleton_to_dense_graph, create_distance_transform_from_mask


def _get_endpoint_direction(endpoint, dense_graph, lookback=5):
    """
    Walk `lookback` steps back from an endpoint into the skeleton.

    Returns (direction, path):
        direction: unit vector pointing outward from the body (from the interior
                   toward the endpoint tip, i.e. toward the gap), or None if the
                   walked path is too short to define one.
        path: ordered list of skeleton voxels starting at the endpoint and
              moving inward.
    """
    path = [endpoint]
    current = endpoint
    previous = None

    for _ in range(lookback):
        neighbors = [n for n in dense_graph.neighbors(current) if n != previous]
        if not neighbors:
            break
        previous = current
        current = neighbors[0]
        path.append(current)

    if len(path) < 2:
        return None, path

    direction = np.array(path[0]) - np.array(path[-1])
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None, path
    return direction / norm, path

def _local_edt_radius(input_mask, voxel_a, voxel_b, padding):
    """
    Compute EDT on a local crop around two voxels and return their radius values.

    Avoids computing the full-mask EDT by extracting a tight region padded
    enough that EDT values at the query voxels match the global transform.
    """
    coords = np.array([voxel_a, voxel_b])
    min_corner = np.maximum(coords.min(axis=0) - padding, 0)
    max_corner = np.minimum(coords.max(axis=0) + padding + 1, np.array(input_mask.shape))

    slices = tuple(slice(lo, hi) for lo, hi in zip(min_corner, max_corner))
    local_dt = create_distance_transform_from_mask(input_mask[slices])

    local_a = np.array(voxel_a) - min_corner
    local_b = np.array(voxel_b) - min_corner
    return float(local_dt[tuple(local_a)]), float(local_dt[tuple(local_b)])


def _fill_bridge(reconnected_mask, startpoint, endpoint, radius1, radius2, dir1, dir2):
    """
    Draw a bridge between two skeleton endpoints using a cubic Bezier curve, expanding each centerline
    voxel into a sphere whose radius is linearly interpolated between radius1
    and radius2.

    The sphere expansion ensures the bridge cross-section matches the vessel
    width at each endpoint rather than leaving a 1-voxel-wide seam.
    """
    shape = reconnected_mask.shape

    # Convert p0 and p3 (start- and endpoints, tuples) into arrays
    p0 = np.array(startpoint, dtype = float)
    p3 = np.array(endpoint, dtype = float)

    dist = np.linalg.norm(p3 - p0)
    n_steps = max(int(np.ceil(dist)) + 1, 2)
    t = np.linspace(0, 1, n_steps)

    # p1 (control point) corresponds to t = 1/3
    # p2 (control point) corresponds to t = 2/3
    p1 = p0 + dir1 * (dist / 3)
    p2 = p3 + dir2 * (dist / 3)

    # B(t) = (1-t)^3 * p0 + 3(1-t)^2 * t * p1 + 3(1-t) * t^2 * p2 + t^3 * p3
    coords = np.outer((1 - t)**3, p0) + np.outer(3 * (1 - t)**2 * t, p1) + np.outer(3 * (1 - t) * t**2, p2) + np.outer(t**3, p3)

    for idx, c in enumerate(coords):
        t = idx / (len(coords) - 1)
        radius = (1 - t) * radius1 + t * radius2
        center = np.clip(np.round(c).astype(int), 0, np.array(shape) - 1)

        r_ceil = int(np.ceil(radius))
        for d0 in range(-r_ceil, r_ceil + 1):
            for d1 in range(-r_ceil, r_ceil + 1):
                for d2 in range(-r_ceil, r_ceil + 1):
                    if d0 ** 2 + d1 ** 2 + d2 ** 2 <= radius ** 2:
                        v = (
                            int(np.clip(center[0] + d0, 0, shape[0] - 1)),
                            int(np.clip(center[1] + d1, 0, shape[1] - 1)),
                            int(np.clip(center[2] + d2, 0, shape[2] - 1)),
                        )
                        reconnected_mask[v] = 1

def reconnect_mask(input_mask, distance_threshold=5, direction_lookback=5, alignment_threshold=0.7, min_body_size=10):
    """
    Attempts to reconnect disconnected regions in a single binary mask.

    For each pair of bodies, skeletonizes both, finds their endpoints, and checks
    whether any cross-body endpoint pair is close enough and directionally aligned
    to be a broken vessel. Aligned pairs within the distance threshold are bridged
    via Bezier interpolation, with each bridge voxel expanded into a sphere whose
    radius is linearly interpolated between the two endpoint radii (read from a
    localized EDT crop around each qualifying endpoint pair).

    The alignment check requires that each endpoint's outward direction vector
    (computed by walking back `direction_lookback` voxels along the skeleton) points
    toward the other endpoint within `alignment_threshold` (cosine of max angle).

    Args:
        input_mask (np.ndarray): 3D binary mask, possibly containing multiple disconnected bodies.
        distance_threshold (float): Maximum Euclidean voxel distance between two endpoints
                                    to attempt reconnection.
        direction_lookback (int): Number of skeleton voxels to walk back from each endpoint
                                  when estimating its outward direction vector.
        alignment_threshold (float): Minimum cosine similarity (dot product) required between
                                     each endpoint's outward direction and the vector toward the
                                     opposing endpoint. 1.0 = perfectly aligned, 0.0 = orthogonal.
        min_body_size (int): Minimum number of voxels for a disconnected body to be
                             considered for reconnection. Smaller fragments are skipped
                             as noise. Defaults to 10.

    Returns:
        tuple: (reconnected_mask, connections_made)
            - reconnected_mask (np.ndarray): Binary mask with interpolated bridge voxels added.
            - connections_made (int): Number of endpoint pairs that were bridged.
    """
    connectivity_structure = np.ones((3, 3, 3), dtype=int)
    labelled_bodies, num_bodies = ndimage.label(input_mask, structure=connectivity_structure)

    if num_bodies <= 1:
        return input_mask.copy(), 0

    edt_padding = distance_threshold + direction_lookback

    body_sizes = np.bincount(labelled_bodies.ravel())
    body_slices = ndimage.find_objects(labelled_bodies)

    labels_by_size = [
        (label, body_sizes[label])
        for label in range(1, num_bodies + 1)
        if body_sizes[label] >= min_body_size
    ]
    labels_by_size.sort(key=lambda x: x[1], reverse=True)
    n_bodies = len(labels_by_size)

    skipped = num_bodies - n_bodies
    if skipped > 0:
        print(f"      Skipped {skipped} fragment(s) below {min_body_size} voxels")

    if n_bodies <= 1:
        return input_mask.copy(), 0

    body_data = []
    for label, size in labels_by_size:
        slc = body_slices[label - 1]
        offset = tuple(s.start for s in slc)

        crop = (labelled_bodies[slc] == label).astype(np.uint8)
        skeleton = extract_centerline_skimage(crop)
        local_graph = skeleton_to_dense_graph(skeleton)

        mapping = {
            node: (node[0] + offset[0], node[1] + offset[1], node[2] + offset[2])
            for node in local_graph.nodes()
        }
        global_graph = nx.relabel_nodes(local_graph, mapping)

        endpoints = [node for node in global_graph.nodes() if global_graph.degree(node) == 1]
        body_data.append({'graph': global_graph, 'endpoints': endpoints})

    reconnected_mask = input_mask.copy()
    connections_made = 0

    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            eps_i = body_data[i]['endpoints']
            eps_j = body_data[j]['endpoints']
            graph_i = body_data[i]['graph']
            graph_j = body_data[j]['graph']

            for ep_i in eps_i:
                for ep_j in eps_j:
                    dist = np.linalg.norm(np.array(ep_i) - np.array(ep_j))
                    if dist > distance_threshold:
                        continue

                    dir_i, path_i = _get_endpoint_direction(ep_i, graph_i, direction_lookback)
                    dir_j, path_j = _get_endpoint_direction(ep_j, graph_j, direction_lookback)

                    if dir_i is None or dir_j is None:
                        continue

                    delta = np.array(ep_j) - np.array(ep_i)
                    delta_norm = np.linalg.norm(delta)
                    if delta_norm < 1e-6:
                        continue
                    delta_unit = delta / delta_norm

                    # dir_i should point toward ep_j; dir_j should point toward ep_i
                    dot_i = np.dot(dir_i, delta_unit)
                    dot_j = np.dot(dir_j, -delta_unit)

                    if dot_i >= alignment_threshold and dot_j >= alignment_threshold:
                        # Sample EDT over interior skeleton voxels — the endpoint
                        # itself sits at the cut face where EDT collapses to ~1
                        # regardless of true vessel radius.

                        start_idx_i = min(len(path_i) - 1, direction_lookback // 2)
                        start_voxel_i = path_i[start_idx_i]

                        start_idx_j = min(len(path_j) - 1, direction_lookback // 2)
                        start_voxel_j = path_j[start_idx_j]

                        radius_i, radius_j = _local_edt_radius(
                            input_mask, start_voxel_i, start_voxel_j, edt_padding
                        )

                        _fill_bridge(reconnected_mask, start_voxel_i, start_voxel_j, radius_i, radius_j, dir_i, dir_j)
                        connections_made += 1
                        print(f"      --> Bridged body {i} {ep_i} <-> body {j} {ep_j} "
                              f"(dist={dist:.1f}, r={radius_i:.1f}->{radius_j:.1f}, "
                              f"dot_i={dot_i:.2f}, dot_j={dot_j:.2f})")

    return reconnected_mask, connections_made


def _process_single_mask(mask_path, output_folder, distance_threshold, direction_lookback, alignment_threshold, min_body_size):
    input_mask, header = load_mask(mask_path)
    stem = mask_path.name.removesuffix('.nii.gz').removesuffix('.nii').removesuffix('.nrrd')
    out_path = Path(output_folder) / f"{stem}.nrrd"

    is_continuous, _ = ensure_continuous_body(input_mask)

    if is_continuous:
        nrrd.write(str(out_path), input_mask, header)
        return mask_path.name, {
            'connections_made': 0,
            'was_continuous': True,
            'is_continuous_after': True,
        }

    reconnected_mask, connections_made = reconnect_mask(
        input_mask,
        distance_threshold=distance_threshold,
        direction_lookback=direction_lookback,
        alignment_threshold=alignment_threshold,
        min_body_size=min_body_size,
    )

    is_continuous_after, _ = ensure_continuous_body(reconnected_mask)

    nrrd.write(str(out_path), reconnected_mask, header)
    return mask_path.name, {
        'connections_made': connections_made,
        'was_continuous': False,
        'is_continuous_after': is_continuous_after,
    }


def reconnect_mask_batch(mask_folder=None, output_folder=None, config=None, config_path=None, visualize=None, n_jobs=1):
    """
    Batch reconnection of disconnected coronary artery segmentation masks.

    For each mask in the input folder, attempts to bridge small gaps between
    disconnected bodies by comparing endpoint directions and distances along
    their skeletons. Masks that are already continuous are copied unchanged.
    Results are saved as NRRD files preserving the original headers.

    When n_jobs != 1, files are processed in parallel using joblib and
    visualization is disabled. When n_jobs=1 (default), files are processed
    sequentially with optional visualization support.

    Args:
        mask_folder (str, optional): Path to folder containing .nrrd/.nii/.nii.gz masks to process.
        output_folder (str, optional): Path to folder where reconnected masks are saved.
        config (dict, optional): Configuration dictionary with pipeline parameters.
        config_path (str, optional): Path to YAML/JSON config file (used if config not given).
        visualize (bool, optional): Whether to visualize reconnected masks (sequential mode only).
        n_jobs (int, optional): Number of parallel workers. -1 uses all cores, 1 runs
                                sequentially with visualization support. Defaults to 1.

    Config keys:
        mask_folder (str): Input folder path.
        output_folder (str): Output folder path.
        distance_threshold (float): Max endpoint distance in voxels to attempt bridging (default: 5).
        direction_lookback (int): Skeleton voxels to walk back for direction estimation (default: 5).
        alignment_threshold (float): Min cosine similarity for directional alignment (default: 0.7).
        n_jobs (int): Number of parallel workers (default: 1).

    Returns:
        dict: Per-file result summary with keys:
            - 'connections_made': Number of endpoint pairs bridged.
            - 'was_continuous': Whether the input was already continuous.
            - 'is_continuous_after': Whether the output is a single body.

    Raises:
        ValueError: If mask_folder or output_folder are not provided.
    """
    if config is None and config_path is not None:
        config = load_config(config_path)
    if config is None:
        config = {}

    if mask_folder is None:
        mask_folder = config.get('mask_folder')
    if output_folder is None:
        output_folder = config.get('output_folder')
    if visualize is None:
        visualize = config.get('visualize', False)

    distance_threshold = config.get('distance_threshold', 5)
    direction_lookback = config.get('direction_lookback', 5)
    alignment_threshold = config.get('alignment_threshold', 0.7)
    min_body_size = config.get('min_body_size', 10)
    n_jobs = config.get('n_jobs', n_jobs)

    if mask_folder is None:
        raise ValueError("mask_folder must be provided either as a parameter or in the config file")
    if output_folder is None:
        raise ValueError("output_folder must be provided either as a parameter or in the config file")

    os.makedirs(output_folder, exist_ok=True)

    mask_files = glob_masks(mask_folder)

    existing_masks = []
    for mask_path in mask_files:
        stem = mask_path.name.removesuffix('.nii.gz').removesuffix('.nii').removesuffix('.nrrd')
        if (Path(output_folder) / f"{stem}.nrrd").exists():
            existing_masks.append(mask_path)

    if existing_masks:
        print(f"\n{len(existing_masks)} of {len(mask_files)} mask(s) already exist in the output folder.")
        choice = input("(o)verwrite all / (s)kip existing? [o/s]: ").strip().lower()
        if choice == 's':
            mask_files = [p for p in mask_files if p not in existing_masks]
            print(f"Skipping {len(existing_masks)} existing mask(s), {len(mask_files)} remaining.")
        else:
            print("Overwriting existing masks.")

    print(f"Found {len(mask_files)} mask(s) to process")
    print(f"  distance_threshold={distance_threshold} voxels, "
          f"direction_lookback={direction_lookback}, "
          f"alignment_threshold={alignment_threshold}, "
          f"min_body_size={min_body_size}")

    if n_jobs != 1:
        print(f"  Parallel mode: n_jobs={n_jobs}")
        batch_start = time.time()
        file_results = Parallel(n_jobs=n_jobs, verbose=10, prefer="threads")(
            delayed(_process_single_mask)(
                mask_path, output_folder, distance_threshold, direction_lookback, alignment_threshold, min_body_size
            )
            for mask_path in mask_files
        )
        elapsed = time.time() - batch_start
        print(f"\nBatch complete: {len(mask_files)} file(s) in {elapsed:.1f}s "
              f"({elapsed / max(len(mask_files), 1):.1f}s avg/file)")
        return dict(file_results)

    results = {}
    batch_start = time.time()
    n_files = len(mask_files)
    for file_idx, mask_path in enumerate(mask_files, 1):
        file_start = time.time()
        print(f"\n[{file_idx}/{n_files}] {mask_path.name}")
        input_mask, header = load_mask(mask_path)
        stem = mask_path.name.removesuffix('.nii.gz').removesuffix('.nii').removesuffix('.nrrd')
        out_path = Path(output_folder) / f"{stem}.nrrd"

        is_continuous, _ = ensure_continuous_body(input_mask)

        if is_continuous:
            print(f"  Already continuous, copying unchanged.")
            nrrd.write(str(out_path), input_mask, header)
            results[mask_path.name] = {
                'connections_made': 0,
                'was_continuous': True,
                'is_continuous_after': True,
            }
            elapsed = time.time() - batch_start
            avg = elapsed / file_idx
            remaining = avg * (n_files - file_idx)
            print(f"  Done in {time.time() - file_start:.1f}s | "
                  f"ETA: {remaining:.0f}s remaining ({file_idx}/{n_files})")
            continue

        reconnected_mask, connections_made = reconnect_mask(
            input_mask,
            distance_threshold=distance_threshold,
            direction_lookback=direction_lookback,
            alignment_threshold=alignment_threshold,
            min_body_size=min_body_size,
        )

        is_continuous_after, _ = ensure_continuous_body(reconnected_mask)
        print(f"  Connections made: {connections_made} | "
              f"Continuous after: {is_continuous_after}")

        nrrd.write(str(out_path), reconnected_mask, header)
        results[mask_path.name] = {
            'connections_made': connections_made,
            'was_continuous': False,
            'is_continuous_after': is_continuous_after,
        }

        elapsed = time.time() - batch_start
        avg = elapsed / file_idx
        remaining = avg * (n_files - file_idx)
        print(f"  Done in {time.time() - file_start:.1f}s | "
              f"ETA: {remaining:.0f}s remaining ({file_idx}/{n_files})")

        if visualize and connections_made > 0:
            from visualizations.visualize_3d import visualize_mask_overlap
            visualize_mask_overlap(
                input_mask,
                reconnected_mask,
                title=f"Reconnection: {mask_path.name}",
                label_1="Original",
                label_2="Reconnected",
            )

    total = time.time() - batch_start
    print(f"\nBatch complete: {n_files} file(s) in {total:.1f}s "
          f"({total / max(n_files, 1):.1f}s avg/file)")
    return results
