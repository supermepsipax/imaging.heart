import os
import nrrd
import numpy as np
from pathlib import Path
from utilities import ensure_continuous_body, load_config, load_nrrd_mask, sort_labelled_bodies_by_size
from utilities import extract_centerline_skimage, skeleton_to_dense_graph, create_distance_transform_from_mask
from visualizations.visualize_3d import visualize_mask_overlap


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

def _fill_bridge(reconnected_mask, startpoint, endpoint, radius1, radius2, dir1, dir2):
    """
    Draw a bridge between two skeleton endpoints using a cubiz Bezier curve, expanding each centerline
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

def reconnect_mask(input_mask, distance_threshold=5, direction_lookback=5, alignment_threshold=0.7):
    """
    Attempts to reconnect disconnected regions in a single binary mask.

    For each pair of bodies, skeletonizes both, finds their endpoints, and checks
    whether any cross-body endpoint pair is close enough and directionally aligned
    to be a broken vessel. Aligned pairs within the distance threshold are bridged
    via linear interpolation, with each bridge voxel expanded into a sphere whose
    radius is linearly interpolated between the two endpoint radii (read from the
    Euclidean distance transform of the full input mask, computed once).

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

    Returns:
        tuple: (reconnected_mask, connections_made)
            - reconnected_mask (np.ndarray): Binary mask with interpolated bridge voxels added.
            - connections_made (int): Number of endpoint pairs that were bridged.
    """
    is_continuous, labelled_bodies = ensure_continuous_body(input_mask)
    if is_continuous:
        return input_mask.copy(), 0

    # Compute once — used to look up vessel radius at each candidate endpoint.
    distance_array = create_distance_transform_from_mask(input_mask)

    sorted_bodies = sort_labelled_bodies_by_size(labelled_bodies)
    n_bodies = len(sorted_bodies)

    body_data = []
    for body_mask in sorted_bodies:
        skeleton = extract_centerline_skimage(body_mask)
        dense_graph = skeleton_to_dense_graph(skeleton)
        endpoints = [node for node in dense_graph.nodes() if dense_graph.degree(node) == 1]
        body_data.append({'graph': dense_graph, 'endpoints': endpoints})

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
                        radius_i = float(distance_array[start_voxel_i])

                        start_idx_j = min(len(path_j) - 1, direction_lookback // 2)
                        start_voxel_j = path_j[start_idx_j]
                        radius_j = float(distance_array[start_voxel_j])

                        _fill_bridge(reconnected_mask, start_voxel_i, start_voxel_j, radius_i, radius_j, dir_i, dir_j)
                        connections_made += 1
                        print(f"      --> Bridged body {i} {ep_i} <-> body {j} {ep_j} "
                              f"(dist={dist:.1f}, r={radius_i:.1f}->{radius_j:.1f}, "
                              f"dot_i={dot_i:.2f}, dot_j={dot_j:.2f})")

    return reconnected_mask, connections_made


def reconnect_mask_batch(mask_folder=None, output_folder=None, config=None, config_path=None, visualize=None):
    """
    Batch reconnection of disconnected coronary artery segmentation masks.

    For each mask in the input folder, attempts to bridge small gaps between
    disconnected bodies by comparing endpoint directions and distances along
    their skeletons. Masks that are already continuous are copied unchanged.
    Results are saved as NRRD files preserving the original headers.

    Args:
        mask_folder (str, optional): Path to folder containing .nrrd masks to process.
        output_folder (str, optional): Path to folder where reconnected masks are saved.
        config (dict, optional): Configuration dictionary with pipeline parameters.
        config_path (str, optional): Path to YAML/JSON config file (used if config not given).
        visualize (bool, optional): Reserved for future visualization support.

    Config keys:
        mask_folder (str): Input folder path.
        output_folder (str): Output folder path.
        distance_threshold (float): Max endpoint distance in voxels to attempt bridging (default: 5).
        direction_lookback (int): Skeleton voxels to walk back for direction estimation (default: 5).
        alignment_threshold (float): Min cosine similarity for directional alignment (default: 0.7).

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


    if mask_folder is None:
        raise ValueError("mask_folder must be provided either as a parameter or in the config file")
    if output_folder is None:
        raise ValueError("output_folder must be provided either as a parameter or in the config file")

    os.makedirs(output_folder, exist_ok=True)

    nrrd_files = list(Path(mask_folder).glob('*.nrrd'))
    results = {}

    print(f"Found {len(nrrd_files)} mask(s) to process")
    print(f"  distance_threshold={distance_threshold} voxels, "
          f"direction_lookback={direction_lookback}, "
          f"alignment_threshold={alignment_threshold}")

    for mask_path in nrrd_files:
        print(f"\n[{mask_path.name}]")
        input_mask, header = load_nrrd_mask(mask_path)
        out_path = Path(output_folder) / mask_path.name

        is_continuous, _ = ensure_continuous_body(input_mask)

        if is_continuous:
            print(f"  Already continuous, copying unchanged.")
            nrrd.write(str(out_path), input_mask, header)
            results[mask_path.name] = {
                'connections_made': 0,
                'was_continuous': True,
                'is_continuous_after': True,
            }
            continue

        reconnected_mask, connections_made = reconnect_mask(
            input_mask,
            distance_threshold=distance_threshold,
            direction_lookback=direction_lookback,
            alignment_threshold=alignment_threshold,
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

        if visualize and connections_made > 0:
            visualize_mask_overlap(
                input_mask,
                reconnected_mask,
                title=f"Reconnection: {mask_path.name}",
                label_1="Original",
                label_2="Reconnected",
            )

    return results
