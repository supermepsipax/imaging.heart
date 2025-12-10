import plotly.graph_objects as go
from skimage import measure
import numpy as np
import networkx as nx


def get_anatomical_branch_color(anatomical_label):
    """
    Determines color based on anatomical branch label.

    Args:
        anatomical_label (str): Anatomical branch label (e.g., "LAD", "LCx", "D1", "OM2", "AM1")

    Returns:
        str: Color name for the branch
    """
    if not anatomical_label:
        return 'gray'

    # Main trunks: Blue
    if anatomical_label in ['RCA', 'Left_Main', 'LAD']:
        return 'blue'

    # LCx: Orange
    if anatomical_label == 'LCx':
        return 'orange'

    # Ramus: Purple
    if anatomical_label == 'Ramus' or anatomical_label.startswith('R'):
        return 'purple'

    # All marginal and diagonal branches: Green
    if (anatomical_label.startswith('D') or     # Diagonal (LAD side branches)
        anatomical_label.startswith('OM') or    # Obtuse Marginal (LCx side branches)
        anatomical_label.startswith('AM')):     # Acute Marginal (RCA side branches)
        return 'green'

    # Default
    return 'gray'


def get_edge_hierarchy_color(edge_position_label):
    """
    Determines color based on edge position label for hierarchical visualization.

    Args:
        edge_position_label (str): The edge position label (e.g., "1", "12", "121", "122")

    Returns:
        str: Color name for the edge
            - Main branch (all 1s): 'blue'
            - Next main (1 followed by all 2s): 'orange'
            - Other branches: 'green'
    """
    if not edge_position_label:
        return 'gray'  # Default for unlabeled edges

    # Check if all characters are '1' (main trunk)
    if all(c == '1' for c in edge_position_label):
        return 'blue'

    # Check if first char is '1' and rest are all '2' (next main branch)
    if len(edge_position_label) > 1 and edge_position_label[0] == '1' and all(c == '2' for c in edge_position_label[1:]):
        return 'orange'

    # All other branches
    return 'green'


def visualize_binary_mask(binary_mask, title="3D Binary Mask", hide_background=False, dark_mode=False):
    """
    Visualizes just the binary mask as a 3D mesh surface without any graph overlay.

    Args:
        binary_mask: 3D numpy array where 1s represent the artery
        title (str): Title for the visualization (default: "3D Binary Mask")
        hide_background (bool): If True, hides background, grid, and axes for cleaner visualization (default: False)
        dark_mode (bool): If True, uses dark background with light grid/axes (default: False)
    """
    # Generate mesh from binary mask
    verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)

    artery_mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        color='lightcoral',
        name='Artery surface',
        flatshading=True
    )

    fig = go.Figure(data=[artery_mesh])

    # Configure layout based on dark_mode and hide_background settings
    if dark_mode and hide_background:
        # Dark mode with hidden grid/axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                zaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                bgcolor='#1a1a1a',
                aspectmode='data'
            ),
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            title=title,
            font=dict(color='white')
        )
    elif dark_mode:
        # Dark mode with visible light grid/axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                bgcolor='#1a1a1a',
                aspectmode='data'
            ),
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            title=title,
            font=dict(color='white')
        )
    elif hide_background:
        # Light mode with hidden grid/axes (transparent background)
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                zaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                bgcolor='rgba(0,0,0,0)',
                aspectmode='data'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=title
        )
    else:
        # Default light mode with visible grid and axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                zaxis=dict(showgrid=True),
                aspectmode='data'
            ),
            title=title
        )

    fig.show()


def visualize_3d_graph(graph, binary_mask=None, title="3D Graph with Artery Surface", hide_background=False, dark_mode=False):
    """
    Will create a 3D visualization that will open in a web browser to view a created networkx graph
    and an optional 3d mesh of the original binary mask.

    Args:
        graph: NetworkX graph where:
            - nodes are (x, y, z) coordinate tuples
            - edges have 'voxels' attribute with list of voxel coordinates in the path
        binary_mask: 3D numpy array where 1s represent the artery
        title (str): Title for the visualization (default: "3D Graph with Artery Surface")
        hide_background (bool): If True, hides background, grid, and axes for cleaner visualization (default: False)
        dark_mode (bool): If True, uses dark background with light grid/axes (default: False)
    """
    traces = []
    
    if binary_mask is not None:
        verts, faces, _, _ = measure.marching_cubes(binary_mask, level=0.5)
        
        artery_mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.2,
            color='lightcoral',
            name='Artery surface',
            flatshading=True
        )
        traces.append(artery_mesh)
    
    straight_x, straight_y, straight_z = [], [], []
    for edge in graph.edges():
        x0, y0, z0 = edge[0]
        x1, y1, z1 = edge[1]
        straight_x.extend([x0, x1, None])
        straight_y.extend([y0, y1, None])
        straight_z.extend([z0, z1, None])
    
    straight_trace = go.Scatter3d(
        x=straight_x, y=straight_y, z=straight_z,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        hoverinfo='none',
        name='Direct connection'
    )
    traces.append(straight_trace)

    # Check if graph is directed and has labeling
    is_directed = isinstance(graph, nx.DiGraph)
    has_anatomical_labels = False
    has_edge_positions = False

    if graph.number_of_edges() > 0:
        # Check for anatomical labels (lca_branch or rca_branch)
        edges_with_anatomical = sum(
            1 for edge in graph.edges()
            if 'lca_branch' in graph.edges[edge] or 'rca_branch' in graph.edges[edge]
        )
        has_anatomical_labels = edges_with_anatomical > 0

        # Check how many edges have edge_position attribute
        edges_with_position = sum(1 for edge in graph.edges() if 'edge_position' in graph.edges[edge])
        has_edge_positions = edges_with_position > 0

        if has_anatomical_labels:
            print(f"[INFO] Visualization: Using anatomical branch labels for coloring")
        elif has_edge_positions:
            print(f"[INFO] Visualization: Using edge position hierarchy for coloring")
            if edges_with_position < graph.number_of_edges():
                print(f"       {edges_with_position}/{graph.number_of_edges()} edges have 'edge_position' attribute.")

    # If directed and has labels, color by anatomy or hierarchy
    if is_directed and (has_anatomical_labels or has_edge_positions):
        # Group edges by color category
        edges_by_color = {'blue': [], 'orange': [], 'green': [], 'purple': [], 'gray': []}

        for edge in graph.edges():
            edge_data = graph.edges[edge]

            # Try anatomical labeling first
            if has_anatomical_labels:
                anatomical_label = edge_data.get('lca_branch') or edge_data.get('rca_branch', '')
                color = get_anatomical_branch_color(anatomical_label)
            else:
                # Fall back to edge position hierarchy
                edge_position = edge_data.get('edge_position', '')
                color = get_edge_hierarchy_color(edge_position)

            edges_by_color[color].append(edge)

        # Create separate traces for each color category
        if has_anatomical_labels:
            color_names = {
                'blue': 'Main vessels (RCA/LAD/Left Main)',
                'orange': 'LCx',
                'purple': 'Ramus',
                'green': 'Side branches (D/OM/AM)',
                'gray': 'Unlabeled edges'
            }
        else:
            color_names = {
                'blue': 'Main trunk (all 1s)',
                'orange': 'First major branch (1+2s)',
                'green': 'Other branches',
                'purple': 'Other branches',
                'gray': 'Unlabeled edges'
            }

        for color, edges_list in edges_by_color.items():
            if len(edges_list) == 0:
                continue

            path_x, path_y, path_z = [], [], []
            for edge in edges_list:
                voxel_path = graph.edges[edge]['voxels']
                for voxel in voxel_path:
                    x, y, z = voxel
                    path_x.append(x)
                    path_y.append(y)
                    path_z.append(z)
                path_x.append(None)
                path_y.append(None)
                path_z.append(None)

            path_trace = go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines',
                line=dict(color=color, width=5),
                hoverinfo='none',
                name=color_names[color]
            )
            traces.append(path_trace)
    else:
        # Original behavior: single color for all edges
        path_x, path_y, path_z = [], [], []
        for edge in graph.edges():
            voxel_path = graph.edges[edge]['voxels']
            for voxel in voxel_path:
                x, y, z = voxel
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)
            path_x.append(None)
            path_y.append(None)
            path_z.append(None)

        path_trace = go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode='lines',
            line=dict(color='darkgreen', width=5),
            hoverinfo='none',
            name='Actual voxel path'
        )
        traces.append(path_trace)
    
    # Check if directed graph to highlight origin node
    if is_directed:
        # Find origin node (in_degree == 0)
        origin_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
        regular_nodes = [node for node in graph.nodes() if graph.in_degree(node) > 0]

        # Add origin node(s) as separate trace in blue
        if origin_nodes:
            origin_x = [node[0] for node in origin_nodes]
            origin_y = [node[1] for node in origin_nodes]
            origin_z = [node[2] for node in origin_nodes]

            origin_trace = go.Scatter3d(
                x=origin_x, y=origin_y, z=origin_z,
                mode='markers',
                marker=dict(size=12, color='blue', symbol='diamond'),
                text=[f"ORIGIN: {str(node)}" for node in origin_nodes],
                hoverinfo='text',
                name='Origin node'
            )
            traces.append(origin_trace)

        # Add regular nodes
        if regular_nodes:
            node_x = [node[0] for node in regular_nodes]
            node_y = [node[1] for node in regular_nodes]
            node_z = [node[2] for node in regular_nodes]

            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers',
                marker=dict(size=8, color='darkred', symbol='diamond'),
                text=[str(node) for node in regular_nodes],
                hoverinfo='text',
                name='Nodes'
            )
            traces.append(node_trace)
    else:
        # Original behavior for undirected graphs
        node_x = [node[0] for node in graph.nodes()]
        node_y = [node[1] for node in graph.nodes()]
        node_z = [node[2] for node in graph.nodes()]

        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=8, color='darkred', symbol='diamond'),
            text=[str(node) for node in graph.nodes()],
            hoverinfo='text',
            name='Nodes'
        )
        traces.append(node_trace)
    
    fig = go.Figure(data=traces)

    # Configure layout based on dark_mode and hide_background settings
    if dark_mode and hide_background:
        # Dark mode with hidden grid/axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                zaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                bgcolor='#1a1a1a',
                aspectmode='data'
            ),
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            title=title,
            font=dict(color='white')
        )
    elif dark_mode:
        # Dark mode with visible light grid/axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                zaxis=dict(
                    showgrid=True,
                    gridcolor='#444444',
                    showbackground=True,
                    backgroundcolor='#1a1a1a',
                    color='white'
                ),
                bgcolor='#1a1a1a',
                aspectmode='data'
            ),
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            title=title,
            font=dict(color='white')
        )
    elif hide_background:
        # Light mode with hidden grid/axes (transparent background)
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                zaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    visible=False
                ),
                bgcolor='rgba(0,0,0,0)',
                aspectmode='data'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=title
        )
    else:
        # Default light mode with visible grid and axes
        fig.update_layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                zaxis=dict(showgrid=True),
                aspectmode='data'
            ),
            title=title
        )

    fig.show()


