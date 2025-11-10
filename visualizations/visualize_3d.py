import plotly.graph_objects as go
from skimage import measure

def visualize_3d_graph(graph, binary_mask=None):
    """
    Will create a 3D visualization that will open in a web browser to view a created networkx graph
    and an optional 3d mesh of the original binary mask. 

    graph: NetworkX graph where:
        - nodes are (x, y, z) coordinate tuples
        - edges have 'voxels' attribute with list of voxel coordinates in the path
    binary_mask: 3D numpy array where 1s represent the artery
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
    fig.update_layout(
        showlegend=True,
        scene=dict(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            zaxis=dict(showgrid=True),
            aspectmode='data'
        ),
        title="3D Graph with Artery Surface"
    )
    fig.show()

