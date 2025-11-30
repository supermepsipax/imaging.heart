import pandas as pd
import numpy as np


def convert_graph_to_dataframes(graph,nodes_csv="nodes.csv", edges_csv="edges.csv"):
    """
    Converts a NetworkX graph into two pandas DataFrames for nodes and edges.

    Creates separate DataFrames for node and edge data, extracting coordinate
    information and all attached attributes. Handles special attributes like
    'voxels' and 'depth_measurements' appropriately for tabular representation.

    Args:
        graph (networkx.Graph or networkx.DiGraph): A graph with coordinate tuples
                                                     as nodes and attributes stored
                                                     on nodes and edges

    Returns:
        nodes_df (pd.DataFrame): DataFrame with columns for node coordinates
                                (node_coord_0, node_coord_1, node_coord_2)
                                and any additional node attributes
        edges_df (pd.DataFrame): DataFrame with columns for source and target
                                coordinates (source_coord_0-2, target_coord_0-2)
                                and any additional edge attributes
    """

    node_data = []
    for node in graph.nodes():
        node_dict = {
            'node_coord_0': node[0],
            'node_coord_1': node[1],
            'node_coord_2': node[2]
        }
        node_attrs = graph.nodes[node]
        for key, value in node_attrs.items():
            if key == 'depth_measurements':
                node_dict[key] = str(value) if value else None
            elif key == 'bifurcation_node':
                continue
            else:
                node_dict[key] = value
        node_data.append(node_dict)

    nodes_df = pd.DataFrame(node_data)

    edge_data = []
    for edge in graph.edges():
        edge_dict = {
            'source_coord_0': edge[0][0],
            'source_coord_1': edge[0][1],
            'source_coord_2': edge[0][2],
            'target_coord_0': edge[1][0],
            'target_coord_1': edge[1][1],
            'target_coord_2': edge[1][2]
        }
        edge_attrs = graph.edges[edge]
        for key, value in edge_attrs.items():
            if key == 'voxels':
                edge_dict['num_voxels'] = len(value) if value else 0
            else:
                edge_dict[key] = value
        edge_data.append(edge_dict)

    edges_df = pd.DataFrame(edge_data)
    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)


    print("=" * 80)
    print("NODES DATAFRAME")
    print("=" * 80)
    print(nodes_df)
    print(f"\nShape: {nodes_df.shape}")
    print(f"Columns: {list(nodes_df.columns)}")

    print("\n" + "=" * 80)
    print("EDGES DATAFRAME")
    print("=" * 80)
    print(edges_df)
    print(f"\nShape: {edges_df.shape}")
    print(f"Columns: {list(edges_df.columns)}")
    print("=" * 80)

    return nodes_df, edges_df
