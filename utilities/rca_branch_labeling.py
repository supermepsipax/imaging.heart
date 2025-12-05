import networkx as nx


def is_side_branch(edge_position, parent_edge_position):
    """
    Determines if an edge is a side branch based on edge position labels.

    Side branches have a digit increase at the end (e.g., 11→12, 111→112).
    Distal continuations keep adding the same digit (e.g., 11→111, 12→122).

    Args:
        edge_position (str): Edge position label
        parent_edge_position (str): Parent edge position label

    Returns:
        bool: True if this is a side branch
    """
    if not edge_position.startswith(parent_edge_position):
        return False

    if len(edge_position) != len(parent_edge_position) + 1:
        return False

    last_digit_parent = parent_edge_position[-1] if parent_edge_position else '0'
    last_digit_child = edge_position[-1]

    return last_digit_child > last_digit_parent


def annotate_rca_graph_with_branch_labels(graph):
    """
    Annotates an RCA graph by adding 'rca_branch' attribute to all edges.

    Labeling scheme:
    - Main RCA trunk (all 1's): 'RCA'
    - Side branches: 'AM1', 'AM2', 'AM3', ... (Acute Marginal branches)

    The main RCA trunk includes all edges where the position is only 1's
    (e.g., "1", "11", "111", "1111").

    Side branches are any edges that branch off the main trunk or its branches.

    Args:
        graph (nx.DiGraph): RCA directed graph with edge_position labels

    Returns:
        nx.DiGraph: Updated graph with 'rca_branch' attributes
    """
    updated_graph = nx.DiGraph(graph)

    am_counter = 0

    all_edges_with_pos = [
        (edge, updated_graph.edges[edge].get('edge_position', ''))
        for edge in updated_graph.edges()
    ]
    all_edges_with_pos.sort(key=lambda x: (len(x[1]), x[1]))

    for edge, edge_pos in all_edges_with_pos:
        if edge_pos and all(c == '1' for c in edge_pos):
            updated_graph.edges[edge]['rca_branch'] = 'RCA'

    for edge, edge_pos in all_edges_with_pos:
        if not edge_pos:
            continue

        if 'rca_branch' in updated_graph.edges[edge]:
            continue

        if len(edge_pos) > 1:
            parent_pos = edge_pos[:-1]

            if is_side_branch(edge_pos, parent_pos):
                am_counter += 1
                label = f"AM{am_counter}"
                updated_graph.edges[edge]['rca_branch'] = label
            else:
                parent_edge = next(
                    (e for e in updated_graph.edges()
                     if updated_graph.edges[e].get('edge_position') == parent_pos),
                    None
                )
                if parent_edge:
                    parent_label = updated_graph.edges[parent_edge].get('rca_branch', 'RCA')
                    updated_graph.edges[edge]['rca_branch'] = parent_label
                else:
                    updated_graph.edges[edge]['rca_branch'] = 'RCA'

    rca_trunk_count = sum(1 for e in updated_graph.edges()
                          if updated_graph.edges[e].get('rca_branch') == 'RCA')

    print(f"\n[RCA Branch Labeling]")
    print(f"                      RCA main trunk: {rca_trunk_count} edges")
    if am_counter > 0:
        print(f"                      Acute Marginal branches (AM): {am_counter}")

    updated_graph.graph['rca_labeling'] = {
        'main_trunk_edges': rca_trunk_count,
        'num_acute_marginal_branches': am_counter
    }

    return updated_graph
