import glob
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# -------------- Functions ----------------------------------
def load_edges_nodes(edges_file):
    """
    Load edges and nodes from CSV-files, and preprocess data.
    
    Args:
        edges_file (str): Path to edges CSV file.

    Returns:
        edges (pd.DataFrame): Edges DataFrame.
        nodes (pd.DataFrame): Nodes DataFrame.
    """

    edges = pd.read_csv(edges_file)
    nodes_file = edges_file.replace("_edges.csv", "_nodes.csv")
    nodes = pd.read_csv(nodes_file)

    if "diameter_profile" in edges.columns:
        edges["diameter_profile"] = edges["diameter_profile"].apply(
            lambda x: eval(x, {"np": np}) if pd.notnull(x) else [])

    return edges, nodes


def compute_segment_metrics(edges, tree, segment):
    """
    Compute diameter and tortuosity metrics for a segment.
    A segment is defined as the artery between two bifurcations.

    Args:
        edges (pd.DataFrame): Edges DataFrame.
        tree (str): Tree name ("LCA" or "RCA").
        segment(str): Segment label ("1", "11" or "12"),

    Returns:
        dict: Dictionary with mean, median and std of diameter, as well as length of segment and tortuosity.
    """

    segment_edges = edges[edges["edge_position"].astype(str) == segment]
    
    if segment_edges.empty:
            
        return {f"{tree}_{segment}_mean_diameter": np.nan,
                f"{tree}_{segment}_median_diameter": np.nan,
                f"{tree}_{segment}_std_diameter": np.nan,
                f"{tree}_{segment}_length": np.nan,
                f"{tree}_{segment}_tortuosity]": np.nan}

    # Add all diameters to a list    
    all_diameters = np.concatenate(segment_edges["diameter_profile"].values)
        
    # For tortuosity
    path_length = segment_edges["path_length_mm"].sum()
    direct_length = segment_edges["direct_length_mm"].sum()

    return{f"{tree}_{segment}_mean_diameter": np.mean(all_diameters),
            f"{tree}_{segment}_median_diameter": np.median(all_diameters),
            f"{tree}_{segment}_std_diameter": np.std(all_diameters),
            f"{tree}_{segment}_length": path_length,
            f"{tree}_{segment}_tortuosity": path_length / direct_length}                      

    
def compute_tree_metrics(edges, tree):
    """
    Compute diameter, tortuosity and number of branches and leaves for a whole tree.
    Includes all segments that start with label "1"

    Args:
        edges (pd.DataFrame): Edges DataFrame.
        tree (str): Tree name ("LCA" or "RCA").
    
    Returns:
        dict: Dictionary with mean, median and std diameter, as well as tortuosity, 
              number of leaves and number of branches for the whole tree.
    """

    branch_edges = edges[edges["edge_position"].astype(str).str.startswith("1")]

    if branch_edges.empty:
            
        return {f"{tree}_whole_branch_mean_diameter": np.nan,
                f"{tree}_whole_branch_median_diameter": np.nan,
                f"{tree}_whole_branch_std_diameter": np.nan,
                f"{tree}_whole_branch_tortuosity": np.nan,
                f"{tree}_num_leaves": np.nan,
                f"{tree}_num_branches": np.nan}

    # Add all diameters to a list    
    all_diameters = np.concatenate(branch_edges["diameter_profile"].values)
        
    # For tortuosity (not correct!)
    total_path = branch_edges["path_length_mm"].sum()
    direct_path = branch_edges["direct_length_mm"].sum()

    # Convert edge labels to strings
    labels = branch_edges["edge_position"].astype(str).tolist()

    # Create dictionary that maps each label to its children
    children = {label: [] for label in labels}

    for parent in labels:
        for child in labels:
            # Child should have one digit more than parent (parent = 11, child = 112)
            if child.startswith(parent) and len(child) == len(parent) + 1:
                children[parent].append(child)

    # Leaves = labels with 0 children
    num_leaves = sum(1 for label, ch in children.items() if len(ch) == 0)

    # Branch points = labels with 2 or more children
    # Unsure regarding how we want to define a branch? 
    num_branches = sum(1 for label, ch in children.items() if len(ch) >= 2)

    return {f"{tree}_whole_branch_mean_diameter": np.mean(all_diameters),
            f"{tree}_whole_branch_median_diameter": np.median(all_diameters),
            f"{tree}_whole_branch_std_diameter": np.std(all_diameters),
            f"{tree}_whole_branch_tortuosity": total_path / direct_path,
            f"{tree}_num_leaves": num_leaves,
            f"{tree}_num_branches": num_branches}    
    
    
def perform_ttests(df, metrics):
    """
    Perform Welch's t-test for each metric, comparing normal and diseased groups.

    Args:
        df (pd.DataFrame): DataFrame with patient statistics, with group labels.
        metrics (list of str): Names of metrics to compare.

    Returns:
        pd.DataFrame: DataFrame with t-test results (t-statistic and p-value) for each metric.
    """

    results = []

    for metric in metrics:
        group1 = df[df["group"] == "Normal"][metric].dropna()
        group2 = df[df["group"] == "Diseased"][metric].dropna()

        if len(group1) > 1 and len(group2) > 1:
            tstat, pvalue = ttest_ind(group1, group2, equal_var = False)

            results.append({
                "metric": metric,
                "normal_mean": group1.mean(),
                "diseased_mean": group2.mean(),
                "t_statistic": tstat,
                "p_value": pvalue,
            })
    return pd.DataFrame(results)
        

# ------------ Main ----------------------------------------

# Branches/segments that are relevant for analysis
main_segments = {
    "LCA": ["1", "11", "12"],
    "RCA": ["1"]
}

folder = "results/test_batch/"
edges_files = glob.glob(os.path.join(folder, "*_edges.csv"))

all_subject_stats = []

for edges_file in edges_files:
    filename = os.path.basename(edges_file)
    names = filename.split("_")
    group = names[0]
    subject_id = names[1]
    tree = names[2]

    edges, nodes = load_edges_nodes(edges_file)

    # Dictionary for storing statistics for each subject
    stats_dict = {"subject_id": subject_id, "group": group, "tree": tree}

    # Segment-wise metrics
    for segment in main_segments.get(tree, []):
        stats_dict.update(compute_segment_metrics(edges, tree, segment))

    # Whole branch (tree) metrics
    stats_dict.update(compute_tree_metrics(edges, tree))

    all_subject_stats.append(stats_dict)

# Save to CSV file
df = pd.DataFrame(all_subject_stats)
df.to_csv("Patient_statistics.csv", index = False)


# Statistical test
df = pd.read_csv("Patient_statistics.csv")

# Extract names of relevant metrics
metrics = [c for c in df.columns if "mean_diameter" in c or "tortuosity" in c]

ttest_df = perform_ttests(df, metrics)

# Save to CSV file
ttest_df.to_csv("Ttest_results.csv", index = False)
