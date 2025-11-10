
def merge_branch_metrics(dict1, dict2, check_common_keys=True):
    """
    Merges two dictionaries with nested dictionary values.

    Takes two dictionaries where each key maps to another dictionary (e.g., branch metrics
    where each branch has multiple measurement types). Combines the nested dictionaries,
    optionally validating that any keys appearing in both nested dicts have matching values.

    Args:
        dict1: First dictionary with nested dict values (e.g., branch lengths)
        dict2: Second dictionary with nested dict values (e.g., branch diameters)
        check_common_keys (bool): If True, validates that common keys have matching values

    Returns:
        merged_dict: Dictionary with all keys from both nested dictionaries merged

    Raises:
        ValueError: If check_common_keys is True and matching keys have different values
    """
    merged = {}

    all_branches = set(dict1.keys()) | set(dict2.keys())

    for branch in all_branches:
        merged[branch] = {}

        if branch in dict1:
            merged[branch].update(dict1[branch])

        if branch in dict2:
            for key, value in dict2[branch].items():
                if check_common_keys and key in merged[branch]:
                    if merged[branch][key] != value:
                        raise ValueError(
                            f"Mismatch in branch '{branch}' for key '{key}': "
                            f"{merged[branch][key]} != {value}"
                        )
                else:
                    merged[branch][key] = value

    return merged
