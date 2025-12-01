import nrrd
import json

def load_nrrd_mask(path, verbose=False):
    data, header = nrrd.read(path)

    if verbose:
        print("Shape:", data.shape)
        print("Available Header Info:")
        for heading, value in header.items():
            print(f'{heading}: {value}')

    return data, header


def load_config(config_path):
    """
    Load a JSON configuration file for pipeline parameters.

    Args:
        config_path (str): Path to the JSON config file

    Returns:
        dict: Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
