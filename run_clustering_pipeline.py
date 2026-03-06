from pipelines import cluster_artery_nodes
from utilities import load_config

config = load_config('clustering_config.yaml')

cluster_artery_nodes(config=config)
