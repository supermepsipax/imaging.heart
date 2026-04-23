from pipelines.traversal_comparison_pipeline import traversal_comparison
from utilities import load_config

config = load_config('traversal_config.yaml')

traversal_comparison(config=config)
