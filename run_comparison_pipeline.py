from pipelines import compare_mask_batch
from utilities import load_config

config = load_config('comparison_config.yaml')

compare_mask_batch(config=config)
