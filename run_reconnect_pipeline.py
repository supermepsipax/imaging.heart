from pipelines import reconnect_mask_batch
from utilities import load_config

config = load_config('reconnect_config.yaml')

reconnect_mask_batch(config=config)
