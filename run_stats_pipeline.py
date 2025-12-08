from pipelines import analyze_artery_batch
from utilities import load_config

config = load_config('stats_config.yaml')

analyze_artery_batch(config=config)



