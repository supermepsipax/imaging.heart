from pipelines import process_batch_arteries
from utilities import load_config

config = load_config('config.json')

process_batch_arteries(config=config)



