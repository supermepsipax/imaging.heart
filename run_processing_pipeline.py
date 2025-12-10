from pipelines import process_batch_arteries
from utilities import load_config

config = load_config('process_config.yaml')

process_batch_arteries(config=config)


  # • Diseased_10 (type: true)
  # • Diseased_13 (type: pseudo)
  # • Diseased_19 (type: pseudo)
  # • Diseased_2 (type: true)
  # • Diseased_6 (type: pseudo)
  # • Normal_13 (type: pseudo)
  # • Normal_16 (type: true)
  # • Normal_4 (type: true)
  # • Normal_7 (type: pseudo)
