from pipelines import process_batch_arteries
from utilities import load_config

config = load_config('process_config.yaml')

process_batch_arteries(config=config)

#FILE ISSUE
#DISEASED 6 - incorreclty labelled remus, input is too aligned with LDA
#DISEASED 8 - original origin node was removed during filtering process to remove short branches. 

