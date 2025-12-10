from classifier import run_classification_pipeline, run_model_comparison
from utilities import load_config

config = load_config('classifier_config.yaml')

# Run the classification pipeline
results = run_classification_pipeline(config=config)

# Uncomment below to compare multiple models instead:
# results = run_model_comparison(
#     input_tar_file=config.get('input_tar_file'),
#     input_folder=config.get('input_folder'),
#     feature_set=config.get('feature_set', 'priority'),
#     verbose=True
# )
