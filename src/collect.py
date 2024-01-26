"""Response Collection: Aggregate Model Responses into a single csv per model"""

import os
import pickle
import argparse
import pandas as pd

from src.config import PATH_RESULTS


################################################################################################
# ARGUMENT PARSER
################################################################################################
parser = argparse.ArgumentParser(description="Collecting Results")
parser.add_argument(
    "--experiment-name",
    default="test",
    type=str,
    help="Name of Experiment - used for logging",
)
parser.add_argument(
    "--dataset", default="high", type=str, help="Dataset to evaluate (low or high)"
)

args = parser.parse_args()


################################################################################################
# SETUP
################################################################################################
path_results = f"{PATH_RESULTS}/{args.experiment_name}/{args.dataset}"
path_results_raw = path_results + "_raw"


################################################################################################
# RESPONSE COLLECTION
################################################################################################
# Collect all pickle result files
results = []
for path, subdirs, files in os.walk(path_results_raw):
    for name in files:
        if name[:-7] == ".pickle":
            path_file = os.path.join(path, name)

        with open(path_file, "rb") as f:
            tmp = pickle.load(f)
            results.append(tmp)

df_results = pd.concat(results)

# Store one csv per model
if not os.path.exists(path_results):
    os.makedirs(path_results)

for model_id in df_results["model_id"].unique():
    results_model = df_results.loc[df_results["model_id"] == model_id]
    results_model.to_csv(
        f"{path_results}/{model_id.split('/')[0]}_{model_id.split('/')[-1]}.csv"
    )
