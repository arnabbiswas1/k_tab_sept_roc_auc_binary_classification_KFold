import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_ts_f1_weighted_SK_10_top_100_features_0826_1027_0.08928.csv"
SUBMISSION_MESSAGE = '"LGB ts f1-weighted SKFold 10 top 100 features"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

os.system(submission_string)
