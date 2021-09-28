"""
LGB,KF5,non-null,mean,sum,max,pseudo labeling
"""

import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-sep-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_K5_nonull_mean_sum_max_pseudo_labeling_0924_1239_0.81584.gz"
SUBMISSION_MESSAGE = '"LGB,KF5,non-null,mean,sum,max,pseudo labeling"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
