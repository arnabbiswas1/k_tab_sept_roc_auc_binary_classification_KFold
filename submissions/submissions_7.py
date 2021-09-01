import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_benchmark_log_loss_SKFold_0807_1325_2.95232.csv"
SUBMISSION_MESSAGE = '"LGB with log_loss SKFold 10"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
df.loss = df.loss.round()
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
