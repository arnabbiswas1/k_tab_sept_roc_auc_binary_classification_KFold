
import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-sep-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_K10_nonull_mean_sum_max_40_48_95_3_pseudo_labeling_mean_imp_no_scaler_params_K_0929_1237_0.82345.gz"
SUBMISSION_MESSAGE = '"LGB,K10,pseudo_labeling,non-null,mean,sum,max,f40,48,95,3,mean impute,no-scaler,log_loss"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
