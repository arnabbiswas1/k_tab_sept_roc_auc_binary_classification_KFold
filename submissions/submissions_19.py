"""
sub_6 = pd.read_csv(f"{constants.SUBMISSION_DIR}/sub_xgb_K10_nonull_mean_sum_max_no_imp_no_scaler_K_params_0922_1630_0.81634.gz")
sub_2 = pd.read_csv(f"{constants.SUBMISSION_DIR}/sub_xgb_K10_nonull_mean_sum_max_custom_imp_StScaler_K_params_0921_2239_0.81649.gz")
sub_5 = pd.read_csv(f"{constants.SUBMISSION_DIR}/sub_lgb_K5_nonull_mean_sum_max_no_imp_no_scaler_params_K_0922_1420_0.81623.gz")
"""

import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-sep-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_power_8_xgb_no_imp_xgb_imp_lgb_no_imp.gz"
SUBMISSION_MESSAGE = '"Power Averaging 8"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
