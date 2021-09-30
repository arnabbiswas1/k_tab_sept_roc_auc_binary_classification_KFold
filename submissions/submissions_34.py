
import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-sep-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_stacking_lgb_xbg_cat_imputer_no_imputer_v2_0930_1422_0.81687.gz"
SUBMISSION_MESSAGE = '"Stacking: LGB, XGB, Cat with and without imputation old/new LGBs/XGB/Cat,tsne,LR,mean/median LGB/XGB/cat, all LGB"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
