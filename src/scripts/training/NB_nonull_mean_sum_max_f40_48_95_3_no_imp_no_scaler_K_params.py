"""
"LR,K10,simple imputer,standard scaler,no null,mean,sum,max,f40,48,95,3,params from K"
params: https://www.kaggle.com/mustafacicek/tps-09-21-xgboost-0-81785
"""

import os
from timeit import default_timer as timer
from datetime import datetime


import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as model

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split(".")[0]

SEED = 666
EXP_DETAILS = (
    "LR,K10,simple imputer,standard scaler,no null,mean,sum,max,f40,48,95,3,params from K"
)

TARGET = "claim"
N_SPLIT = 10

MODEL_TYPE = "logistic_regression"

LOGGER_NAME = "sub_1"
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

common.update_tracking(RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
common.update_tracking(RUN_ID, "metric", "roc_auc")

train_df, test_df, sample_submission_df = process_data.read_processed_data(
    logger, constants.PROCESSED_DATA_DIR, train=True, test=True, sample_submission=True,
)

combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])

features_df = pd.read_parquet(
    f"{constants.FEATURES_DATA_DIR}/features_row_wise_stat.parquet"
)
features_to_use = ["no_null", "mean", "sum", "max"]

features_df = features_df[features_to_use]
combined_df = pd.concat([combined_df, features_df], axis=1)

combined_df["f40_bin"] = pd.cut(
    combined_df.f40,
    bins=[combined_df.f40.min(), 0.04, 0.14, 0.936, combined_df.f40.max()],
    labels=[0, 1, 2, 3],
)

combined_df["f48"] = np.log1p(combined_df["f48"])

combined_df["f95_log"] = np.log1p(
    combined_df["f95"].clip(lower=0, upper=combined_df["f95"].max())
)

combined_df["f3_cbrt"] = np.cbrt(combined_df["f3"])

predictors = list(combined_df.columns)

imputer = SimpleImputer()
combined_df = imputer.fit_transform(combined_df)

sc = MinMaxScaler()
combined_df = sc.fit_transform(combined_df)
combined_df = pd.DataFrame(combined_df, columns=predictors)

target = train_df[TARGET]

train_df = combined_df.loc[train_df.index]
train_df[TARGET] = target

test_df = combined_df.loc[test_df.index]

train_X = train_df.drop([TARGET], axis=1)
train_Y = train_df[TARGET]
test_X = test_df

logger.info(
    f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
)

predictors = list(train_X.columns)
logger.info(f"List of predictors {predictors}")

kfold = KFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
common.update_tracking(RUN_ID, "cv_method", "KFold")
common.update_tracking(RUN_ID, "n_fold", N_SPLIT)

sklearn_model = GaussianNB()

results_dict = model.sklearn_train_validate_on_cv(
    logger,
    run_id=RUN_ID,
    sklearn_model=sklearn_model,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    kf=kfold,
    features=predictors,
    metric="roc_auc",
)

common.update_tracking(RUN_ID, "lb_score", 0, is_integer=True)

train_index = train_df.index

common.save_artifacts(
    logger,
    target=TARGET,
    is_plot_fi=False,
    result_dict=results_dict,
    submission_df=sample_submission_df,
    train_index=train_index,
    model_number=MODEL_NAME,
    run_id=RUN_ID,
    sub_dir=constants.SUBMISSION_DIR,
    oof_dir=constants.OOF_DIR,
    fi_dir=constants.FI_DIR,
    fi_fig_dir=constants.FI_FIG_DIR,
)

end = timer()
common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
logger.info("Execution Complete")
