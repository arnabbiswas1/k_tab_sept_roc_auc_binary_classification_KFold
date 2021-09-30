"""
LGB,SK10,non-null,mean,sum,max,f40,48,95,3,mean encoding,mean impute,no-scaler
params : https://www.kaggle.com/arnabbiswas1/tps-sep-2021-single-lgbm
"""

import os
from timeit import default_timer as timer
from datetime import datetime


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as model

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split(".")[0]

SEED = 2021
EXP_DETAILS = "LGB,SK10,non-null,mean,sum,max,f40,48,95,3,mean encoding,mean impute,no-scaler"

TARGET = "claim"
N_SPLIT = 10

MODEL_TYPE = "lgb"
OBJECTIVE = "binary"
BOOSTING_TYPE = "gbdt"
METRIC = "auc"
VERBOSE = 100
N_THREADS = 8
NUM_LEAVES = 6
MAX_DEPTH = 2
N_ESTIMATORS = 20000
LEARNING_RATE = 5e-3
EARLY_STOPPING_ROUNDS = 200


lgb_params = {
    "objective": OBJECTIVE,
    "boosting_type": BOOSTING_TYPE,
    "learning_rate": LEARNING_RATE,
    # "num_leaves": NUM_LEAVES,
    "n_jobs": N_THREADS,
    "seed": SEED,
    # "max_depth": MAX_DEPTH,
    # "metric": METRIC,
    "verbose": -1,

    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.4,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-1,
    'min_child_weight': 256,
    'min_child_samples': 20,
}

LOGGER_NAME = "sub_1"
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

common.update_tracking(RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
common.update_tracking(RUN_ID, "metric", "roc_auc")
common.update_tracking(RUN_ID, "n_estimators", N_ESTIMATORS)
common.update_tracking(RUN_ID, "learning_rate", LEARNING_RATE)
common.update_tracking(RUN_ID, "num_leaves", NUM_LEAVES)
common.update_tracking(RUN_ID, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)

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
predictors = list(combined_df.columns)

combined_df[predictors] = combined_df[predictors].fillna(combined_df[predictors].mean())

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

kfold = StratifiedKFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
common.update_tracking(RUN_ID, "cv_method", "KFold")
common.update_tracking(RUN_ID, "n_fold", N_SPLIT)


results_dict = model.lgb_train_validate_on_cv_mean_encoding(
    logger,
    run_id=RUN_ID,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    metric="roc_auc",
    kf=kfold,
    features=predictors,
    params=lgb_params,
    n_estimators=N_ESTIMATORS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    cat_features="auto",
    verbose_eval=100,
    retrain=False,
    target_val="claim",
    cat_enc_cols=["no_null"]
)

common.update_tracking(RUN_ID, "lb_score", 0, is_integer=True)

train_index = train_df.index

common.save_artifacts(
    logger,
    target=TARGET,
    is_plot_fi=True,
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
