"""
"Cat,K10,no imputer,no scaler, no null,mean,sum,max,f40,48,95,3,params from k"
params: https://www.kaggle.com/jonigooner/catboost-classifier
"""

import os
from timeit import default_timer as timer
from datetime import datetime
import numpy as np


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

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
EXP_DETAILS = "Cat,K10,no imputer,no scaler, no null,mean,sum,max,f40,48,95,3,params from k"

TARGET = "claim"
N_SPLIT = 10

MODEL_TYPE = "cat"
BOOSTING_TYPE = "Bernoulli"
OBJECTIVE = "CrossEntropy"
EVAL_METRIC = "AUC"
# 15585
N_ESTIMATORS = 20000
LEARNING_RATE = 0.023575206684596582
EARLY_STOPPING_ROUNDS = 500
USE_BEST_MODEL = True
N_THREADS = 8
VERBOSE_EVAL = 100
MAX_DEPTH = 7

MAX_BIN = 254
NUM_LEAVES = 31

# Silent, Verbose, Info, Debug
LOG_LEVEL = "Verbose"

cat_params = {
    "bootstrap_type": BOOSTING_TYPE,
    "objective": OBJECTIVE,  # Alias loss_function
    "eval_metric": EVAL_METRIC,
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "random_seed": SEED,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "use_best_model": USE_BEST_MODEL,
    "max_depth": MAX_DEPTH,
    "thread_count": N_THREADS,
    "verbose_eval": VERBOSE_EVAL,

    'reg_lambda': 36.30433203563295,
    'random_strength': 43.75597655616195,
    'min_data_in_leaf': 11,
    'leaf_estimation_iterations': 1,
    'subsample': 0.8227911142845009,
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

kfold = KFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
common.update_tracking(RUN_ID, "cv_method", "KFold")
common.update_tracking(RUN_ID, "n_fold", N_SPLIT)


results_dict = model.cat_train_validate_on_cv(
    logger,
    run_id=RUN_ID,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    metric="roc_auc",
    kf=kfold,
    features=predictors,
    params=cat_params,
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
