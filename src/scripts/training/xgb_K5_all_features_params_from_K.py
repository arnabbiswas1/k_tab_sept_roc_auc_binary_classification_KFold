"""
"XGB K5,non-null,all features, params from K"
params: https://www.kaggle.com/mustafacicek/tps-09-21-xgboost-0-81785
"""

import os
from timeit import default_timer as timer
from datetime import datetime


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    "XGB K5,non-null,all features, params from K"
)

TARGET = "claim"
N_SPLIT = 5

MODEL_TYPE = "xgb"
OBJECTIVE = "binary:logistic"
BOOSTING_TYPE = "gbtree"
TREE_METHOD = "hist"
METRIC = "auc"
N_THREADS = 8
MAX_DEPTH = 6
LEARNING_RATE = 0.005
#  0 (silent), 1 (warning), 2 (info), 3 (debug)
VERBOSITY = 1
VERBOSE_EVAL = 100

N_ESTIMATORS = 40000
EARLY_STOPPING_ROUNDS = 300


xgb_params = {
    "objective": OBJECTIVE,
    "booster": BOOSTING_TYPE,
    "eval_metric": METRIC,
    "learning_rate": LEARNING_RATE,
    "validate_parameters": True,
    "nthread": N_THREADS,
    "tree_method": TREE_METHOD,
    "seed": SEED,
    "max_depth": MAX_DEPTH,
    "verbosity": VERBOSITY,
    "colsample_bytree": 0.7098433872257219,
    "min_child_weight": 482,
    "subsample": 0.8406820875269025,
    "reg_alpha": 3.2594867475105374,
    "reg_lambda": 0.1534227221930378,
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

features_df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/all_combined.parquet")

target = train_df[TARGET]

train_index = train_df.index
test_index = test_df.index

del train_df, test_df
common.trigger_gc(logger)

train_X = features_df.loc[train_index]
train_Y = target
test_X = features_df.loc[test_index]

del features_df, target
common.trigger_gc(logger)

logger.info(
    f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
)

predictors = list(train_X.columns)
logger.info(f"List of predictors {predictors}")

kfold = KFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)

common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
common.update_tracking(RUN_ID, "cv_method", "KFold")
common.update_tracking(RUN_ID, "n_fold", N_SPLIT)


results_dict = model.xgb_train_validate_on_cv(
    logger,
    run_id=RUN_ID,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    metric="roc_auc",
    kf=kfold,
    features=predictors,
    params=xgb_params,
    n_estimators=N_ESTIMATORS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=VERBOSE_EVAL,
)

common.update_tracking(RUN_ID, "lb_score", 0, is_integer=True)

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
