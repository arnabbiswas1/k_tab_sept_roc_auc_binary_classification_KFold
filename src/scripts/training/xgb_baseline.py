"""
XGB benchmark KFold-5
"""

import os
from timeit import default_timer as timer
from datetime import datetime

import pandas as pd
from sklearn.model_selection import KFold

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as model

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split(".")[0]

SEED = 42
EXP_DETAILS = "XGB benchmark KFold-5"

TARGET = "claim"
N_SPLIT = 5

MODEL_TYPE = "xgb"
OBJECTIVE = "binary:logistic"
BOOSTING_TYPE = "gbtree"
TREE_METHOD = "hist"
METRIC = "auc"
N_THREADS = 8
NUM_LEAVES = 31
MAX_DEPTH = 6
LEARNING_RATE = 0.1
MAX_BIN = 256
#  0 (silent), 1 (warning), 2 (info), 3 (debug)
VERBOSITY = 1
VERBOSE_EVAL = 100

N_ESTIMATORS = 10000
EARLY_STOPPING_ROUNDS = 100


xgb_params = {
    "objective": OBJECTIVE,
    "booster": BOOSTING_TYPE,
    "eval_metric": METRIC,
    "learning_rate": LEARNING_RATE,
    "validate_parameters": True,
    "nthread": N_THREADS,
    "max_depth": MAX_DEPTH,
    "tree_method": TREE_METHOD,
    "seed": SEED,
    "max_bin": MAX_BIN,
    "max_depth": MAX_DEPTH,
    "verbosity": VERBOSITY,
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
    logger,
    constants.PROCESSED_DATA_DIR,
    train=True,
    test=True,
    sample_submission=True,
)

combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])
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
