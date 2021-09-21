"""
"Cat,K10,non-null,full data,mean,sum,max,bins/params from k"
params: https://www.kaggle.com/jonigooner/catboost-classifier
features: https://www.kaggle.com/dlaststark/tps-sep-single-lightgbm-model
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

SEED = 666
EXP_DETAILS = "cat,K10,non-null,full data,mean,sum,max,bins/params from k"

TARGET = "claim"
N_SPLIT = 10

MODEL_TYPE = "cat"
BOOSTING_TYPE = "Bernoulli"
OBJECTIVE = "CrossEntropy"
EVAL_METRIC = "AUC"
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
    "boosting_type": BOOSTING_TYPE,
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

train_df["f5_bin"] = train_df["f5"].apply(lambda x: 0 if x <= 0.05 else 1)
train_df["f29_bin"] = train_df["f29"].apply(lambda x: 0 if x <= 0.5 else 1)
train_df["f40_bin"] = train_df["f40"].apply(lambda x: 0 if x <= 0.5 else 1)
train_df["f42_bin"] = train_df["f42"].apply(
    lambda x: 0 if x <= 0.3 else 1 if x > 0.3 and x <= 0.8 else 2
)
train_df["f50_bin"] = train_df["f50"].apply(lambda x: 0 if x <= 0.05 else 1)
train_df["f65_bin"] = train_df["f65"].apply(lambda x: 0 if x <= 50000 else 1)
train_df["f70_bin"] = train_df["f70"].apply(lambda x: 0 if x <= 0.5 else 1)
train_df["f74_bin"] = train_df["f74"].apply(lambda x: 0 if x <= 3e12 else 1)
train_df["f75_bin"] = train_df["f75"].apply(lambda x: 0 if x <= 0.5 else 1)
train_df["f91_bin"] = train_df["f91"].apply(lambda x: 0 if x <= 0.05 else 1)

test_df["f5_bin"] = test_df["f5"].apply(lambda x: 0 if x <= 0.05 else 1)
test_df["f29_bin"] = test_df["f29"].apply(lambda x: 0 if x <= 0.5 else 1)
test_df["f40_bin"] = test_df["f40"].apply(lambda x: 0 if x <= 0.5 else 1)
test_df["f42_bin"] = test_df["f42"].apply(
    lambda x: 0 if x <= 0.3 else 1 if x > 0.3 and x <= 0.8 else 2
)
test_df["f50_bin"] = test_df["f50"].apply(lambda x: 0 if x <= 0.05 else 1)
test_df["f65_bin"] = test_df["f65"].apply(lambda x: 0 if x <= 50000 else 1)
test_df["f70_bin"] = test_df["f70"].apply(lambda x: 0 if x <= 0.5 else 1)
test_df["f74_bin"] = test_df["f74"].apply(lambda x: 0 if x <= 3e12 else 1)
test_df["f75_bin"] = test_df["f75"].apply(lambda x: 0 if x <= 0.5 else 1)
test_df["f91_bin"] = test_df["f91"].apply(lambda x: 0 if x <= 0.05 else 1)

combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])

features_df = pd.read_parquet(
    f"{constants.FEATURES_DATA_DIR}/features_row_wise_stat.parquet"
)
features_to_use = ["no_null", "mean", "sum", "max"]

features_df = features_df[features_to_use]
combined_df = pd.concat([combined_df, features_df], axis=1)

# features_to_drop = ["f40"]
# combined_df = combined_df.drop(features_to_drop, axis=1)

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
