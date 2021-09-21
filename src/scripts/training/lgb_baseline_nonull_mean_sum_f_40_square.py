"""
LGB K10, non-null, full data, mean, sum, max, f40_square, drop f40
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
EXP_DETAILS = "LGB K10, non-null, full data, mean, sum, max, f40_square, drop f40"

TARGET = "claim"
N_SPLIT = 5

MODEL_TYPE = "lgb"
OBJECTIVE = "binary"
BOOSTING_TYPE = "gbdt"
METRIC = "auc"
VERBOSE = 100
N_THREADS = 8
NUM_LEAVES = 31
MAX_DEPTH = -1
N_ESTIMATORS = 10000
LEARNING_RATE = 0.1
EARLY_STOPPING_ROUNDS = 100


lgb_params = {
    "objective": OBJECTIVE,
    "boosting_type": BOOSTING_TYPE,
    "learning_rate": LEARNING_RATE,
    "num_leaves": NUM_LEAVES,
    "tree_learner": "serial",
    "n_jobs": N_THREADS,
    "seed": SEED,
    "max_depth": MAX_DEPTH,
    "max_bin": 255,
    "metric": METRIC,
    "verbose": -1,
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

features_df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/all_combined.parquet")

target = train_df[TARGET]

# train_df = features_df.loc[train_df.index]
# train_df[TARGET] = target

# test_df = features_df.loc[test_df.index]

train_index = train_df.index
test_index = test_df.index

del train_df, test_df
common.trigger_gc(logger)

features_to_keep = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    "f25",
    "f26",
    "f27",
    "f28",
    "f29",
    "f30",
    "f31",
    "f32",
    "f33",
    "f34",
    "f35",
    "f36",
    "f37",
    "f38",
    "f39",

    "f41",
    "f42",
    "f43",
    "f44",
    "f45",
    "f46",
    "f47",
    "f48",
    "f49",
    "f50",
    "f51",
    "f52",
    "f53",
    "f54",
    "f55",
    "f56",
    "f57",
    "f58",
    "f59",
    "f60",
    "f61",
    "f62",
    "f63",
    "f64",
    "f65",
    "f66",
    "f67",
    "f68",
    "f69",
    "f70",
    "f71",
    "f72",
    "f73",
    "f74",
    "f75",
    "f76",
    "f77",
    "f78",
    "f79",
    "f80",
    "f81",
    "f82",
    "f83",
    "f84",
    "f85",
    "f86",
    "f87",
    "f88",
    "f89",
    "f90",
    "f91",
    "f92",
    "f93",
    "f94",
    "f95",
    "f96",
    "f97",
    "f98",
    "f99",
    "f100",
    "f101",
    "f102",
    "f103",
    "f104",
    "f105",
    "f106",
    "f107",
    "f108",
    "f109",
    "f110",
    "f111",
    "f112",
    "f113",
    "f114",
    "f115",
    "f116",
    "f117",
    "f118",

    "no_null",
    "mean",
    "sum",
    "max",
    "f40_square",
]

features_df = features_df[features_to_keep]

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


results_dict = model.lgb_train_validate_on_cv(
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
    retrain=True
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
