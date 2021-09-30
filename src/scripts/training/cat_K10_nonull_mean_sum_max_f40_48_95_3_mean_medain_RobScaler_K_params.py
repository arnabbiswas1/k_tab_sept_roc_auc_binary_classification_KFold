"""
"Cat,K10,SImp,RobScaler,no null,mean,sum,max,f40,48,95,3,params from k"
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
EXP_DETAILS = "Cat,K10,SImp,RobScaler,no null,mean,sum,max,params from k"

TARGET = "claim"
N_SPLIT = 5

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

predictors = list(combined_df.columns)

features_to_use = ["no_null", "mean", "sum", "max"]

features_df = features_df[features_to_use]
combined_df = pd.concat([combined_df, features_df], axis=1)

fill_value_dict = {
    "f1": "Mean",
    "f2": "Mean",
    "f3": "Mode",
    "f4": "Mode",
    "f5": "Mode",
    "f6": "Mean",
    "f7": "Mean",
    "f8": "Median",
    "f9": "Mode",
    "f10": "Mode",
    "f11": "Mode",
    "f12": "Median",
    "f13": "Mode",
    "f14": "Median",
    "f15": "Mean",
    "f16": "Median",
    "f17": "Mode",
    "f18": "Median",
    "f19": "Median",
    "f20": "Median",
    "f21": "Median",
    "f22": "Mean",
    "f23": "Mode",
    "f24": "Median",
    "f25": "Median",
    "f26": "Median",
    "f27": "Median",
    "f28": "Median",
    "f29": "Mean",
    "f30": "Median",
    "f31": "Mode",
    "f32": "Median",
    "f33": "Median",
    "f34": "Mean",
    "f35": "Median",
    "f36": "Median",
    "f37": "Median",
    "f38": "Mode",
    "f39": "Median",
    "f40": "Mean",
    "f41": "Median",
    "f42": "Mean",
    "f43": "Mode",
    "f44": "Median",
    "f45": "Median",
    "f46": "Mean",
    "f47": "Mean",
    "f48": "Median",
    "f49": "Mode",
    "f50": "Mean",
    "f51": "Median",
    "f52": "Median",
    "f53": "Median",
    "f54": "Median",
    "f55": "Mode",
    "f56": "Mean",
    "f57": "Mean",
    "f58": "Median",
    "f59": "Median",
    "f60": "Mode",
    "f61": "Mode",
    "f62": "Median",
    "f63": "Median",
    "f64": "Median",
    "f65": "Mean",
    "f66": "Mode",
    "f67": "Median",
    "f68": "Median",
    "f69": "Mode",
    "f70": "Mean",
    "f71": "Median",
    "f72": "Median",
    "f73": "Median",
    "f74": "Median",
    "f75": "Mean",
    "f76": "Mean",
    "f77": "Median",
    "f78": "Median",
    "f79": "Median",
    "f80": "Median",
    "f81": "Median",
    "f82": "Median",
    "f83": "Median",
    "f84": "Median",
    "f85": "Median",
    "f86": "Median",
    "f87": "Median",
    "f88": "Median",
    "f89": "Median",
    "f90": "Mean",
    "f91": "Mode",
    "f92": "Median",
    "f93": "Median",
    "f94": "Mode",
    "f95": "Median",
    "f96": "Median",
    "f97": "Mean",
    "f98": "Median",
    "f99": "Median",
    "f100": "Mean",
    "f101": "Median",
    "f102": "Median",
    "f103": "Median",
    "f104": "Median",
    "f105": "Mode",
    "f106": "Median",
    "f107": "Median",
    "f108": "Median",
    "f109": "Median",
    "f110": "Mode",
    "f111": "Median",
    "f112": "Median",
    "f113": "Median",
    "f114": "Median",
    "f115": "Mode",
    "f116": "Median",
    "f117": "Median",
    "f118": "Mean",
}

for col in predictors:
    if fill_value_dict.get(col) == "Mean":
        fill_value = combined_df[col].mean()
    elif fill_value_dict.get(col) == "Median":
        fill_value = combined_df[col].median()
    elif fill_value_dict.get(col) == "Mode":
        fill_value = combined_df[col].mode().iloc[0]
    combined_df[col] = combined_df[col].fillna(fill_value)

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

sc = RobustScaler()
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
