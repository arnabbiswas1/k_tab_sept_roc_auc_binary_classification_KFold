"""
Ranking: LGB, XGB, Cat with and without imputation (meta=Logistic)
"""

import os
from timeit import default_timer as timer
from datetime import datetime
from functools import reduce

import pandas as pd

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as model

from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split(".")[0]

SEED = 42
EXP_DETAILS = "Ranking: LGB, XGB, Cat with and without imputation"
IS_TEST = False
PLOT_FEATURE_IMPORTANCE = False

TARGET = "claim"
MODEL_TYPE = "Ranking"

LOGGER_NAME = "ranking"
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

common.update_tracking(RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True)
common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
common.update_tracking(RUN_ID, "metric", "roc_auc")

train_df, test_df, sample_submission_df = process_data.read_processed_data(
    logger, constants.PROCESSED_DATA_DIR, train=True, test=True, sample_submission=True
)

# Read different submission files and merge them to create dataset
# for level 2

sub_1_predition_name = (
    "sub_lgb_K5_nonull_mean_sum_max_mean_imp_no_scaler_params_K_0922_1212_0.81623.gz"
)
sub_1_oof_name = (
    "oof_lgb_K5_nonull_mean_sum_max_mean_imp_no_scaler_params_K_0922_1212_0.81623.csv"
)

sub_1_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_1_predition_name}")
sub_1_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_1_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_1_test_pred.shape}, {sub_1_oof_pred.shape}"
)


sub_3_predition_name = (
    "sub_xgb_K10_nonull_mean_sum_max_custom_imp_StScaler_K_params_0921_2239_0.81649.gz"
)
sub_3_oof_name = (
    "oof_xgb_K10_nonull_mean_sum_max_custom_imp_StScaler_K_params_0921_2239_0.81649.csv"
)

sub_3_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_3_predition_name}")
sub_3_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_3_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_3_test_pred.shape}, {sub_3_oof_pred.shape}"
)

sub_5_predition_name = (
    "sub_cat_K10_nonull_full_data_mean_sum_max_Kaggle_bin_params_0921_2000_0.81612.gz"
)
sub_5_oof_name = (
    "oof_cat_K10_nonull_full_data_mean_sum_max_Kaggle_bin_params_0921_2000_0.81612.csv"
)

sub_5_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_5_predition_name}")
sub_5_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_5_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_5_test_pred.shape}, {sub_5_oof_pred.shape}"
)


oof_dfs = [
    sub_1_oof_pred,
    sub_3_oof_pred,
    sub_5_oof_pred,
]
l1_train_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), oof_dfs
)
l1_train_df.columns = ["id", "sub_1", "sub_3", "sub_5"]
l1_train_df = l1_train_df.set_index("id")

prediction_dfs = [
    sub_1_test_pred,
    sub_3_test_pred,
    sub_5_test_pred,
]
l1_test_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), prediction_dfs
)
l1_test_df.columns = ["id", "sub_1", "sub_3", "sub_5"]
l1_test_df = l1_test_df.set_index("id")

train_X = l1_train_df
train_Y = train_df[TARGET]
test_X = l1_test_df
features = list(train_X.columns)

logger.info("train_X")
logger.info(train_X.head())

logger.info("test_X")
logger.info(test_X.head())

sk = KFold(n_splits=10, shuffle=False)
sk_model = CalibratedClassifierCV(RidgeClassifier(random_state=SEED), cv=10)

results_dict = model.sklearn_train_validate_on_cv(
    logger=logger,
    run_id=RUN_ID,
    sklearn_model=sk_model,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    kf=sk,
    features=features,
    metric="roc_auc"
    )

train_index = train_df.index

logger.info("============")
logger.info(f"prediction {results_dict['prediction']}")
logger.info("============")

common.save_artifacts(
    logger,
    target="claim",
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
