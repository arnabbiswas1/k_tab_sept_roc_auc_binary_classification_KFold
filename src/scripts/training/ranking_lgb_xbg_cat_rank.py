"""
Ranking: LGB, XGB, Cat with and without imputation
"""

import os
from timeit import default_timer as timer
from datetime import datetime
from functools import reduce

import pandas as pd

import src.common as common
import src.config.constants as constants
import src.munging as process_data
import src.modeling as train_util

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

sub_2_predition_name = (
    "sub_lgb_K5_nonull_mean_sum_max_no_imp_no_scaler_params_K_0922_1420_0.81623.gz"
)
sub_2_oof_name = (
    "oof_lgb_K5_nonull_mean_sum_max_no_imp_no_scaler_params_K_0922_1420_0.81623.csv"
)

sub_2_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_2_predition_name}")
sub_2_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_2_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_2_test_pred.shape}, {sub_2_oof_pred.shape}"
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

sub_4_predition_name = (
    "sub_xgb_K10_nonull_mean_sum_max_no_imp_no_scaler_K_params_0922_1630_0.81634.gz"
)
sub_4_oof_name = (
    "oof_xgb_K10_nonull_mean_sum_max_no_imp_no_scaler_K_params_0922_1630_0.81634.csv"
)

sub_4_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_4_predition_name}")
sub_4_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_4_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_4_test_pred.shape}, {sub_4_oof_pred.shape}"
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


sub_6_predition_name = (
    "sub_cat_K10_nonull_mean_sum_max_noImp_noScaler_K_params_0922_0747_0.81549.gz"
)
sub_6_oof_name = (
    "oof_cat_K10_nonull_mean_sum_max_noImp_noScaler_K_params_0922_0747_0.81549.csv"
)

sub_6_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_6_predition_name}")
sub_6_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_6_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_6_test_pred.shape}, {sub_6_oof_pred.shape}"
)

oof_dfs = [
    sub_1_oof_pred,
    sub_2_oof_pred,
    sub_3_oof_pred,
    sub_4_oof_pred,
    sub_5_oof_pred,
    sub_6_oof_pred,
]
l1_train_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), oof_dfs
)
l1_train_df.columns = ["id", "sub_1", "sub_2", "sub_3", "sub_4", "sub_5", "sub_6"]
l1_train_df = l1_train_df.set_index("id")

prediction_dfs = [
    sub_1_test_pred,
    sub_2_test_pred,
    sub_3_test_pred,
    sub_4_test_pred,
    sub_5_test_pred,
    sub_6_test_pred,
]
l1_test_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), prediction_dfs
)
l1_test_df.columns = ["id", "sub_1", "sub_2", "sub_3", "sub_4", "sub_5", "sub_6"]
l1_test_df = l1_test_df.set_index("id")

test_prediction = train_util.get_rank_mean(l1_test_df)
oof_prediction = train_util.get_rank_mean(l1_train_df)

oof_score = round(
    train_util._calculate_perf_metric("roc_auc", train_df[TARGET], oof_prediction), 6
)

logger.info(f"OOF Score {oof_score}")
common.update_tracking(
    RUN_ID, "oof_score_", oof_score, is_integer=False, no_of_digits=6
)
common.update_tracking(RUN_ID, "lb_score", 0, is_integer=False)

sample_submission_df[TARGET] = test_prediction.values
common.save_file(
    logger,
    sample_submission_df,
    constants.SUBMISSION_DIR,
    f"sub_{MODEL_NAME}_{RUN_ID}_{oof_score:.5f}.csv",
    compression="gzip"
)

# common.save_artifacts(
#     logger,
#     is_test=False,
#     is_plot_fi=False,
#     result_dict=results_dict,
#     submission_df=sample_submission_df,
#     train_index=None,
#     model_number=MODEL_NAME,
#     run_id=RUN_ID,
#     sub_dir=constants.SUBMISSION_DIR,
#     oof_dir=constants.OOF_DIR,
#     fi_dir=constants.FI_DIR,
#     fi_fig_dir=constants.FI_FIG_DIR,
# )

end = timer()
common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
