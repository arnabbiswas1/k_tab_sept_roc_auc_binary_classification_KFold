"""
"Stacking: LGB, XGB, Cat with and without imputation (old & new LGBs),tsne,logistic"
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
EXP_DETAILS = "Stacking: LGB, XGB, Cat with and without imputation (old & new LGBs),tsne,logistic"
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
    "sub_lgb_K5_nonull_mean_sum_max_no_imp_no_scaler_params_K_0924_1159_0.81605.gz"
)
sub_1_oof_name = (
    "oof_lgb_K5_nonull_mean_sum_max_no_imp_no_scaler_params_K_0924_1159_0.81605.csv"
)

sub_1_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_1_predition_name}")
sub_1_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_1_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_1_test_pred.shape}, {sub_1_oof_pred.shape}"
)

sub_2_predition_name = (
    "sub_lgb_K10_nonull_mean_sum_max_mean_imp_no_scaler_params_K_0924_1406_0.81633.gz"
)
sub_2_oof_name = (
    "oof_lgb_K10_nonull_mean_sum_max_mean_imp_no_scaler_params_K_0924_1406_0.81633.csv"
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


# New submissions

sub_7_predition_name = "sub_lgb_K10_nonull_mean_sum_max_40_48_95_3_mean_imp_no_scaler_params_K_0928_1536_0.81645.gz"
sub_7_oof_name = "oof_lgb_K10_nonull_mean_sum_max_40_48_95_3_mean_imp_no_scaler_params_K_0928_1536_0.81645.csv"

sub_7_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_7_predition_name}")
sub_7_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_7_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_7_test_pred.shape}, {sub_7_oof_pred.shape}"
)

sub_8_predition_name = "sub_lgb_K10_nonull_mean_sum_max_40_48_95_3_no_imp_no_scaler_params_K_0927_2311_0.81622.gz"
sub_8_oof_name = "oof_lgb_K10_nonull_mean_sum_max_40_48_95_3_no_imp_no_scaler_params_K_0927_2311_0.81622.csv"

sub_8_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_8_predition_name}")
sub_8_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_8_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_8_test_pred.shape}, {sub_8_oof_pred.shape}"
)

# tsne
sub_9_predition_name = "sub_lgb_K5_nonull_mean_sum_max_tsne_0917_1621_0.81337.gz"
sub_9_oof_name = "oof_lgb_K5_nonull_mean_sum_max_tsne_0917_1621_0.81337.csv"

sub_9_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_9_predition_name}")
sub_9_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_9_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_9_test_pred.shape}, {sub_9_oof_pred.shape}"
)

# all features : Didn't improve the score
sub_10_predition_name = "sub_lgb_all_features_0916_1816_0.81303.gz"
sub_10_oof_name = "oof_lgb_all_features_0916_1816_0.81303.csv"

sub_10_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_10_predition_name}")
sub_10_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_10_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_10_test_pred.shape}, {sub_10_oof_pred.shape}"
)

# Logistic Regression
sub_11_predition_name = "sub_logistic_K10_nonull_mean_sum_max_f40_48_95_3_no_imp_no_scaler_K_params_0928_1259_0.79925.gz"
sub_11_oof_name = "oof_logistic_K10_nonull_mean_sum_max_f40_48_95_3_no_imp_no_scaler_K_params_0928_1259_0.79925.csv"

sub_11_test_pred = pd.read_csv(f"{constants.SUBMISSION_DIR}/{sub_11_predition_name}")
sub_11_oof_pred = pd.read_csv(f"{constants.OOF_DIR}/{sub_11_oof_name}")
logger.info(
    f"Shape of submission and oof file {sub_11_test_pred.shape}, {sub_11_oof_pred.shape}"
)


oof_dfs = [
    sub_1_oof_pred,
    sub_2_oof_pred,
    sub_3_oof_pred,
    sub_4_oof_pred,
    sub_5_oof_pred,
    sub_6_oof_pred,
    sub_7_oof_pred,
    sub_8_oof_pred,
    sub_9_oof_pred,
    # sub_10_oof_pred,
    sub_11_oof_pred,
]
l1_train_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), oof_dfs
)
l1_train_df.columns = [
    "id",
    "sub_1",
    "sub_2",
    "sub_3",
    "sub_4",
    "sub_5",
    "sub_6",
    "sub_7",
    "sub_8",
    "sub_9",
    # "sub_10",
    "sub_11",
]
l1_train_df = l1_train_df.set_index("id")

prediction_dfs = [
    sub_1_test_pred,
    sub_2_test_pred,
    sub_3_test_pred,
    sub_4_test_pred,
    sub_5_test_pred,
    sub_6_test_pred,
    sub_7_test_pred,
    sub_8_test_pred,
    sub_9_test_pred,
    # sub_10_test_pred,
    sub_11_test_pred,
]
l1_test_df = reduce(
    lambda left, right: pd.merge(left, right, on=["id"], how="left"), prediction_dfs
)
l1_test_df.columns = [
    "id",
    "sub_1",
    "sub_2",
    "sub_3",
    "sub_4",
    "sub_5",
    "sub_6",
    "sub_7",
    "sub_8",
    "sub_9",
    # "sub_10",
    "sub_11",
]
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
# sk_model = LogisticRegression()

results_dict = model.sklearn_train_validate_on_cv(
    logger=logger,
    run_id=RUN_ID,
    sklearn_model=sk_model,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    kf=sk,
    features=features,
    metric="roc_auc",
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
