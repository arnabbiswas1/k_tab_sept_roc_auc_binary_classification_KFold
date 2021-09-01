"""
XGB with SK(10) row wise stat features
"""

import os
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd

from sklearn.model_selection import StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.modeling as model
import src.munging as process_data

if __name__ == "__main__":
    common.set_timezone()
    start = timer()

    # Create RUN_ID
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    SEED = 42
    EXP_DETAILS = "XGB with SK(10) row wise stat features"
    IS_TEST = False
    PLOT_FEATURE_IMPORTANCE = False

    TARGET = "loss"

    MODEL_TYPE = "xgb"
    OBJECTIVE = "reg:squarederror"
    NUM_CLASSES = None
    METRIC = "rmse"
    BOOSTING_TYPE = "gbtree"
    VERBOSE = 100
    N_THREADS = -1
    NUM_LEAVES = 31
    MAX_DEPTH = 6
    N_ESTIMATORS = 10000
    LEARNING_RATE = 0.1
    EARLY_STOPPING_ROUNDS = 100

    xgb_params = {
        # Learning task parameters
        "objective": OBJECTIVE,
        "eval_metric": METRIC,
        "seed": SEED,
        # Type of the booster
        "booster": BOOSTING_TYPE,
        # parameters for tree booster
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "max_leaves": NUM_LEAVES,
        "max_bin": 256,
        # General parameters
        "nthread": -1,
        "verbosity": 2,
        "validate_parameters": True,
    }

    LOGGER_NAME = "sub_1"
    logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
    common.set_seed(SEED)
    logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

    common.update_tracking(
        RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True
    )
    common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
    common.update_tracking(RUN_ID, "is_test", IS_TEST)
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

    features_df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/generated_features.parquet")
    logger.info(f"Shape of the features {features_df.shape}")

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])
    orginal_features = list(test_df.columns)
    combined_df = pd.concat([combined_df, features_df], axis=1)

    logger.info(f"Shape of combined data with features {combined_df.shape}")
    feature_names = process_data.get_row_wise_stat_feature_names()

    logger.info(f"Selceting row  wise stat features {feature_names}")
    combined_df = combined_df.loc[:, orginal_features + feature_names]
    logger.info(f"Shape of the data after selecting features {combined_df.shape}")

    train_X = combined_df.iloc[0: len(train_df)]
    train_Y = train_df[TARGET]
    test_X = combined_df.iloc[len(train_df):]

    logger.info(
        f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
    )

    predictors = list(train_X.columns)
    sk = StratifiedKFold(n_splits=10, shuffle=True)

    common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
    common.update_tracking(RUN_ID, "cv_method", "StratifiedKFold")

    results_dict = model.xgb_train_validate_on_cv(
        logger=logger,
        run_id=RUN_ID,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        metric="rmse",
        num_class=None,
        kf=sk,
        features=predictors,
        params=xgb_params,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
        is_test=IS_TEST,
    )

    train_index = train_df.index

    common.save_artifacts(
        logger,
        is_test=False,
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
