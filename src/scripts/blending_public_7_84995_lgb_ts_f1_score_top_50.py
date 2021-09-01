import os
from datetime import datetime

import pandas as pd

import src.config.constants as constants
import src.munging as process_data
import src.common as common

if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    logger = common.get_logger("blend")

    train_df, test_df, sample_submission_df = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    # File with public score 7.84995
    # https://www.kaggle.com/pavfedotov/blending-tool-tps-aug-2021
    df_sub_ext = pd.read_csv(f"{constants.PUB_SUBMISSION_DIR}/0.part")

    # LGB ts f1-weighted SKFold 10 top 50 features
    df_lgb_log_loss_top_10 = pd.read_csv(
        f"{constants.SUBMISSION_DIR}/sub_lgb_ts_f1_weighted_SK_10_top_50_features_0825_1755_0.08395.csv"
    )

    # Giving more importnace to external submission
    sample_submission_df.loss = (
        0.99 * df_sub_ext.loss + 0.01 * df_lgb_log_loss_top_10.loss
    ).values

    file_name = f"sub_{MODEL_NAME}_{RUN_ID}.csv"

    logger.info(f"Saving to submission file {constants.SUBMISSION_DIR}/{file_name}")
    sample_submission_df.to_csv(f"{constants.SUBMISSION_DIR}/{file_name}")

    logger.info(pd.read_csv(f"{constants.SUBMISSION_DIR}/{file_name}"))
