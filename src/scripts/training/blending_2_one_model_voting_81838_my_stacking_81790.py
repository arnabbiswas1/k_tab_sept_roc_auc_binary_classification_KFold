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

    # https://www.kaggle.com/martynovandrey/one-model-voting-from-0-81800-to-0-81837/notebook
    # PL : 0.81838
    df_1 = pd.read_csv(
        f"{constants.PUB_SUBMISSION_DIR}/one_model_voting/submission.csv"
    )

    # stacking_lgb_xbg_cat_imputer_no_imputer
    df_2 = pd.read_csv(
        f"{constants.SUBMISSION_DIR}/sub_stacking_lgb_xbg_cat_imputer_no_imputer_0923_1747_0.81670.gz"
    )

    # Giving more importnace to external submission
    sample_submission_df.claim = (
        0.5 * df_1.claim + 0.5 * df_2.claim
    ).values

    file_name = f"sub_{MODEL_NAME}_{RUN_ID}.csv"

    logger.info(f"Saving to submission file {constants.SUBMISSION_DIR}/{file_name}")
    sample_submission_df.to_csv(f"{constants.SUBMISSION_DIR}/{file_name}")

    logger.info(pd.read_csv(f"{constants.SUBMISSION_DIR}/{file_name}"))