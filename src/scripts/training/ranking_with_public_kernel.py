import os
from datetime import datetime

import pandas as pd

import src.config.constants as constants
import src.munging as process_data
import src.common as common
import src.modeling as train_util

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

    # https://www.kaggle.com/xiaoxiaoxiaoxiaoxiao/averaging-top-5-solutions
    # PL: 0.81856
    df_1 = pd.read_csv(
        f"{constants.PUB_SUBMISSION_DIR}/averaging-top-5-solutions/submission.csv"
    )

    # stacking_lgb_xbg_cat_imputer_no_imputer
    # PL: 0.81803
    df_2 = pd.read_csv(
        f"{constants.SUBMISSION_DIR}/sub_stacking_lgb_xbg_cat_imputer_no_imputer_v2_0929_1549_0.81680.gz"
    )

    merged_df = pd.merge(df_1, df_2, how="left", on="id")
    merged_df.columns = ["id", "sub_1", "sub_2"]

    merged_df = merged_df.set_index("id")

    rank_df = train_util.get_rank_mean(merged_df)

    sample_submission_df.claim = rank_df.values

    logger.info(sample_submission_df.head(20))

    file_name = f"sub_{MODEL_NAME}_{RUN_ID}.csv"

    logger.info(f"Saving to submission file {constants.SUBMISSION_DIR}/{file_name}")
    sample_submission_df.to_csv(f"{constants.SUBMISSION_DIR}/{file_name}")

    logger.info(pd.read_csv(f"{constants.SUBMISSION_DIR}/{file_name}"))
