import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import src.config.constants as constants
import src.common as common
import src.munging as process_data
import src.ts as ts_util


def reverse_min_max_scaling(logger, source_df, target_df, features, scaler_dict):
    for name in features:
        mm = scaler_dict[name]
        target_df.loc[:, name] = mm.inverse_transform(source_df[[name]])
    return target_df


def impute_data(logger, df_scaled, scaler_dict, fill_with, features, file_name):
    logger.info(f"Imputing data with {fill_with}")
    df_filled = df_scaled.copy()
    # for k in range(0, len(df_scaled)):
    #     if k not in [3839, 4285]:
    #         logger.info(k)
    #         logger.info(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
    #         df_filled.iloc[k] = fill_with(df_scaled.iloc[k].reset_index(drop=True))
    #     else:
    #         logger.info(df_scaled.iloc[k])
    #         logger.info(df_scaled.iloc[k].reset_index(drop=True))
    #         if k == 4285:
    #             logger.info(fill_with(df_scaled.iloc[k].reset_index(drop=True)))

    for k in range(0, len(df_scaled)):
        df_filled.iloc[k] = fill_with(df_scaled.iloc[k].reset_index(drop=True))

    df_reverted = df_scaled.copy()
    logger.info("Reverting back the scaling")
    df_reverted = reverse_min_max_scaling(
        logger=logger,
        source_df=df_filled,
        target_df=df_reverted,
        features=features,
        scaler_dict=scaler_dict,
    )
    del df_filled
    common.trigger_gc(logger)

    df_reverted.to_parquet(f"{constants.FEATURES_DATA_DIR}/{file_name}", index=True)
    logger.info(
        f"Stored imputed data to features to {constants.FEATURES_DATA_DIR}/{file_name}"
    )
    del df_reverted
    common.trigger_gc(logger)


def main():
    try:
        # Create a Stream only logger
        logger = common.get_logger("generate_features")
        logger.info("Starting to generate features")

        TARGET = "claim"

        train_df, test_df, _ = process_data.read_processed_data(
            logger,
            constants.PROCESSED_DATA_DIR,
            train=True,
            test=True,
            sample_submission=True,
        )

        combined_df = pd.concat([train_df.drop(TARGET, axis=1), test_df])
        features = train_df.drop([TARGET], axis=1).columns

        logger.info("Null description before imputation")
        logger.info(process_data.check_null(combined_df))

        scaler_dict = {}
        combined_df_min_max = combined_df.copy()
        for name in features:
            logger.info(f"Min-Max scaling {name}")
            mm = MinMaxScaler()
            mm.fit(combined_df[[name]])
            combined_df_min_max.loc[:, name] = mm.transform(combined_df[[name]])
            scaler_dict[name] = mm

        impute_data(
            logger=logger,
            df_scaled=combined_df_min_max,
            scaler_dict=scaler_dict,
            fill_with=ts_util.fill_with_gauss,
            features=features,
            file_name="imputed_data_w_gaussian.parquet",
        )

        impute_data(
            logger=logger,
            df_scaled=combined_df_min_max,
            scaler_dict=scaler_dict,
            fill_with=ts_util.fill_with_po3,
            features=features,
            file_name="imputed_data_w_pol_3.parquet",
        )

        impute_data(
            logger=logger,
            df_scaled=combined_df_min_max,
            scaler_dict=scaler_dict,
            fill_with=ts_util.fill_with_lin,
            features=features,
            file_name="imputed_data_w_lin.parquet",
        )

        impute_data(
            logger=logger,
            df_scaled=combined_df_min_max,
            scaler_dict=scaler_dict,
            fill_with=ts_util.fill_with_mix,
            features=features,
            file_name="imputed_data_w_mix.parquet",
        )
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
