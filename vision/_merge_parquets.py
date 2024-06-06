import pandas as pd
from loguru import logger
from repsim.benchmark.paths import EXPERIMENT_RESULTS_PATH
from tqdm import tqdm


def remove_duplicates(joint_df: pd.DataFrame) -> pd.DataFrame:
    clean_values = []
    grouped_df = joint_df.groupby("id")
    logger.info("Deduplicating dataframes...")
    for group_id, group_df in tqdm(grouped_df):
        if len(group_df) > 1:
            non_nan_vals = group_df[~group_df["metric_value"].isna()]
            if non_nan_vals.empty:
                res_dict = group_df.iloc[0].to_dict()
                res_dict["setting"] = res_dict["id"].split("_")[0]
                clean_values.append(res_dict)
            else:
                res_dict = group_df.iloc[-1].to_dict()
                res_dict["setting"] = res_dict["id"].split("_")[0]
                clean_values.append(res_dict)
        else:
            res_dict = group_df.iloc[0].to_dict()
            res_dict["setting"] = res_dict["id"].split("_")[0]
            clean_values.append(res_dict)
            # clean_values.append(group_df.iloc[0].to_dict())
    logger.info("Creating joint new df.")
    joint_clean_df = pd.DataFrame(clean_values)
    joint_clean_df.set_index("id", inplace=True)
    logger.info("Done.")
    return joint_clean_df


def main():
    parquet_dir = EXPERIMENT_RESULTS_PATH / "parquets_2024_06_05_1031"
    final_parquet_filepath = EXPERIMENT_RESULTS_PATH / "final.parquet"
    parquets_in_dir = list(parquet_dir.iterdir())

    # Only create new one if old does not exist yet.
    if not final_parquet_filepath.exists():
        all_dataframes = []
        for parquet in parquets_in_dir:
            parquet_name = parquet.name
            if not parquet_name.endswith("parquet"):
                continue
            all_dataframes.append(pd.read_parquet(parquet))

        joint_df = pd.concat(all_dataframes)
        joint_df_without_dups = remove_duplicates(joint_df)
        joint_df_without_dups.to_parquet(final_parquet_filepath)


if __name__ == "__main__":

    main()
