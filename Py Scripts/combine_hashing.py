import pandas as pd
import os
import sys

def get_split_dataset_using_hash(
    libri2mix_overall_clean_detailed: pd.DataFrame,
    category: str,
    split_type: str,
    hash_number: int,
) -> pd.DataFrame:
    if not os.environ.get("PYTHONHASHSEED"):
        os.environ["PYTHONHASHSEED"] = "1234"
        os.execv(sys.executable, ["python3"] + sys.argv)

    if libri2mix_overall_clean_detailed is None:
        raise ValueError("libri2mix_overall_clean_detailed is None")

    libri2mix_df = libri2mix_overall_clean_detailed[
        libri2mix_overall_clean_detailed["type"] == category
    ].copy()  # Ensure we're working on a copy

    libri2mix_df["hashkey"] = (
        libri2mix_df["audio_id"] + libri2mix_df["source1_ranges_list"]
    ).apply(hash)
    libri2mix_df["hash_%10"] = libri2mix_df["hashkey"] % 10
    libri2mix_sub_dataset = libri2mix_df[libri2mix_df["hash_%10"] <= hash_number]
    libri2mix_sub_dataset.insert(1, "split", split_type)
    return libri2mix_sub_dataset.copy()

def generate_modelling_dataset(libri2mix_overall_clean_detailed: pd.DataFrame) -> pd.DataFrame:
    training_dataset = get_split_dataset_using_hash(
        libri2mix_overall_clean_detailed, "train_360", "training", 3
    )
    evaluation_dataset = get_split_dataset_using_hash(
        libri2mix_overall_clean_detailed, "train_100", "evaluation", 4
    )

    testing_dataset = libri2mix_overall_clean_detailed[
        (libri2mix_overall_clean_detailed["type"] == "dev")
        | (libri2mix_overall_clean_detailed["type"] == "test")
    ].copy()
    testing_dataset.insert(1, "split", "test")

    libri2mix_modeling_dataset = pd.concat(
        [
            training_dataset.drop(columns=["hashkey", "hash_%10"]),
            evaluation_dataset.drop(columns=["hashkey", "hash_%10"]),
            testing_dataset,
        ],
        ignore_index=True,
    )

    return libri2mix_modeling_dataset

def main():
    # Paths to the concatenated CSV files
    dev_path = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_dev.csv"
    test_path = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_test.csv"
    train_100_path = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_100.csv"
    train_360_path = "/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/libri2mix_clean_overall_detailed_train_360.csv"


    # Read the concatenated CSV files
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    train_100_df = pd.read_csv(train_100_path)
    train_360_df = pd.read_csv(train_360_path)

    # Combine them into a single DataFrame
    libri2mix_overall_clean_detailed = pd.concat([dev_df, test_df, train_100_df, train_360_df], ignore_index=True)

    # Save the combined DataFrame
    libri2mix_overall_clean_detailed.to_csv("/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/hash_libri2_detailed/libri2mix_clean_detailed_datasets_final.csv", index=False)

    # Generate the modeling dataset
    libri2mix_modeling_dataset = generate_modelling_dataset(libri2mix_overall_clean_detailed)

    # Save the modeling dataset
    libri2mix_modeling_dataset.to_csv("/home/users/ntu/scsekyad/scratch/OSD/final deployment package/klass-osd-kedro-pipeline/klass-osd/data/02_intermediate/hash_libri2_detailed/libri2mix_modeling_dataset_final.csv", index=False)

if __name__ == "__main__":
    main()
