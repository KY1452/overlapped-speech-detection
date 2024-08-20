# """
# This is a boilerplate pipeline 'create_hashing_dataset_libri2mix_detailed'
# generated using Kedro 0.18.14
# """
# """
# This is a boilerplate pipeline 'create_hashing_dataset'
# generated using Kedro 0.18.14
# """
# import ast
# import gc
# import logging
# import math
# import os
# import re
# import sys
# import warnings
# from typing import Any, Dict, List, Tuple

# import librosa
# import numpy as np
# import pandas as pd
# from kedro.config import ConfigLoader
# from pandas import DataFrame

# conf_loader = ConfigLoader("conf")
# parameters = conf_loader.get("parameters.yml")

# warnings.filterwarnings("ignore")


# def libri2mix_gender_labels(database_path: str) -> DataFrame:
#     """
#     This function reads and concatenates metadata files from the libri2mix_metadata_folder and returns the combined DataFrame.
#     """
#     libri2mix_labels: DataFrame = pd.DataFrame()
#     for foldername, subfolder, filenames in os.walk(database_path):
#         for filename in filenames:
#             if "._" in filename or "_info.csv" not in filename:
#                 continue
#             else:
#                 print(filename)
#                 file_path = os.path.join(foldername, filename)
#                 libri2mix_labels = pd.concat([libri2mix_labels, pd.read_csv(file_path)])
#                 libri2mix_labels.reset_index(drop=True, inplace=True)
#     return libri2mix_labels


# def add_audio_amp(
#     audio_id: str,
#     librimix_audio_folder: str,
#     folder_freq: str,
#     folder_audio_mix_length: str,
#     folder_mix_type: str,
#     folder: str,
#     mix_type: str = "Libri2Mix",
# ) -> Tuple[np.ndarray, int]:
#     """
#     A function to add audio amplitude, with parameters audio_id, mix_type, and folder, and returns a tuple of numpy array and integer.
#     """

#     folder: str = folder.replace("_", "-")
#     file: str = audio_id + ".wav"
#     file_path: str = os.path.join(
#         librimix_audio_folder,
#         mix_type,
#         folder_freq,
#         folder_audio_mix_length,
#         folder,
#         folder_mix_type,
#         file,
#     )
#     amp, _ = librosa.load(file_path, sr=None)
#     return amp


# def get_signal_from_time(
#     row: Tuple[List[float], List[float]], audio_freq: int
# ) -> float:
#     """
#     Calculate the average squared amplitude of the selected signal within the specified time ranges.

#     Args:
#     row: A tuple containing a list of time ranges and a list of amplitudes.

#     Returns:
#     The average squared amplitude of the selected signal within the specified time ranges.
#     """
#     selected_amp = []
#     time_list = ast.literal_eval(row[0])  # Convert the string representation to a list
#     for time_range in time_list:
#         start_index = int(time_range[0] * audio_freq)
#         end_index = int(time_range[1] * audio_freq)
#         selected_amp.extend(row[1][start_index:end_index])
#     return np.average(np.array(selected_amp) ** 2)


# def calculate_SNR(s: float, n: float) -> float:
#     """
#     Calculate the Signal-to-Noise Ratio (SNR).

#     Args:
#     s (float): the signal power
#     n (float): the noise power

#     Returns:
#     float: the SNR in decibels
#     """
#     return 10 * math.log10(s / n)


# def get_total_time(time_str: str) -> int:
#     """
#     Calculate the total time difference between two time values.

#     Args:
#     time_str (str): A string representation of a list containing two lists,
#                     each with a single integer representing the time value.

#     Returns:
#     int: The total time difference between the two time values.
#     """
#     try:
#         time = ast.literal_eval(time_str)
#         return time[1][0] - time[0][0]
#     except (ValueError, TypeError, IndexError):
#         # Handle the cases where time_str is not a valid Python expression
#         # or the lists are empty or have incorrect structure
#         return None


# def calculate_duration(overlap_time_ranges: List[Tuple[float, float]]) -> float:
#     """
#     Calculate the total duration for a list of time ranges.

#     Args:
#     - overlap_time_ranges (List[Tuple[float, float]]): A list of tuples representing start and end times of overlapping periods.

#     Returns:
#     - float: The total overlap period.
#     """
#     range_sum = sum(j - i for i, j in overlap_time_ranges)
#     return range_sum


# def convert_to_list(arr: list) -> list:
#     """
#     Convert the input array to a list of lists, with each item in the input array as a separate list.

#     Args:
#         arr (list): The input array to be converted

#     Returns:
#         list: The converted array, with each item as a separate list
#     """
#     converted_arr = []
#     for item in arr:
#         if isinstance(item, np.float64):  # check if item is a numpy float
#             converted_arr.append([item])  # pass as a single-item list
#         else:
#             converted_arr.append(list(item))

#     return converted_arr


# def process_dataset(
#     dataset_type: str,
#     database_path: str,
#     librimix_audio_folder: str,
#     librimix_audio_folder_metadata: str,
#     folder_freq: str,
#     audio_freq: int,
#     folder_audio_mix_length: str,
#     folder_mix_type: str,
#     audio_background: str,
# ) -> pd.DataFrame:
#     """
#     Import the labelled data for the specified dataset type and process it.

#     Args:
#     dataset_type (str): The type of dataset to be processed.

#     Returns:
#     pd.DataFrame: The processed dataset.
#     """

#     # Import the labelled data for the specified dataset type
#     logger = logging.getLogger(__name__)
#     libri2mix_data = pd.DataFrame()
#     for partition_id, partition_load_func in sorted(
#         librimix_audio_folder_metadata.items()
#     ):
#         logger.info(partition_id)
#         if partition_id.startswith(
#             f"{dataset_type}_{audio_background}"
#         ) and partition_id.endswith(".gzip"):
#             data = partition_load_func()
#             libri2mix_data = pd.concat([libri2mix_data, data], ignore_index=True)

#     # Clear memory
#     gc.collect()

#     libri2mix_data["type"] = dataset_type

#     # Calculate amp of wav file using librosa
#     libri2mix_data["amp"] = libri2mix_data["audio_id"].apply(
#         lambda x: add_audio_amp(
#             x,
#             librimix_audio_folder,
#             folder_freq,
#             folder_audio_mix_length,
#             folder_mix_type,
#             folder=dataset_type,
#         )
#     )

#     # Convert the array representation to a list to remove the 'array' in dataframe
#     columns_to_convert = [
#         "source1_ranges_list",
#         "source2_ranges_list",
#         "total_time",
#         "speech_times",
#         "non_speech",
#         "one_spk",
#         "overlaps",
#     ]
#     libri2mix_data[columns_to_convert] = libri2mix_data[columns_to_convert].applymap(
#         convert_to_list
#     )

#     # Then convert the list representation to a string
#     libri2mix_data[columns_to_convert] = libri2mix_data[columns_to_convert].astype(
#         "str"
#     )

#     # Cut the amp into 3 parts by time into 'non_speech', 'one_spk', 'overlaps'
#     for variable in ["non_speech", "one_spk", "overlaps"]:
#         libri2mix_data[variable + "_signal"] = libri2mix_data.apply(
#             lambda row: get_signal_from_time((row[variable], row["amp"]), audio_freq),
#             axis=1,
#         )

#     # Calculate SNR
#     libri2mix_data["one_spk_SNR"] = libri2mix_data.apply(
#         lambda row: calculate_SNR(row["one_spk_signal"], row["non_speech_signal"]),
#         axis=1,
#     )
#     libri2mix_data["two_spk_SNR"] = libri2mix_data.apply(
#         lambda row: calculate_SNR(row["overlaps_signal"], row["non_speech_signal"]),
#         axis=1,
#     )

#     # Merge the dataset with labels on gender info using audio_id and mixture_id
#     libri2mix_labels = libri2mix_gender_labels(database_path)
#     libri2mix_data_detailed = libri2mix_data.merge(
#         libri2mix_labels, left_on="audio_id", right_on="mixture_ID", how="inner"
#     )

#     del libri2mix_data

#     # Add total time for each audio file
#     libri2mix_data_detailed["total_time_period"] = libri2mix_data_detailed[
#         "total_time"
#     ].apply(get_total_time)

#     libri2mix_data_detailed["non_speech"] = libri2mix_data_detailed["non_speech"].apply(
#         ast.literal_eval
#     )
#     libri2mix_data_detailed["one_spk"] = libri2mix_data_detailed["one_spk"].apply(
#         ast.literal_eval
#     )
#     libri2mix_data_detailed["overlaps"] = libri2mix_data_detailed["overlaps"].apply(
#         ast.literal_eval
#     )

#     # Calculate the time periods for each section
#     libri2mix_data_detailed["non_speech_period"] = libri2mix_data_detailed[
#         "non_speech"
#     ].apply(calculate_duration)
#     libri2mix_data_detailed["one_spk_period"] = libri2mix_data_detailed[
#         "one_spk"
#     ].apply(calculate_duration)
#     libri2mix_data_detailed["two_spk_period"] = libri2mix_data_detailed[
#         "overlaps"
#     ].apply(calculate_duration)

#     # # Calculate the percentage of overlap for each section and gender combination
#     # libri2mix_data_detailed["percent_two_spk"] = (
#     #     libri2mix_data_detailed["two_spk_period"]
#     #     / libri2mix_data_detailed["total_time_period"]
#     # )

#     # Combine speaker genders into a single column
#     libri2mix_data_detailed["gender_combi"] = (
#         libri2mix_data_detailed["speaker_1_sex"]
#         + libri2mix_data_detailed["speaker_2_sex"]
#     )

#     # Save the processed data to a CSV file
#     return libri2mix_data_detailed


# def get_libri2mix_clean_detailed_datasets(
#     database_path: str,
#     librimix_audio_folder: str,
#     train360_labels: Dict[str, callable],
#     train100_labels: Dict[str, callable],
#     test_labels: Dict[str, callable],
#     dev_labels: Dict[str, callable],
#     folder_freq: str,
#     audio_freq: int,
#     folder_audio_mix_length: str,
#     folder_mix_type: str,
#     audio_background: str,
#     datasets: List,
#     dev: str,
#     test: str,
#     train_100: str,
#     train_360: str,
# ) -> pd.DataFrame:
#     """
#     A function that generates a detailed dataset and concatenates them into one DataFrame.

#     Args:
#         database_path (str): The path to the database.
#         librimix_audio_folder (str): The folder containing LibriMix audio files.
#         train360_labels: Partitioned dataset of source folder for train360 labels,
#         train100_labels: Partitioned dataset of source folder for train100 labels,
#         test_labels: Partitioned dataset of source folder for test labels,
#         dev_labels: Partitioned dataset of source folder for dev labels,
#         librimix_audio_folder_metadata (str): The metadata folder for LibriMix audio files.
#         folder_freq (str): The folder frequency.
#         audio_freq (int): The audio frequency.
#         folder_audio_mix_length (str): The folder audio mix length.
#         folder_mix_type (str): The mix type of the folders.
#         datasets (List): List of datasets to process.
#         dev (str): Path to the dev dataset.
#         test (str): Path to the test dataset.
#         train_100 (str): Path to the train_100 dataset.
#         train_360 (str): Path to the train_360 dataset.

#     Returns:
#         pd.DataFrame: The concatenated DataFrame of all processed datasets.
#     """
#     logger = logging.getLogger(__name__)
#     if [train360_labels, train100_labels, test_labels, dev_labels]:
#         libri2mix_label_folder = train360_labels
#     for dataset in datasets:
#         logger.info(dataset)
#         detailed_dataset = process_dataset(
#             dataset,
#             database_path,
#             librimix_audio_folder,
#             libri2mix_label_folder,
#             folder_freq,
#             audio_freq,
#             folder_audio_mix_length,
#             folder_mix_type,
#             audio_background,
#         )
#         # to make memory usage more efficient, the respective detailed_dataset is saved as csv into separate datasets
#         detailed_dataset.to_csv(
#             f"data/02_intermediate/libri2mix_clean_overall_detailed_{dataset}.csv"
#         )
#         del detailed_dataset
#         gc.collect()

#     logger.info("finish processing datasets")
#     # each dataset is then read from the respective and concatenated as one df
#     dev = pd.read_csv(dev)
#     test = pd.read_csv(test)
#     train_100 = pd.read_csv(train_100)
#     train_360 = pd.read_csv(train_360)
#     logger.info("concatenating datasets")
#     libri2mix_clean_overall_detailed = pd.concat([dev, test, train_100, train_360])
#     return libri2mix_clean_overall_detailed


# def get_split_dataset_using_hash(
#     libri2mix_overall_clean_detailed: DataFrame,
#     category: str,
#     split_type: str,
#     hash_number: int,
# ) -> DataFrame:
#     """
#     Generate a sub-dataset from the given Libri2mix dataset based on the specified category, split type, and hash number.

#     Args:
#         libri2mix_overall_clean_detailed (DataFrame): The overall clean detailed Libri2mix dataset.
#         category (str): The category of the audio files to include in the sub-dataset.
#         split_type (str): The type of split for the sub-dataset.
#         hash_number (int): The hash number used to select the rows in the sub-dataset.

#     Returns:
#         DataFrame: The sub-dataset containing the selected rows based on the category, split type, and hash number.

#     Raises:
#         ValueError: If libri2mix_overall_clean_detailed is None.

#     Notes:
#         - The function filters the libri2mix_overall_clean_detailed dataset based on the specified category.
#         - It calculates a hash key for each audio file using the audio ID and source1_ranges_list.
#         - The hash key is then used to assign a hash value modulo 10 to the 'hash_%10' column.
#         - Rows with a hash value less than or equal to the specified hash number are selected for the sub-dataset.
#         - The 'split' column is inserted at index 1 in the sub-dataset with the specified split type.
#         - A copy of the sub-dataset is returned.

#     """

#     # Set hash seed and restart interpreter.
#     # This will be done only once if the env var is clear.
#     if not os.environ.get("PYTHONHASHSEED"):
#         os.environ["PYTHONHASHSEED"] = "1234"
#         os.execv(sys.executable, ["python3"] + sys.argv)

#     if libri2mix_overall_clean_detailed is None:
#         # Handle the case when libri2mix_overall_clean_detailed is None
#         # You can raise an exception, return a default value, or perform any other appropriate action
#         raise ValueError("libri2mix_overall_clean_detailed is None")

#     libri2mix_df = libri2mix_overall_clean_detailed[
#         libri2mix_overall_clean_detailed["type"] == category
#     ]
#     # Get hashkey from the audio file id and source1_ranges_list
#     libri2mix_df["hashkey"] = (
#         libri2mix_df["audio_id"] + libri2mix_df["source1_ranges_list"]
#     ).apply(hash)
#     # use the modulo operator to the hashed value
#     libri2mix_df["hash_%10"] = libri2mix_df["hashkey"] % 10
#     # Select ~20k rows as the training dataset and ~6k rows as the evaluation dataset
#     libri2mix_sub_dataset = libri2mix_df[libri2mix_df["hash_%10"] <= hash_number]
#     # Insert the 'train' and 'evaluation' identifier for the dataset
#     libri2mix_sub_dataset.insert(1, "split", split_type)
#     return libri2mix_sub_dataset.copy()


# def generate_modelling_dataset(
#     libri2mix_overall_clean_detailed: DataFrame,
# ) -> DataFrame:
#     """
#     Generate and return the modelling dataset.

#     Args:
#         libri2mix_overall_clean_detailed (DataFrame): The input DataFrame containing clean detailed data.

#     Returns:
#         DataFrame: The generated modelling dataset.
#     """
#     training_dataset = get_split_dataset_using_hash(
#         libri2mix_overall_clean_detailed, "train_360", "training", 3
#     )
#     evaluation_dataset = get_split_dataset_using_hash(
#         libri2mix_overall_clean_detailed, "train_100", "evaluation", 4
#     )

#     testing_dataset = libri2mix_overall_clean_detailed[
#         (libri2mix_overall_clean_detailed["type"] == "dev")
#         | (libri2mix_overall_clean_detailed["type"] == "test")
#     ].copy()
#     testing_dataset.insert(1, "split", "test")

#     libri2mix_modeling_dataset = pd.concat(
#         [
#             training_dataset.drop(columns=["hashkey", "hash_%10"]),
#             evaluation_dataset.drop(columns=["hashkey", "hash_%10"]),
#             testing_dataset,
#         ],
#         ignore_index=True,
#     )

#     return libri2mix_modeling_dataset

"""
This is a boilerplate pipeline 'create_hashing_dataset_libri2mix_detailed'
generated using Kedro 0.18.14
"""
import ast
import gc
import logging
import math
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from kedro.config import ConfigLoader
from pandas import DataFrame

conf_loader = ConfigLoader("conf")
parameters = conf_loader.get("parameters.yml")

warnings.filterwarnings("ignore")


def libri2mix_gender_labels(database_path: str) -> DataFrame:
    libri2mix_labels: DataFrame = pd.DataFrame()
    for foldername, subfolder, filenames in os.walk(database_path):
        for filename in filenames:
            if "._" in filename or "_info.csv" not in filename:
                continue
            else:
                print(filename)
                file_path = os.path.join(foldername, filename)
                libri2mix_labels = pd.concat([libri2mix_labels, pd.read_csv(file_path)])
                libri2mix_labels.reset_index(drop=True, inplace=True)
    return libri2mix_labels


def add_audio_amp(
    audio_id: str,
    librimix_audio_folder: str,
    folder_freq: str,
    folder_audio_mix_length: str,
    folder_mix_type: str,
    folder: str,
    mix_type: str = "Libri2Mix",
) -> Tuple[np.ndarray, int]:
    folder: str = folder.replace("_", "-")
    file: str = audio_id + ".wav"
    file_path: str = os.path.join(
        librimix_audio_folder,
        mix_type,
        folder_freq,
        folder_audio_mix_length,
        folder,
        folder_mix_type,
        file,
    )
    amp, _ = librosa.load(file_path, sr=None)
    return amp


def get_signal_from_time(
    row: Tuple[List[float], List[float]], audio_freq: int
) -> float:
    selected_amp = []
    time_list = ast.literal_eval(row[0])
    for time_range in time_list:
        start_index = int(time_range[0] * audio_freq)
        end_index = int(time_range[1] * audio_freq)
        selected_amp.extend(row[1][start_index:end_index])
    return np.average(np.array(selected_amp) ** 2)


def calculate_SNR(s: float, n: float) -> float:
    return 10 * math.log10(s / n)


def get_total_time(time_str: str) -> int:
    try:
        time = ast.literal_eval(time_str)
        return time[1][0] - time[0][0]
    except (ValueError, TypeError, IndexError):
        return None


def calculate_duration(overlap_time_ranges: List[Tuple[float, float]]) -> float:
    range_sum = sum(j - i for i, j in overlap_time_ranges)
    return range_sum


def convert_to_list(arr: list) -> list:
    converted_arr = []
    for item in arr:
        if isinstance(item, np.float64):
            converted_arr.append([item])
        else:
            converted_arr.append(list(item))
    return converted_arr


def process_chunk(
    chunk: pd.DataFrame,
    database_path: str,
    librimix_audio_folder: str,
    folder_freq: str,
    audio_freq: int,
    folder_audio_mix_length: str,
    folder_mix_type: str,
    audio_background: str,
) -> pd.DataFrame:
    chunk["type"] = chunk["dataset_type"]

    chunk["amp"] = chunk["audio_id"].apply(
        lambda x: add_audio_amp(
            x,
            librimix_audio_folder,
            folder_freq,
            folder_audio_mix_length,
            folder_mix_type,
            folder=chunk["dataset_type"].iloc[0],
        )
    )

    columns_to_convert = [
        "source1_ranges_list",
        "source2_ranges_list",
        "total_time",
        "speech_times",
        "non_speech",
        "one_spk",
        "overlaps",
    ]
    chunk[columns_to_convert] = chunk[columns_to_convert].applymap(convert_to_list)
    chunk[columns_to_convert] = chunk[columns_to_convert].astype("str")

    for variable in ["non_speech", "one_spk", "overlaps"]:
        chunk[variable + "_signal"] = chunk.apply(
            lambda row: get_signal_from_time((row[variable], row["amp"]), audio_freq),
            axis=1,
        )

    chunk["one_spk_SNR"] = chunk.apply(
        lambda row: calculate_SNR(row["one_spk_signal"], row["non_speech_signal"]),
        axis=1,
    )
    chunk["two_spk_SNR"] = chunk.apply(
        lambda row: calculate_SNR(row["overlaps_signal"], row["non_speech_signal"]),
        axis=1,
    )

    libri2mix_labels = libri2mix_gender_labels(database_path)
    chunk = chunk.merge(
        libri2mix_labels, left_on="audio_id", right_on="mixture_ID", how="inner"
    )

    chunk["total_time_period"] = chunk["total_time"].apply(get_total_time)
    chunk["non_speech"] = chunk["non_speech"].apply(ast.literal_eval)
    chunk["one_spk"] = chunk["one_spk"].apply(ast.literal_eval)
    chunk["overlaps"] = chunk["overlaps"].apply(ast.literal_eval)

    chunk["non_speech_period"] = chunk["non_speech"].apply(calculate_duration)
    chunk["one_spk_period"] = chunk["one_spk"].apply(calculate_duration)
    chunk["two_spk_period"] = chunk["overlaps"].apply(calculate_duration)

    chunk["gender_combi"] = chunk["speaker_1_sex"] + chunk["speaker_2_sex"]

    return chunk


def process_dataset_in_chunks(
    dataset_type: str,
    database_path: str,
    librimix_audio_folder: str,
    librimix_audio_folder_metadata: str,
    folder_freq: str,
    audio_freq: int,
    folder_audio_mix_length: str,
    folder_mix_type: str,
    audio_background: str,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    libri2mix_data = pd.DataFrame()
    for partition_id, partition_load_func in sorted(
        librimix_audio_folder_metadata.items()
    ):
        logger.info(partition_id)
        if partition_id.startswith(
            f"{dataset_type}_{audio_background}"
        ) and partition_id.endswith(".gzip"):
            data = partition_load_func()
            data["dataset_type"] = dataset_type
            for start in range(0, data.shape[0], chunk_size):
                chunk = data.iloc[start : start + chunk_size]
                processed_chunk = process_chunk(
                    chunk,
                    database_path,
                    librimix_audio_folder,
                    folder_freq,
                    audio_freq,
                    folder_audio_mix_length,
                    folder_mix_type,
                    audio_background,
                )
                libri2mix_data = pd.concat(
                    [libri2mix_data, processed_chunk], ignore_index=True
                )
                gc.collect()

    return libri2mix_data

def get_libri2mix_clean_detailed_datasets(
    database_path: str,
    librimix_audio_folder: str,
    train360_labels: Dict[str, callable],
    train100_labels: Dict[str, callable],
    test_labels: Dict[str, callable],
    dev_labels: Dict[str, callable],
    folder_freq: str,
    audio_freq: int,
    folder_audio_mix_length: str,
    folder_mix_type: str,
    audio_background: str,
    datasets: List,
    dev: str,
    test: str,
    train_100: str,
    train_360: str,
    part_number: int,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    libri2mix_clean_overall_detailed = pd.DataFrame()
    for dataset in datasets:
        logger.info(dataset)
        if dataset == "train_360":
            labels_dict = train360_labels
        elif dataset == "train_100":
            labels_dict = train100_labels
        elif dataset == "test":
            labels_dict = test_labels
        elif dataset == "dev":
            labels_dict = dev_labels
        else:
            continue

        # Process only the specified part_number
        label_keys = [key for key in labels_dict.keys() if f"_part{part_number}_" in key]
        if not label_keys:
            continue
        partial_labels = {k: labels_dict[k] for k in label_keys}

        detailed_dataset = process_dataset_in_chunks(
            dataset,
            database_path,
            librimix_audio_folder,
            partial_labels,
            folder_freq,
            audio_freq,
            folder_audio_mix_length,
            folder_mix_type,
            audio_background,
        )
        detailed_dataset.to_csv(
            f"data/02_intermediate/libri2mix_clean_overall_detailed_{dataset}_part{part_number}.csv",
            index=False
        )
        libri2mix_clean_overall_detailed = pd.concat(
            [libri2mix_clean_overall_detailed, detailed_dataset], ignore_index=True
        )
        del detailed_dataset
        gc.collect()

    logger.info("finish processing datasets")
    return libri2mix_clean_overall_detailed

def get_split_dataset_using_hash(
    libri2mix_overall_clean_detailed: DataFrame,
    category: str,
    split_type: str,
    hash_number: int,
) -> DataFrame:
    if not os.environ.get("PYTHONHASHSEED"):
        os.environ["PYTHONHASHSEED"] = "1234"
        os.execv(sys.executable, ["python3"] + sys.argv)

    if libri2mix_overall_clean_detailed is None:
        raise ValueError("libri2mix_overall_clean_detailed is None")

    libri2mix_df = libri2mix_overall_clean_detailed[
        libri2mix_overall_clean_detailed["type"] == category
    ]
    libri2mix_df["hashkey"] = (
        libri2mix_df["audio_id"] + libri2mix_df["source1_ranges_list"]
    ).apply(hash)
    libri2mix_df["hash_%10"] = libri2mix_df["hashkey"] % 10
    libri2mix_sub_dataset = libri2mix_df[libri2mix_df["hash_%10"] <= hash_number]
    libri2mix_sub_dataset.insert(1, "split", split_type)
    return libri2mix_sub_dataset.copy()


def generate_modelling_dataset(
    libri2mix_overall_clean_detailed: DataFrame,
) -> DataFrame:
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
