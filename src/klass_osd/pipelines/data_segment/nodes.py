"""
This is a boilerplate pipeline 'data_segment'
generated using Kedro 0.18.14
"""

import logging
import os
import sys
from typing import List, Tuple, Union

import pandas as pd

sys.path.append(os.getcwd())

from src.klass_osd.utils.data_prep import add_audio_amp_librimix, remove_filtered_cols

# #### Rename columns, read parquet or csv files


def renamecol(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the 'overlaps' column in the DataFrame to 'two_spk'.

    Parameters:
    - dataset (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the 'overlaps' column renamed to 'two_spk'.
    """
    dataset.rename(columns={"overlaps": "two_spk"}, inplace=True)
    return dataset


# #### Preprocessing


def get_signal_from_time(
    row: Tuple[Union[List[Tuple[float, float]], None], List[float]], sample_rate: int
) -> List[List[float]]:
    """
    Extract audio signal from the specified time ranges.

    Parameters:
    - row (Tuple[Union[List[Tuple[float, float]], None], List[float]]):
    A tuple containing the time list and amplitude list.
    - sample_rate (int): Sampling rate.

    Returns:
    - List[List[float]]: A list containing selected amplitudes based on the
    provided time ranges.
    """

    selected_amp = []

    try:
        time_list = row[0]

        # Skip processing if the time_list is empty
        if len(time_list) == 0:
            selected_amp.append([])

        else:
            for time_range in time_list:
                start_index = int(time_range[0] * sample_rate)
                end_index = int(time_range[1] * sample_rate)
                selected_amp.append(row[1][start_index:end_index])

    except IndexError as index_error:
        logging.error("IndexError processing row %s: %s", row.name, index_error)
        logging.error("Row content: %s", row)

    except ValueError as value_error:
        logging.error("ValueError processing row %s: %s", row.name, value_error)
        logging.error("Row content: %s", row)

    except Exception as exception:
        logging.error("Error processing row %s: %s", row.name, exception)
        logging.error("Row content: %s", row)

    return selected_amp


def generate_file_name(
    split_type: str, mix_type: str, part_type: str, num_speaker_mix: str
) -> tuple:
    """
    Generate a specific file name based on the provided parameters.

    Args:
        split_type (str): The type of split, which can optionally contain a hyphen.
        mix_type (str): The type of mix.
        part_type (str): The part type, which can be None if not applicable.
        num_speaker_mix (str): The number of speaker mix, used to extract the last
        4 characters.

    Returns:
        tuple: A tuple containing the specific file name and the split mix type.
               The specific file name is constructed based on the input parameters and
               includes information about the split type, mix type, part type, and
               the number of speaker mix. The split mix type is a combination of split
               type and mix type, separated by an underscore or adjusted based on the
               presence of a hyphen in the split type.

    Example:
        >>> generate_file_name('train-360', 'clean', 'part3', 'Libri2Mix')
        ('train_360_clean_part_2mix_osd_labels.parquet.gzip', 'train_mix_clean')
    """
    if "-" in split_type:
        parts = split_type.split("-")
        split_mix_type = f"{parts[0]}_{parts[1]}_{mix_type}"
    else:
        split_mix_type = f"{split_type}_{mix_type}"

    mix_label = num_speaker_mix[
        -4:
    ].lower()  # Extracting last 4 characters and converting to lowercase
    if part_type is not None:
        specific_file_name = (
            f"{split_mix_type}_{part_type}_{mix_label}_osd_labels.parquet.gzip"
        )
    else:
        specific_file_name = f"{split_mix_type}_{mix_label}_osd_labels.parquet.gzip"

    return specific_file_name


def get_signals_only_librispeech_mix_in_parts(
    wave_type: str,
    mix_type: str,
    sample_rate: int,
    split_type: str,
    num_speaker_mix: str,
    hash_audioid_mix: pd.DataFrame,
    librimix_audio_folder: str,
    metadata_directory_pathway: str,
    part_type: str = None,
) -> pd.DataFrame:
    """
    Get annotated chunks from LibriSpeech metadata with two speakers in parts.

    Parameters:
    - wave_type (str): Wave file type (e.g., 'wav8k', 'wav16k').
    - mix_type (str): Mixture type ('both', 'clean').
    - sample_rate (int): Sampling rate.
    - split_type (str): Split type (e.g., 'dev', 'train-100', 'test').
    - num_speaker_mix (str): Number of speakers in the mix ('Libri3Mix').
    - librimix_audio_folder: '/pvc-data/open-source-data/librimix_storage_dir'.
    - part_type (str, optional): Part type for handling metadata.

    Returns:
    - pd.DataFrame: DataFrame with annotated segments.

    """

    specific_file_name = generate_file_name(
        split_type, mix_type, part_type, num_speaker_mix
    )

    metadata_pathway = os.path.join(
        metadata_directory_pathway,
        num_speaker_mix,
        wave_type,
        "max",
        "metadata",
        specific_file_name,
    )

    dataset = pd.read_parquet(metadata_pathway)
    dataset = renamecol(dataset)

    # Read, drop duplicates, filter for "training" (only training dataset gets
    # segmenting treatment)
    audio_id_df = hash_audioid_mix
    audio_id_df = audio_id_df[audio_id_df["split"] == "training"]

    dataset = dataset[dataset["audio_id"].isin(audio_id_df["audio_id"])]
    dataset.reset_index(drop=True, inplace=True)

    # if number of speaker = 3, instantiate list of name of columns
    if num_speaker_mix == "Libri3Mix":
        column_list = [
            "source1_ranges_list",
            "source2_ranges_list",
            "source3_ranges_list",
            "non_speech",
            "one_spk",
            "two_spk",
            "two_or_three_spk",
            "three_spk",
        ]

        remove_col_list = [
            "source1_ranges_list",
            "source2_ranges_list",
            "source3_ranges_list",
            "non_speech",
            "one_spk",
            "two_spk",
            "two_or_three_spk",
            "three_spk",
            "total_time",
            "speech_times",
            "source1_ranges_list_signal",
            "source2_ranges_list_signal",
            "source3_ranges_list_signal",
        ]

    elif num_speaker_mix == "Libri2Mix":
        column_list = [
            "source1_ranges_list",
            "source2_ranges_list",
            "non_speech",
            "one_spk",
            "two_spk",
        ]
        remove_col_list = [
            "source1_ranges_list",
            "source2_ranges_list",
            "non_speech",
            "one_spk",
            "two_spk",
            "total_time",
            "speech_times",
            "source1_ranges_list_signal",
            "source2_ranges_list_signal",
        ]

    else:
        logger.error(  # noqa F821 logger is defined but in parent function
            "num_speaker_mix is not Libri2Mix or Libri3Mix"
        )
        raise ValueError("num_speaker_mix is not Libri2Mix or Libri3Mix")

    dataset = remove_filtered_cols(
        column_list,
        dataset=dataset,
    )

    dataset["amp"] = dataset["audio_id"].apply(
        add_audio_amp_librimix,
        wave_type=wave_type,
        split_type=split_type,
        mix_type=mix_type,
        num_speaker_mix=num_speaker_mix,
        librimix_audio_folder=librimix_audio_folder,
    )

    for variables in column_list:
        dataset[variables + "_signal"] = dataset.apply(
            lambda row: get_signal_from_time(row[[variables, "amp"]], sample_rate),
            axis=1,
        )

    remove_col_list.append("amp")
    dataset.drop(columns=remove_col_list, inplace=True)

    return dataset


def save_signals_only_librimix_in_parts(
    wave_type_list: List[str],
    mixture_type_list: List[str],
    split_type_list: List[str],
    num_speaker_mix: str,
    hash_audioid_mix: pd.DataFrame,
    part_type_list: List[str],
    librimix_audio_folder: str,
    metadata_directory_pathway: str,
    batchsize: int,
) -> pd.DataFrame:
    """
    Save annotated chunks from LibriMix metadata in parts.

    Parameters:
    - wave_type_list (List[str]): List of wave file types (e.g., ['wav8k', 'wav16k']).
    - mixture_type_list (List[str]): List of mixture types (e.g., ['both', 'clean']).
    - split_type_list (List[str]): List of split types (e.g., ['dev', 'train-100', 'test']).
    - num_speaker_mix (str): Number of speakers in the mix ('Libri3Mix' or 'Libri2Mix).
    - hash_audioid_mix (list): partitioned data.
    - part_type_list (List[str]): List of part types (e.g., ['train-360', 'dev', 'test']).
    - librimix_audio_folder (str): pathway to directory.
    - batchsize (int): 1000

    Returns:
    - row_df (pd.DataFrame)

    """

    # Log messages to the console. You can also configure it to log to a file.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    partitions = {}

    for wave_types in wave_type_list:
        if wave_types == "wav16k":
            sampling_rate = 16000
        elif wave_types == "wav8k":
            sampling_rate = 8000
        else:
            # Handle unexpected input
            logging.error("Invalid wave format: %s", wave_types)
            continue  # skip to next iteration if input is invalid

        for split_type in split_type_list:
            for mixture_types in mixture_type_list:
                all_data = pd.DataFrame()  # Initialize an empty DataFrame
                for part_type in part_type_list:
                    dataset = get_signals_only_librispeech_mix_in_parts(
                        wave_types,
                        mixture_types,
                        sampling_rate,
                        split_type,
                        num_speaker_mix,
                        hash_audioid_mix,
                        librimix_audio_folder,
                        metadata_directory_pathway,
                        part_type,
                    )

                    dataset.reset_index(drop=True, inplace=True)

                    all_data = pd.concat([all_data, dataset])

                # Save DataFrame in batches
                batch_size = batchsize  # 100, 800 or 1000
                num_batches = len(all_data) // batch_size + 1

                for batch_num in range(num_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(all_data))

                    batch_df = all_data.iloc[start_idx:end_idx]

                    batch_df.reset_index(drop=True, inplace=True)

                    # Append part number to the file name
                    part_number = batch_num + 1

                    logger.info("%s length: %s", part_number, len(batch_df))

                    # Construct a unique key for each partition
                    partition_key = f"{num_speaker_mix}_{wave_types}_{split_type}_{mixture_types}_part{part_number}.parquet"
                    partitions[partition_key] = batch_df

                return partitions
