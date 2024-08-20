"""
This is a boilerplate pipeline 'data_chunks'
generated using Kedro 0.18.14
"""

### Create pipeline for chunking, and  metadata

import logging
import os
import sys
from typing import Dict, List, Union

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.klass_osd.utils.data_prep import add_audio_amp_librimix, remove_filtered_cols

##### Preprocessing


# Function to filter out dictionaries with the same start and end timings
def filter_duplicate_timings(
    lst: List[Dict[str, List[float]]]
) -> List[Dict[str, List[float]]]:
    """
    Filter a list of items to remove duplicates based on timing information.

    Parameters:
    - lst (List[Dict[str, List[float]]]): A list of dictionaries,
    where each dictionary has a "timing" key associated with a list of
    two floating-point numbers representing start and end timings.

    Returns:
    - filtered_list (List[Dict[str, List[float]]]): A filtered list of unique items
    where the start and end timings are different.
    """
    unique_timings = set()
    filtered_list = []

    for item in lst:
        timings = item["timing"]

        # Check if start and end timings are not the same
        if timings[0] != timings[1] and timings not in unique_timings:
            unique_timings.add(timings)
            filtered_list.append(item)

    return filtered_list


# Function to determine the labels for a given time range (for 2 spk)
def get_labels(
    start_time: float,
    end_time: float,
    non_speech_ranges: List[List[float]],
    one_spk_ranges: List[List[float]],
    two_spk_ranges: List[List[float]],
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Determine the labels for a given time range.

    Parameters:
    - start_time (float): The start time of the time range.
    - end_time (float): The end time of the time range.
    - non_speech_ranges (List[List[float]]): List of non-speech time ranges.
    - one_spk_ranges (List[List[float]]): List of one speaker time ranges.
    - two_spk_ranges (List[List[float]]): List of two speaker speech time ranges.

    Returns:
    - labels (List[Dict]): List of dictionaries representing labels and timings
    within the given time range.
    Example output:
    [
    {"label": "one_spk", "timing": [0, 0.4]},
    {"label": "non_speech", "timing": [0.4, 1.5]},
    {"label": "two_spk", "timing": [1.5, 2.0]}
    ]

    """
    labels = []

    labels.extend(
        get_label_timings(start_time, end_time, non_speech_ranges, "non_speech")
    )
    labels.extend(get_label_timings(start_time, end_time, one_spk_ranges, "one_spk"))
    labels.extend(get_label_timings(start_time, end_time, two_spk_ranges, "two_spk"))

    # If no labels were added, it means it's an 'other' type.
    if not labels:
        labels.append({"label": "other", "timing": [start_time, end_time]})

    return labels


def produce_chunk_timing_and_labels(
    dataset: pd.DataFrame, chunk_seconds: float
) -> pd.DataFrame:
    """
    Generate chunk timings and labels for each row in the DataFrame.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame containing columns 'total_time',
    'non_speech', 'one_spk', and 'overlaps'.
    Each list within these columns consists of sublists that specify the starting
    and ending times (in seconds) for each occurrence of the
    respective phenomena. For example, a value in the 'non_speech' column might
    look like [[0.11, 0.13], [2.32, 2.66], [3.05, 3.09]...],
    indicating several intervals where non-speech audio is detected.
    - chunk_seconds (float): duration of chunks in seconds (e.g. 0.6s)
    Returns:
    - dataset (pd.DataFrame): Updated DataFrame with a new column
    'labels_and_timings_of_chunks' containing labels and timings for each chunk.
    Example output in column:
    [[{'label': 'non_speech', 'timing': array([0.  , 0.27])},
              {'label': 'one_spk', 'timing': array([0.27, 0.28])},
              {'label': 'two_spk', 'timing': array([0.28, 0.44])},
              {'label': 'non_speech', 'timing': array([0.44, 0.6 ])}]...
    """

    # Create a new column 'chunk_labels' in the original DataFrame
    dataset["labels_and_timings_of_chunks"] = None

    # Iterate through each row in the original DataFrame
    for index, row in dataset.iterrows():
        total_time = row["total_time"]
        non_speech_ranges = row["non_speech"]
        one_spk_ranges = row["one_spk"]
        two_spk_ranges = row["two_spk"]

        # Generate time points for each time (in seconds) interval within
        # 'total_time' range
        time_points = np.arange(total_time[0], total_time[1], chunk_seconds)

        # Determine labels for each chunk
        chunk_labels = [
            get_labels(
                start,
                start + chunk_seconds,
                non_speech_ranges,
                one_spk_ranges,
                two_spk_ranges,
            )
            for start in time_points
        ]

        # Apply the filtering function to each inner list
        filtered_chunks = [
            filter_duplicate_timings(inner_list) for inner_list in chunk_labels
        ]
        # Drop last chunk. This step counters potential discrepancies in length
        # due to floating-point precision errors.
        filtered_chunks = filtered_chunks[:-1]

        # Assign the list of dictionaries to the 'chunk_labels' column
        dataset.at[index, "labels_and_timings_of_chunks"] = filtered_chunks

    return dataset


## Helper function for get_labels and get_labels_3spk


def get_label_timings(
    start_time: float, end_time: float, time_ranges: List[List[float]], label_name: str
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Helper function to generate labels, given start and end time.

    Parameters:
    - start_time (float): The start time of the time range.
    - end_time (float): The end time of the time range.
    - time_ranges (List[List[float]]): List of time ranges for the specific label.
    - label_name (str): The label name.

    Returns:
    - generated_labels (List[Dict[str, Union[str, List[float]]]]): List of
    dictionaries with label and timings.
    """

    generated_labels = []

    if any(
        start_time <= range_end and end_time >= range_start
        for range_start, range_end in time_ranges
    ):
        valid_ranges = [
            (max(start_time, range_start), min(end_time, range_end))
            for range_start, range_end in time_ranges
            if start_time <= range_end and end_time >= range_start
        ]
        generated_labels.extend(
            [
                {"label": label_name, "timing": valid_range}
                for valid_range in valid_ranges
            ]
        )

    return generated_labels


# Function to determine the labels for a given time range for 3 speaker
def get_labels_3spk(
    start_time: float,
    end_time: float,
    non_speech_ranges: List[List[float]],
    one_spk_ranges: List[List[float]],
    two_spk_ranges: List[List[float]],
    three_spk_ranges: List[List[float]],
) -> List[Dict[str, Union[str, List[float]]]]:
    """
    Determine the labels for a given time range (for 3 spk).

    Parameters:
    - start_time (float): The start time of the time range.
    - end_time (float): The end time of the time range.
    - non_speech_ranges (List[List[float]]): List of non-speech time ranges.
    - one_spk_ranges (List[List[float]]): List of one speaker time ranges.
    - two_spk_ranges (List[List[float]]): List of two speaker speech time ranges.
    - three_spk_ranges (List[List[float]]): List of three speaker speech time ranges.

    Returns:
    - labels(List[Dict]): List of dictionaries representing labels and
    timings within the given time range.
    Example output:
    [
    {"label": "one_spk", "timing": [0, 0.4]},
    {"label": "non_speech", "timing": [0.4, 1.5]},
    {"label": "two_spk", "timing": [1.5, 2.0]},
    {"label": "three_spk", "timing": [2.0, 2.4]}...
    ]
    """
    labels = []

    labels.extend(
        get_label_timings(start_time, end_time, non_speech_ranges, "non_speech")
    )
    labels.extend(get_label_timings(start_time, end_time, one_spk_ranges, "one_spk"))
    labels.extend(get_label_timings(start_time, end_time, two_spk_ranges, "two_spk"))
    labels.extend(
        get_label_timings(start_time, end_time, three_spk_ranges, "three_spk")
    )

    # If no labels were added, it means it's an 'other' type.
    if not labels:
        labels.append({"label": "other", "timing": [start_time, end_time]})

    return labels


def produce_chunk_timing_and_labels_3spk(
    dataset: pd.DataFrame, chunk_seconds: float
) -> pd.DataFrame:
    """
    Generate chunk timings and labels for each row in the DataFrame.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame containing columns 'total_time',
     'non_speech', 'one_spk', 'two_spk' and 'three_spk'.
    Each list within these columns consists of sublists that specify the starting
    and ending times (in seconds) for each occurrence of the
    respective phenomena. For example, a value in the 'non_speech' column might
    look like [[0.11, 0.13], [2.32, 2.66], [3.05, 3.09]...],
    indicating several intervals where non-speech audio is detected.
    - chunk_seconds (float): duration of chunks in seconds (e.g. 0.6s)


    Returns:
    - dataset (pd.DataFrame): Updated DataFrame with a new column
    'labels_and_timings_of_chunks' containing labels and timings for each stated chunk.
    Example output in column:
    [[{'label': 'non_speech', 'timing': array([0.  , 0.27])},
              {'label': 'one_spk', 'timing': array([0.27, 0.28])},
              {'label': 'two_spk', 'timing': array([0.28, 0.44])},
              {'label': 'three_spk', 'timing': array([0.44, 0.6 ])}]...
    """

    # Create a new column 'chunk_labels' in the original DataFrame
    dataset["labels_and_timings_of_chunks"] = None

    # Iterate through each row in the original DataFrame
    for index, row in dataset.iterrows():
        total_time = row["total_time"]
        non_speech_ranges = row["non_speech"]
        one_spk_ranges = row["one_spk"]
        two_spk_ranges = row["two_spk"]
        three_spk_ranges = row["three_spk"]

        # Generate time points for each time interval within 'total_time' range
        time_points = np.arange(total_time[0], total_time[1], chunk_seconds)

        # Determine labels for each chunk seconds
        chunk_labels = [
            get_labels_3spk(
                start,
                start + chunk_seconds,
                non_speech_ranges,
                one_spk_ranges,
                two_spk_ranges,
                three_spk_ranges,
            )
            for start in time_points
        ]

        # Apply the filtering function to each inner list
        filtered_chunks = [
            filter_duplicate_timings(inner_list) for inner_list in chunk_labels
        ]
        # Drop last chunk. This step counters potential discrepancies in
        # length due to floating-point precision errors.
        filtered_chunks = filtered_chunks[:-1]

        # Assign the list of dictionaries to the 'chunk_labels' column
        dataset.at[index, "labels_and_timings_of_chunks"] = filtered_chunks

    return dataset


def produce_chunk_label(
    list_of_labels_timings_in_chunks: List[Dict],
) -> List[List]:
    """
    This function identifies the label(s) within each chunk of signals that
    have the maximum cumulative duration,
    effectively determining the majority class for each chunk based on timing.

    Parameters:
    - list_of_labels_timings_in_chunks (List[Dict]): A list of sublists,
    where each sublist contains dictionaries
    with 'label' and 'timing' keys. Each sublist corresponds to a chunk. For example:
    list_of_labels_timings_in_chunks =
    [[{'label': 'non_speech', 'timing': ([0., 0.27])},
    {'label': 'one_spk', 'timing': ([0.27, 0.28])},
    {'label': 'two_spk', 'timing': ([0.28, 0.44])},
    {'label': 'three_spk', 'timing': ([0.44, 0.6 ])}],
    [{'label': 'three_spk', 'timing': ([0.6, 1.2])}],
    [{'label': 'two_spk', 'timing': ([1.2, 1.61 ])},
    {'label': 'three_spk', 'timing': ([1.61 , 1.8])}],
    [{'label': 'two_spk', 'timing': ([1.8 , 2.31])},
    {'label': 'three_spk', 'timing': ([2.31, 2.4 ])}]]

    Returns:
    - labels_with_max_duration (List[List[str]]): Returns a list of lists,
    where each inner list contains the label(s) that occupy the maximum
    total duration within its respective chunk. When a sublist contains
    more than one label, it indicates that these labels
    share the maximum cumulative timing for that chunk, signifying multiple
    majority classes.
    For example, the output [[three_spk], [three_spk], [two_spk], [three_spk]...]
    indicates that 'three_spk' is
    the predominant label in the first, second, and fourth chunks, while
    'two_spk' dominates the third chunk.
    In cases where sublists contain multiple labels, they represent chunks
     where there is a tie for the maximum total duration between different labels.

    """

    labels_with_max_duration = []
    for sublist in list_of_labels_timings_in_chunks:
        current_timing = {}

        for item in sublist:
            label = item["label"]
            duration = item["timing"]
            # This retrieves the current accumulated duration for the label.
            # If the label doesn't exist in the dictionary yet, it returns 0.
            # It's a way of initializing the duration for new labels.
            current_timing[label] = current_timing.get(label, 0) + (
                duration[1] - duration[0]
            )

        # Find the longest duration
        longest_duration = max(current_timing.values())
        max_labels = [
            label
            for label, duration in current_timing.items()
            if duration == longest_duration
        ]
        labels_with_max_duration.append(max_labels)

    return labels_with_max_duration


def chunk_signal(
    signal_list: List[float], sample_rate: int, chunk_seconds: float
) -> List[List[float]]:
    """
    Chunk the input signal list into smaller chunks. Signals that do not fit
    completely into a chunk are excluded in the final chunk list. Drop last chunk for good measure.

    Parameters:
    - signal_list (List[float]): A list containing the input signal values,
     where each value represents the signal
     amplitude at a discrete time point. For example, [0.0024719238, 0.002319336,
      0.0013427734, ...] represents a sequence of signal amplitudes.
    - sample_rate (int): The sampling rate of the signal, specified in Hertz (Hz).
    This indicates the number of samples collected per second. For instance, a
    sample_rate of 16000 means the signal is sampled 16000 times per second.
    - chunk_seconds (float): The duration of each chunk, measured in seconds.
    This parameter defines how long each segment of the signal should be when
    the signal is divided into smaller, consecutive chunks.
    For example, a chunk_seconds value of 0.6 indicates that each
    chunk of the signal spans 0.6 seconds.

    Returns:
    - list_of_chunk_signals (List[List[float]]): List of chunked signals.
    For instance: [[0.0024719238, 0.002319336, 0.0013427734, 0.0...]]
    where each sublist within a list represents signals for a specified duration.
    """

    chunk_size_ms = chunk_seconds * 1000

    # Floor division is used here to ensure the result is an integer.
    # This operation rounds down to the nearest integer.
    # As a result, any remaining signals that do not fit completely
    # into a chunk are excluded from the final chunked list.
    num_samples_per_chunk = int(sample_rate * chunk_size_ms // 1000)

    list_of_chunk_signals = []

    for i in range(0, len(signal_list), num_samples_per_chunk):
        chunk = signal_list[i : i + num_samples_per_chunk]
        # If length of chunk is equivalent to the number of samples per chunk,
        # append to list
        if len(chunk) == num_samples_per_chunk:
            list_of_chunk_signals.append(chunk)

    # list_of_chunk_signals = list_of_chunk_signals[:-1]

    return list_of_chunk_signals


def chunk_timings_metadata(row: dict, chunk_seconds: float) -> List[str]:
    """
    Chunk timings in the metadata and generate filenames.

    Parameters:
    - row (dict): Dictionary representing a row in the metadata.
    - chunk_seconds (float): duration of a chunk (in seconds)

    Returns:
    - filenames (List[str]): List of filenames generated based on
    chunked timings (obtained from
    length of  chunks_class).
    """

    filenames = []
    num_indx = len(row["chunks_class"])
    audio_id_name = row["audio_id"]

    for i in range(num_indx):
        filenames.append(
            f"{audio_id_name}_{round(i * (chunk_seconds), 3)}_{round(((i*chunk_seconds)+chunk_seconds), 3)}"
        )

    return filenames


def adjust_lists(row):
    """
    Adjust list within each row (dictionary) to ensure that the number of
    elements in "amp_chunks" and "labels_and_timings_of_chunks" are equal.
    This function is necessary to handle potential discrepancies in lengths
    that may arise due to floating-point precision errors, ensuring data consistency.

    Parameters:
    - row (Dict[str, List]): A dictionary representing a row, containing
    the keys 'amp_chunks' and 'labels_and_timings_of_chunks'.

    Returns:
    - row (Dict[str, List]): The adjusted row dictionary with equal length
    lists for 'amp_chunks' and 'labels_and_timings_of_chunks'.
    """

    # Extract the amplitude chunks and labels and timings of chunks from the row
    chunk_amps = row["amp_chunks"]
    labels_and_timings_of_chunks = row["labels_and_timings_of_chunks"]

    # Calculate the difference in lengths between the two lists
    length_difference = len(labels_and_timings_of_chunks) - len(chunk_amps)

    # Trim excess if one column has more 'content' than the other
    if length_difference > 0:
        num_lack_indexes = len(labels_and_timings_of_chunks) - len(chunk_amps)
        labels_and_timings_of_chunks = labels_and_timings_of_chunks[:-num_lack_indexes]

        row["labels_and_timings_of_chunks"] = labels_and_timings_of_chunks

    elif length_difference < 0:
        num_lack_indexes = len(chunk_amps) - len(labels_and_timings_of_chunks)
        chunk_amps = chunk_amps[:-num_lack_indexes]

        row["amp_chunks"] = chunk_amps

    return row


def generate_specific_file_name(
    split_type: str, mix_type: str, part_type: str, file_extension: str
) -> str:
    """
    Generate a specific file name based on the split type, mix type, part type,
     and file extension.

    Args:
        split_type (str): The type of split, which may contain a hyphen.
        mix_type (str): The type of mix.
        part_type (str): The part type, which can be None if not applicable.
        file_extension (str): The file extension to be appended to the file name.

    Returns:
        str: The constructed specific file name.

    Examples:
        >>> generate_specific_file_name('train-100', 'both', 'part1',
        '_2mix_osd_labels.parquet.gzip')
        'train_100_both_part1_2mix_osd_labels.parquet.gzip'

    """
    if "-" in split_type:
        parts = split_type.split("-")
        split_mix_type = f"{parts[0]}_{parts[1]}_{mix_type}"
    else:
        split_mix_type = f"{split_type}_{mix_type}"

    if part_type is not None:
        specific_file_name = f"{split_mix_type}_{part_type}{file_extension}"
    else:
        specific_file_name = f"{split_mix_type}{file_extension}"

    return specific_file_name


def get_chunks_annotation_librispeech_in_parts(
    wave_type: str,
    mix_type: str,
    sample_rate: int,
    split_type: str,
    num_speaker_mix: str,
    chunk_seconds: float,
    metadata_directory_pathway: str,
    file_extension: str,
    librimix_audio_folder: str,
    hash_audioid_mix: pd.DataFrame,
    part_type: str = None,
) -> pd.DataFrame:
    """
    Get annotated chunks from LibriSpeech metadata with two speakers in parts.

    Parameters:
    - wave_type (str): Wave file type (e.g., 'wav8k', 'wav16k').
    - mix_type (str): Mixture type ('both', 'clean').
    - sample_rate (int): Sampling rate.
    - split_type (str): Split type (e.g., 'dev', 'train-100', 'test').
    - num_speaker_mix (str): Number of speakers in the mix ('Libri2Mix', 'Libri3Mix').
    - chunk_seconds (float): Duration of chunks in seconds ('0.6').
    - metadata_directory_pathway (str): directory to metadata folder
    - file_extension (str): metadata file extension (e.g. "_2mix_osd_labels.parquet.gzip")
    - librimix_audio_folder (str): pathway to folder containing audio clips
    - hash_audioid_mix (pd.DataFrame): hashing document
    - part_type (str, optional): Part type for handling metadata.

    Returns:
    - dataset (pd.DataFrame): DataFrame with annotated chunks (e.g. producing
    the following additional columns:
    signals of chunks, respective filenames, chunk class: majority labels).

    """

    specific_file_name = generate_specific_file_name(
        split_type, mix_type, part_type, file_extension
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
    dataset.rename(columns={"overlaps": "two_spk"}, inplace=True)

    dataset = remove_filtered_cols(
        ["non_speech", "one_spk", "two_spk"], dataset=dataset
    )
    # Filter accordingly (hashing document)
    if split_type == "train-100":
        audio_id_df = hash_audioid_mix
        audio_id_df = audio_id_df[audio_id_df["split"] == "evaluation"]
        dataset = dataset[dataset["audio_id"].isin(audio_id_df["audio_id"])]

    # Create input signal list
    dataset["amp"] = dataset["audio_id"].apply(
        add_audio_amp_librimix,
        wave_type=wave_type,
        split_type=split_type,
        mix_type=mix_type,
        num_speaker_mix=num_speaker_mix,
        librimix_audio_folder=librimix_audio_folder,
    )

    # Chunk the input signal list into smaller chunks.
    dataset["amp_chunks"] = dataset["amp"].apply(
        lambda x: chunk_signal(x, sample_rate, chunk_seconds)
    )

    # Generate chunk timings and labels for each row in the DataFrame.
    dataset = produce_chunk_timing_and_labels(dataset, chunk_seconds)

    dataset = dataset.apply(adjust_lists, axis=1)

    # Identifies the label(s) within each chunk of signals that have the
    # maximum cumulative duration (majority class)
    dataset["chunks_class"] = dataset["labels_and_timings_of_chunks"].apply(
        produce_chunk_label
    )

    # Chunk timings in metadata and generate respective filenames.
    dataset["chunk_filenames"] = dataset.apply(
        lambda row: chunk_timings_metadata(row, chunk_seconds),
        axis=1,
    )

    dataset.drop(
        columns=[
            "source1_ranges_list",
            "source2_ranges_list",
            "total_time",
            "speech_times",
            "non_speech",
            "one_spk",
            "two_spk",
            "amp",
            "labels_and_timings_of_chunks",
        ],
        inplace=True,
    )

    return dataset


def save_annotated_chunks_libri2_mix_in_parts(
    wave_type_list: List[str],
    mixture_type_list: List[str],
    split_type_list: List[str],
    num_speaker_mix: str,
    chunk_seconds: float,
    part_type_list: List[str],
    metadata_directory_pathway: str,
    file_extension: str,
    librimix_audio_folder: str,
    hash_audioid_mix: pd.DataFrame,
    batchsize: int,
) -> None:
    """
    Save annotated chunks from Libri2Mix metadata in parts.

    Parameters:
    - wave_type_list (List[str]): List of wave file types (e.g., ['wav8k', 'wav16k']).
    - mixture_type_list (List[str]): List of mixture types (e.g., ['both', 'clean']).
    - split_type_list (List[str]): List of split types (e.g., ['dev',
    'train-100', 'test']).
    - num_speaker_mix (str): Number of speakers in the mix ('Libri2Mix', 'Libri3Mix').
    - chunk_seconds (float): Duration of chunks in seconds ('0.6').
    - part_type_list (List[str]): List of part types (e.g., ['train-360', 'dev',
    'test']).
    - metadata_directory_pathway (str): directory to metadata folder
    - file_extension (str): metadata file extension
    (e.g. "_2mix_osd_labels.parquet.gzip")
    - librimix_audio_folder (str): pathway to folder containing audio clips
    - hash_audioid_mix (pd.DataFrame): hashing document
    - batchsize (int): number of rows to be saved in one dataset

    Returns:
    - partitions (partitioned dataset): containing batches of datasets

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
            logger.error(f"Invalid wave format: {wave_types}")

        for split_type in split_type_list:
            for mixture_types in mixture_type_list:
                for part_type in part_type_list:
                    dataset = get_chunks_annotation_librispeech_in_parts(
                        wave_types,
                        mixture_types,
                        sampling_rate,
                        split_type,
                        num_speaker_mix,
                        chunk_seconds,
                        metadata_directory_pathway,
                        file_extension,
                        librimix_audio_folder,
                        hash_audioid_mix,
                        part_type,
                    )
                    dataset.reset_index(drop=True, inplace=True)

                    # Save DataFrame in batches
                    batch_size = batchsize  # 100 or 800 or 1000
                    num_batches = len(dataset) // batch_size + 1

                    for batch_num in range(num_batches):
                        start_idx = batch_num * batch_size
                        end_idx = min((batch_num + 1) * batch_size, len(dataset))

                        batch_df = dataset.iloc[start_idx:end_idx]
                        batch_df.reset_index(drop=True, inplace=True)

                        # Append part number to the file name
                        part_number = batch_num + 1

                        # Construct a unique key for each partition
                        partition_key = f"libri2mix_{wave_types}_{split_type}_{mixture_types}_{part_type}_part{part_number}.parquet"
                        partitions[partition_key] = batch_df

    return partitions


def get_chunks_annotation_librispeech_3spk_in_parts(
    wave_type: str,
    mix_type: str,
    sample_rate: int,
    split_type: str,
    num_speaker_mix: str,
    chunk_seconds: float,
    metadata_directory_pathway: str,
    file_extension: str,
    librimix_audio_folder: str,
    hash_audioid_mix: pd.DataFrame,
    part_type: str = None,
) -> pd.DataFrame:
    """
    Get annotated chunks from LibriSpeech metadata.

    Parameters:
    - wave_type (str): Wave file type (e.g., 'wav8k', 'wav16k').
    - mix_type (str): Mixture type ('both', 'clean').
    - sample_rate (int): Sampling rate.
    - split_type (str): Split type (e.g., 'dev', 'train-100', 'test').
    - num_speaker_mix (str): Number of speakers in the mix ('Libri2Mix', 'Libri3Mix')
    - chunk_seconds (float): Duration of chunks in seconds ('0.6').
    - metadata_directory_pathway (str): directory to metadata folder
    - file_extension (str): metadata file extension
    (e.g. "_3mix_osd_labels.parquet.gzip").
    - librimix_audio_folder (str): pathway to folder containing audio clips
    - hash_audioid_mix (pd.DataFrame): hashing document
    - part_type (str): file part ('part1').

    Returns:
    - pd.DataFrame: DataFrame with annotated chunks.

    Example:
    >>> dataset = get_chunks_annotation_librispeech_3spk_in_parts('wav16k', 'both',
    16000, 'train-100', 'Libri3Mix', 0.6, 'part1')
    """

    specific_file_name = generate_specific_file_name(
        split_type, mix_type, part_type, file_extension
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

    dataset = remove_filtered_cols(
        ["non_speech", "one_spk", "two_spk", "three_spk"], dataset=dataset
    )

    # Filter accordingly (hashing document)
    if split_type == "train-100":
        audio_id_df = hash_audioid_mix
        audio_id_df = audio_id_df[audio_id_df["split"] == "evaluation"]
        dataset = dataset[dataset["audio_id"].isin(audio_id_df["audio_id"])]

    dataset["amp"] = dataset["audio_id"].apply(
        add_audio_amp_librimix,
        wave_type=wave_type,
        split_type=split_type,
        mix_type=mix_type,
        num_speaker_mix=num_speaker_mix,
        librimix_audio_folder=librimix_audio_folder,
    )

    # Chunk the input signal list into smaller chunks.
    dataset["amp_chunks"] = dataset["amp"].apply(
        lambda x: chunk_signal(x, sample_rate, chunk_seconds)
    )
    # Generate chunk timings and labels for each row in the DataFrame.
    dataset = produce_chunk_timing_and_labels_3spk(dataset, chunk_seconds)

    dataset = dataset.apply(adjust_lists, axis=1)

    # Identifies the label(s) within each chunk of signals that have the maximum
    # cumulative duration (majority class)
    dataset["chunks_class"] = dataset["labels_and_timings_of_chunks"].apply(
        lambda x: produce_chunk_label(x)
    )

    # Chunk timings in metadata and generate respective filenames.
    dataset["chunk_filenames"] = dataset.apply(
        lambda row: chunk_timings_metadata(row, chunk_seconds), axis=1
    )
    dataset.drop(
        columns=[
            "source1_ranges_list",
            "source2_ranges_list",
            "source3_ranges_list",
            "total_time",
            "speech_times",
            "non_speech",
            "one_spk",
            "two_spk",
            "three_spk",
            "amp",
            "labels_and_timings_of_chunks",
        ],
        inplace=True,
    )

    return dataset


def save_annotated_chunks_libri3_mix_in_parts(
    wave_type_list: List[str],
    mixture_type_list: List[str],
    split_type_list: List[str],
    num_speaker_mix: str,
    chunk_seconds: float,
    part_type_list: List[str],
    metadata_directory_pathway: str,
    file_extension: str,
    librimix_audio_folder: str,
    hash_audioid_mix: pd.DataFrame,
    batchsize: int,
) -> None:
    """
    Save annotated chunks from Libri3Mix metadata in parts.

    Parameters:
    - wave_type_list (List[str]): List of wave file types (e.g., ['wav8k', 'wav16k']).
    - mixture_type_list (List[str]): List of mixture types (e.g., ['both', 'clean']).
    - split_type_list (List[str]): List of split types (e.g., ['dev', 'train-100',
    'test']).
    - num_speaker_mix (str): Number of speakers in the mix ('Libri3Mix').
    - chunk_seconds (float): Duration of chunks (in seconds).
    - part_type_list (List[str]): List of part types (e.g., ['train-360', 'dev',
    'test']).
    - metadata_directory_pathway (str): directory to metadata folder
    - file_extension (str): metadata file extension.
    (e.g. "_3mix_osd_labels.parquet.gzip")
    - librimix_audio_folder (str): pathway to folder containing audio clips
    - hash_audioid_mix (pd.DataFrame): hashing document
    - batchsize (int): number of rows to be saved in one dataset


    Returns:
    - None

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
            logger.error("Invalid wave format: %s", wave_types)

        for split_type in split_type_list:
            for mixture_types in mixture_type_list:
                for part_type in part_type_list:
                    dataset = get_chunks_annotation_librispeech_3spk_in_parts(
                        wave_types,
                        mixture_types,
                        sampling_rate,
                        split_type,
                        num_speaker_mix,
                        chunk_seconds,
                        metadata_directory_pathway,
                        file_extension,
                        librimix_audio_folder,
                        hash_audioid_mix,
                        part_type,
                    )

                    dataset.reset_index(drop=True, inplace=True)

                    # Save DataFrame in batches
                    batch_size = batchsize  # 100, 800 or 1000
                    num_batches = len(dataset) // batch_size + 1

                    for batch_num in range(num_batches):
                        start_idx = batch_num * batch_size
                        end_idx = min((batch_num + 1) * batch_size, len(dataset))

                        batch_df = dataset.iloc[start_idx:end_idx]

                        batch_df.reset_index(drop=True, inplace=True)

                        # Append part number to the file name
                        part_number = batch_num + 1

                        # Construct a unique key for each partition
                        partition_key = f"libri3mix_{wave_types}_{split_type}_{mixture_types}_{part_type}_part{part_number}.parquet"
                        partitions[partition_key] = batch_df

    return partitions
