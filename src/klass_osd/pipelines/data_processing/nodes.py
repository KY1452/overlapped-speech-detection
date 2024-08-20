"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

import logging
import os
import re
from functools import reduce
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Create unaligned dataframe from unaligned txt file
def create_unaligned_dataframe(unligned_txt_file):
    """
    Create a pandas DataFrame from a txt file containing audio IDs that were unaligned
    and save it as a pandas dataframe (parquet file with gzip compression).

    Args:
        unligned_txt_file (str): The path to the text file containing audio IDs.
        unligned_df_pathway (str): The path to save the Parquet file with gzip
        compression.

    Returns:
        dataset (pd.DataFrame): dataframe saved as parquet file with gzip compression.
    """

    # Extract Audio IDs from the txt file
    ids = re.findall(r"\d{3,5}-\d{3,7}-\d{3,5}", unligned_txt_file)

    # Create a DataFrame
    dataset = pd.DataFrame({"timing": ids})

    print("Saved as pandas dataframe (pq format)")
    return dataset


# Common Intersection Functions (Used to create labels for sparselibrimix and librimix)
def get_intersect(range1: List, range2: List) -> List:
    """
    Returns the intersection of two ranges.

    Args:
        range1 (list): The first range.
        range2 (list): The second range.

    Returns:
        list or None: The intersection of two ranges, or None if no intersection.

    Example:

    """
    intersect_range = [max(range1[0], range2[0]), min(range1[1], range2[1])]
    # If the intersect range consist of 2 of the same numbers, return None, otherwise,
    # return intersect range
    return intersect_range if intersect_range[0] < intersect_range[1] else None


def get_intersect_of_two_list(list1: List[List], list2: List[List]) -> List[List]:
    """
    Returns the intersection of two lists of ranges.

    Args:
        list1 (List[List[float]]): The first list of ranges, where each range
        is specified as [start, end].
        list2 (List[List[float]]): The second list of ranges, where each range
        is specified as [start, end].

    Returns:
        List[List[float]]: The intersection of the two lists of ranges.

    Example:
        list1 = [[3, 5], [9, 10], [11, 25]]
        list2 = [[2, 4], [7, 19], [21, 30]]

        get_intersect_of_two_list(list1, list2)
        >>> [[3, 4], [9, 10], [11, 19], [21, 25]]
    """

    result = []
    for range1 in list1:
        for range2 in list2:
            intersection = get_intersect(range1, range2)
            if intersection:
                result.append(intersection)

    return result


def get_intersect_of_multiple_range(ranges: List[List]) -> List[List]:
    """
    Find the intersection of multiple ranges.

    Args:
        ranges (List[List]): List of ranges.

    Returns:
        List[List]: The intersection of all ranges.

    Example:

        multiple_ranges = [[[3, 5], [9, 10], [11, 25]],
        [[3, 4], [7, 19], [21, 30]],
        [[3, 5], [6, 8], [9, 11]]]

        get_intersect_of_multiple_range(multiple_ranges)

        >>> [[3, 4], [9, 10]]
    """
    return reduce(get_intersect_of_two_list, ranges)


def combine_overlaps(ranges: List[List]) -> List[List]:
    """
    Combines overlapping ranges into a single range.

    This function calculates the inverted timings by removing the specified ranges
    from the total range. It partitions the total range into intervals that are not
    overlapped by any of the removal ranges, and then further refines these intervals
    by excluding portions intersecting with the removal ranges.

    Args:
        ranges (list): A list of ranges to be combined.

    Returns:
        list: The combined ranges.

    Example:
        >>> combine_overlaps([[1, 3], [2, 4], [5, 7]])
        [[1, 4], [5, 7]]
    """
    combined_ranges = []
    for start, end in sorted(ranges):  # sort input ranges by their start values
        if combined_ranges and combined_ranges[-1][1] >= start:
            # If the current range overlaps with the previous range, update the end
            # value of the previous range to the maximum of the two end values.
            combined_ranges[-1][1] = max(combined_ranges[-1][1], end)
        else:
            # If the current range does not overlap with the previous range, add it to
            # the combined_ranges list.
            combined_ranges.append([start, end])
    return combined_ranges


def invert_timings(
    all_range_to_remove: List[List], all_total_range: List[List]
) -> List[List]:
    """
    Inverts the timings by removing the given range from the total range.

    This function calculates the inverted timings by removing the specified ranges
    from the total range. It does this in two steps:

    1. Partitioning: It divides the total range into intervals that do not overlap
       with any of the removal ranges.

    2. Refinement: It further refines these intervals by excluding any portions that
       intersect with the removal ranges.

    Args:
        all_range_to_remove (List[List]): The list of ranges to be removed.
        all_total_range (List[List]): The total range to be inverted.
            Each range should be specified as a list containing two elements:
            [start, end].

    Returns:
        result_intervals (List[List]): The inverted timings represented as a list of
        non-overlapping intervals.
            Each interval is specified as a list containing two elements: [start, end].

    Example (code):
        invert_timings([[3, 5], [7, 11]], [[0, 30]])
        >>> [[0, 3], [5, 7], [11, 30]]

    Example:
        Illustration:
        Total_range: |--------------------|
        Range_to_remove:   |----|
        Appended:    *-----*    *---------*

    """

    result_intervals = []
    range_no_overlaps = []
    non_overlap_part_within_intersect = []

    # Find total_range with no overlaps
    for total_range in all_total_range:
        if all(
            not get_intersect(range_to_remove, total_range)
            for range_to_remove in all_range_to_remove
        ):
            range_no_overlaps.append(total_range)

    result_intervals.extend(range_no_overlaps)

    # Find non_overlap parts within intersects:
    # Example 1:
    # Total_range: |--------------------|
    # Range_to_remove:   |----|
    # Appended:    *-----*    *---------*
    # Note: Range_to_remove is always within Total_range in our case

    for total_range in all_total_range:
        for range_to_remove in all_range_to_remove:
            if get_intersect(range_to_remove, total_range):
                non_overlap_part_within_intersect.append(
                    [total_range[0], range_to_remove[0]]
                )
                non_overlap_part_within_intersect.append(
                    [range_to_remove[1], total_range[1]]
                )

    i = 1
    while i < len(non_overlap_part_within_intersect):
        # Scenario 1 - Process no overlaps within non_overlap_part_within_intersect:
        # non_overlap_part_within_intersect[i-1]: |------------|
        # non_overlap_part_within_intersect[i]:                     |--------------|
        # Appended:                               *------------*

        if (
            non_overlap_part_within_intersect[i - 1][1]
            < non_overlap_part_within_intersect[i][0]
        ):
            result_intervals.append(non_overlap_part_within_intersect[i - 1])

        # Scenario 2 - Process intersects within non_overlap_part_within_intersect:
        # non_overlap_part_within_intersect[i-1]:         |------------|
        # non_overlap_part_within_intersect[i]:  |--------------|
        # Appended:                                       *-----* and skip to i+1
        # element for next iteration

        else:
            result_intervals.append(
                [
                    max(
                        non_overlap_part_within_intersect[i - 1][0],
                        non_overlap_part_within_intersect[i][0],
                    ),
                    min(
                        non_overlap_part_within_intersect[i - 1][1],
                        non_overlap_part_within_intersect[i][1],
                    ),
                ]
            )
            i += 1

        # Scenario 3 - Handling the last element in non_overlap_part_within_intersect
        # not processed in scenario 1 & 2:
        # non_overlap_part_within_intersect[i]:           |--------------|
        # Appended:                                       *--------------*

        if i == len(non_overlap_part_within_intersect) - 1:
            result_intervals.append(non_overlap_part_within_intersect[i])
        i += 1

    # Remove duplicate ranges and return sorted intervals
    result_intervals = [
        [first, second] for first, second in result_intervals if first != second
    ]

    return sorted(result_intervals)


# SPARSELIBRIMIX
def convert_sparselibrimix_to_dataframe(
    data: list, partition_id: str, no_of_speakers: int
) -> pd.DataFrame:
    """
    Convert list data to a DataFrame.

    Args:
        data (list): The path to the JSON file.
        partition_id (str): The ID of the partition to identify the source folder.
        no_of_speakers (int): The number of speakers in the data.

    Returns:
        pd.DataFrame: The converted DataFrame.
    """

    # Use pd.json_normalize to convert the JSON to a DataFrame
    # Extract labels for 's1', 's2', and 'noise' from 'data' and add
    # 'mixture_name' as a column
    df_source1 = pd.json_normalize(data, "s1", ["mixture_name"])
    df_source2 = pd.json_normalize(data, "s2", ["mixture_name"])
    df_noise = pd.json_normalize(data, "noise", ["mixture_name"])

    # Select specific columns from df_s1 and df_s2
    df_source1 = df_source1[["start", "stop", "mixture_name", "source"]]
    df_source2 = df_source2[["start", "stop", "mixture_name", "source"]]
    df_noise = df_noise[["start", "stop", "mixture_name", "source"]].round(3)

    # Create a new column 'start-stop' by combining 'start' and 'stop' columns
    # as an array
    df_source1["start-stop"] = df_source1[["start", "stop"]].apply(np.array, axis=1)
    df_source2["start-stop"] = df_source2[["start", "stop"]].apply(np.array, axis=1)
    df_noise["start-stop"] = df_noise[["start", "stop"]].apply(np.array, axis=1)

    df_source3 = pd.DataFrame()

    if no_of_speakers == 3:
        df_source3 = pd.json_normalize(data, "s3", ["mixture_name"])
        df_source3 = df_source3[
            ["start", "stop", "sub_utt_num", "mixture_name", "source"]
        ]
        df_source3["start-stop"] = df_source3[["start", "stop"]].apply(np.array, axis=1)

    # Concatenate df_s1, df_s2, df_s3, and df_n
    concatenated_df = pd.concat([df_source1, df_source2, df_source3, df_noise])

    folder_name = partition_id
    concatenated_df["mixture_name"] = folder_name + concatenated_df["mixture_name"]

    return concatenated_df


def get_labels_sparse2(concatenated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in concatenated dataframe and returns a dictionary of label intersections
    for each mixture: no_speech, 1_speaker, 2_speaker.

    Args:
        concatenated_df (pd.DataFrame): The concatenated DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the labels for each mixture.
    """

    # Initialize empty lists to store labels
    overlaps = []
    audio_id = []
    speech = []
    total_time = []
    non_speech = []
    one_spk = []

    # Iterate over each group in concat_grouped
    for mixture_name, group in concatenated_df.groupby("mixture_name"):
        # Filter the group for 'source_1' and 'source_2' sources, and reset the index
        source_1 = (
            group[group["source"] == "s1"]["start-stop"].reset_index(drop=True)
        ).values.tolist()
        source_2 = (
            group[group["source"] == "s2"]["start-stop"].reset_index(drop=True)
        ).values.tolist()
        durations = (
            group[group["source"] == "noise"]["start-stop"].reset_index(drop=True)
        ).values.tolist()

        overlaps.append(get_intersect_of_two_list(source_1, source_2))
        audio_id.append(mixture_name)

        speech_times = combine_overlaps(
            [
                (float(x[0]), float(x[1]))
                for sublist in [source_1, source_2]
                for x in sublist
            ]
        )
        speech.append(speech_times)

        total_time.append(durations)

        non_speech.append(invert_timings(speech_times, durations))

        overlap_or_nonspeech = get_intersect_of_two_list(
            source_1, source_2
        ) + invert_timings(speech_times, durations)
        one_spk.append(
            invert_timings(sorted(overlap_or_nonspeech, key=lambda x: x[0]), durations)
        )

    label_df = pd.DataFrame(
        {
            "audio_id": audio_id,
            "total_time": total_time,
            "speech_times": speech,
            "non_speech": non_speech,
            "one_spk": one_spk,
            "two_spk": overlaps,
        }
    )

    return label_df


def get_labels_sparse3(concatenated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in concatenated dataframe and returns a dataframe of labels for each mixture:
    no_speech, 1_speaker, 2_speaker, 3_speaker.

    Args:
        concatenated_df (pd.DataFrame): The concatenated DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the labels for each mixture.
    """

    # Initialize empty lists to store labels
    audio_id = []
    speech = []
    total_time = []
    non_speech = []
    one_spk = []
    two_spk = []
    two_or_three_spk = []
    three_spk = []

    # Iterate over each mixture name and its corresponding group
    for mixture_name, group in concatenated_df.groupby("mixture_name"):
        any_intersect = []

        # Get start-stop arrays for each source
        source_1 = (
            group[group["source"] == "s1"]["start-stop"].reset_index(drop=True)
        ).values.tolist()
        source_2 = (
            group[group["source"] == "s2"]["start-stop"].reset_index(drop=True)
        ).values.tolist()
        source_3 = (
            group[group["source"] == "s3"]["start-stop"].reset_index(drop=True)
        ).values.tolist()
        durations = (
            group[group["source"] == "noise"]["start-stop"].reset_index(drop=True)
        ).values.tolist()

        # Find intersections between timelines
        for intersection in get_intersect_of_two_list(source_1, source_2):
            any_intersect.append(intersection)
        for intersection in get_intersect_of_two_list(source_1, source_3):
            any_intersect.append(intersection)
        for intersection in get_intersect_of_two_list(source_2, source_3):
            any_intersect.append(intersection)

        two_or_three_spk.append(combine_overlaps(any_intersect))

        three_spk.append(
            get_intersect_of_multiple_range([source_1, source_2, source_3])
        )
        audio_id.append(mixture_name)

        speech_times = [
            (float(x[0]), float(x[1]))
            for sublist in [source_1, source_2, source_3]
            for x in sublist
        ]
        speech.append(combine_overlaps(speech_times))

        total_time.append(durations)

        silences = invert_timings(combine_overlaps(speech_times), durations)
        non_speech.append(silences)

        overlap_or_nonspeech = combine_overlaps(any_intersect) + silences
        one_spk.append(
            invert_timings(sorted(overlap_or_nonspeech, key=lambda x: x[0]), durations)
        )
        two_spk.append(
            invert_timings(
                get_intersect_of_multiple_range([source_1, source_2, source_3]),
                combine_overlaps(any_intersect),
            )
        )

    label_df = pd.DataFrame(
        {
            "audio_id": audio_id,
            "total_time": total_time,
            "speech_times": speech,
            "non_speech": non_speech,
            "one_spk": one_spk,
            "two_or_three_spk": two_or_three_spk,
            "two_spk": two_spk,
            "three_spk": three_spk,
        }
    )
    return label_df


def iterate_sparselibrimix_2(partitioned_input: Dict[str, callable]) -> List:
    """Iterates through partitions for sparselibrimix2 and generates processed labels.

    Args:
        partitioned_input (dict): A dictionary containing partition IDs as keys and
                                  partition load functions as values.

    Returns:
        list: Processed labels.

    """
    concatenated_df_2 = pd.DataFrame()

    for partition_id, partition_load_func in sorted(partitioned_input.items()):
        if "sparse_2" in partition_id:
            partition_data = partition_load_func()  # load the actual partition data
            partition_data = convert_sparselibrimix_to_dataframe(
                partition_data, partition_id, 2
            )
            concatenated_df_2 = pd.concat(
                [concatenated_df_2, partition_data], ignore_index=True, sort=True
            )
            processed_labels = get_labels_sparse2(concatenated_df_2)

    return processed_labels


def iterate_sparselibrimix_3(partitioned_input: Dict[str, callable]) -> List:
    """Iterates through partitions for sparselibrimix3 and generates processed labels.

    Args:
        partitioned_input (dict): A dictionary containing partition IDs as keys and
                                  partition load functions as values.

    Returns:
        list: Processed labels.

    """
    concatenated_df_3 = pd.DataFrame()

    for partition_id, partition_load_func in sorted(partitioned_input.items()):
        if "sparse_3" in partition_id:
            partition_data = partition_load_func()  # load the actual partition data
            partition_data = convert_sparselibrimix_to_dataframe(
                partition_data, partition_id, 3
            )
            concatenated_df_3 = pd.concat(
                [concatenated_df_3, partition_data], ignore_index=True, sort=True
            )
            processed_labels = get_labels_sparse3(concatenated_df_3)

    return processed_labels


# LIBRIMIX
# Part 1: Iterates through all .txt files in the
# specified root directory and processes each file to extract
# relevant information, creating a DataFrame.


def create_table(partition_data: str) -> pd.DataFrame:
    """
    Creates a table from data in a file.

    Args:
        partition_data (str): input txt from dataset

    Returns:
        pandas.DataFrame: The created table.
    """

    # Create empty lists to store the data for each column
    unique_ids = []  # List to store unique ids
    words = []  # List to store words
    timing = []  # List to store timing

    partition_data = partition_data.split("\n")

    try:
        for line in partition_data:
            line = line.strip()  # Remove leading and trailing whitespaces
            parts = line.split(
                '"', 2
            )  # Split the line based on the closing double quote, but only split twice
            if len(parts) == 3:
                unique_id = parts[0].strip(', "')  # Extract unique id
                words_part = parts[1].strip(', "').split(",")  # Extract words

                # Remove leading and trailing '""'
                if words_part[0] == '""':
                    words_part.pop(0)
                if words_part[-1] == '""':
                    words_part.pop(-1)

                words_str = [
                    f'"{word}"' for word in words_part
                ]  # Add double quotes to words
                numbers_part = parts[2].strip().replace('"', "")  # Extract numbers
                numbers_list = [
                    float(num) for num in numbers_part.split(",")
                ]  # Convert timing to float

                unique_ids.append(unique_id)
                words.append(words_str)
                timing.append(numbers_list)

    except Exception:
        raise ValueError("An error occurred while processing the partition data")
        # pass

    # Create a DataFrame from the extracted data
    data = {"unique_id": unique_ids, "words": words, "timing": timing}

    dataset = pd.DataFrame(data)

    return dataset


def create_start_stop(dataset: pd.DataFrame) -> List:
    """
    Creates a list of start and stop indexes for each pause "" in the list of words.

    Args:
        dataset (DataFrame): The input DataFrame.

    Returns:
        List(float): A list of start and stop indexes for each word.
    """

    start_stop = []

    for num in range(len(dataset["words"])):
        # Find the indexes where the word is '""'(silence)
        indexes = [i for i, item in enumerate(dataset["words"][num]) if item == '""']
        start_stop.append(indexes)

    return start_stop


def create_pause_start_time(dataset: pd.DataFrame, start_stop: List) -> List:
    """
    Create a list of pause start times based on the pause locations within a DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing "timing" and "pause_start_time"
        columns.
        start_stop (list[float]): A list of pause locations.

    Returns:
        list[float]: A list of start times for each audio segment, extracted from the
        "timing" column.
    """

    # List of pause start time
    list_of_pause_start_time = []
    for num in range(len(dataset["timing"])):
        pause_indices = start_stop[num]

        # Initialize an empty list to store the timings
        pause_timings = []

        # Iterate over the indices and extract the corresponding index
        for index in pause_indices:
            pause_timings.append(dataset["timing"][num][index])

        list_of_pause_start_time.append(pause_timings)

    return list_of_pause_start_time


def create_pause_stop_time(dataset: pd.DataFrame, start_stop: List) -> List:
    """
    Create a list of stop times based on the pause locations within a DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing "timing" and
        "pause_start_time" columns.
        start_stop (list(float)): A list of pause locations (float).

    Returns:
        list: A list of stop times for each audio segment, extracted
        from the "timing" column.
    """

    # List of pause stop time

    list_of_pause_stop_time = []

    for num in range(len(dataset["timing"])):
        indices = start_stop[num]

        # Initialize an empty list to store the selected numbers
        pause_timings = []

        # Iterate over the indices and extract the corresponding index
        for index in indices:
            pause_timings.append(dataset["timing"][num][index + 1])

        list_of_pause_stop_time.append(pause_timings)

    return list_of_pause_stop_time


# Function to calculate start and stop times
def audio_start_stop_time(row: pd.Series) -> Tuple[List, List]:
    """
    Calculate the start and stop times of the actual audio based on
    pauses and word timings.

    Args:
        row (pd.Series): A row from the DataFrame containing "timing,"
        "pause_start_time,"
        and "pause_stop_time" columns.
        dataset: dataframe

    Returns:
        tuple: A tuple containing two lists of start and stop times for the
        actual audio, where start and stop times are floats.
        The start times indicate when the audio starts, and the stop times
        indicate when the audio stops.
    """

    start_times = [row["timing"][0]]
    stop_times = []

    for i in range(len(row["pause_start_time"])):
        start_times.append(row["pause_stop_time"][i])
        stop_times.append(row["pause_start_time"][i])

    stop_times.append(row["timing"][-2])

    return start_times, stop_times


# Function to process a single .txt file
def process_txt_file(partitioned_data: list, final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single .txt file to extract audio timing information and
    append it to a DataFrame.

    Args:
        partitioned_data (list): .txt file in list format.
        final_df (pd.DataFrame): The DataFrame to which the extracted
        information will be appended.

    Returns:
        final_df (pd.DataFrame): The updated DataFrame containing the audio timing
        information.
    """

    dataset = create_table(partitioned_data)
    start_stop = create_start_stop(dataset)
    list_of_pause_stop_time = create_pause_stop_time(dataset, start_stop)
    list_of_pause_start_time = create_pause_start_time(dataset, start_stop)
    dataset["pause_start_time"] = list_of_pause_start_time
    dataset["pause_stop_time"] = list_of_pause_stop_time
    dataset[["start_time", "stop_time"]] = dataset.apply(
        audio_start_stop_time, axis=1, result_type="expand"
    )
    final_df = pd.concat([final_df, dataset])
    return final_df


def iterate_files_produce_df(txt_directory: str) -> pd.DataFrame:
    """
    Iterates through all .txt files in the specified directory and
    its subdirectories, processes each file to extract relevant information,
    and creates a DataFrame.

    Parameters:
    - txt_directory (str): Path to the directory containing .txt files.

    Returns:
    - dataset (pd.DataFrame): DataFrame containing the extracted information.
    """
    dataset = pd.DataFrame(
        data=None, columns=["unique_id", "words", "timing", "start_time", "stop_time"]
    )

    for root, dirs, files in os.walk(txt_directory):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    partition_data = f.read()
                dataset = process_txt_file(partition_data, dataset)
                dataset = dataset.reset_index(drop=True)

    return dataset


# Get duration of audio
def get_duration(dataset: pd.DataFrame, sampling_rate: float) -> pd.DataFrame:
    """
    Adds a column called "audio_total_duration" to the provided DataFrame
    based on the audio length and sampling rate.

    Args:
        dataset: A Pandas DataFrame containing an "audio_total_duration" column in
        addition to a "length" column.
        length: The "length" column should represent the audio duration in seconds.
        sampling_rate: The sampling rate of the audio (e.g., 16000 for 16 kHz audio;
        8000 for 8kHz audio).

    Returns:
        A new DataFrame with the same data as the original, but with an added
        "audio_total_duration" column representing the audio duration in seconds.

    Raises:
        TypeError: If either the DataFrame or the sampling rate is not of the
        expected type.
    """

    dataset["audio_total_duration"] = dataset["length"] / sampling_rate
    return dataset


# Remove unaligned audio


def remove_unaligned_3spk(dataset, unaligned_df):
    """
    Remove unaligned rows from a DataFrame based on a list of numbers.

    Args:
        dataset (pandas.DataFrame): The input DataFrame.
        unaligned_df (pandas.DataFrame): The DataFrame containing the unaligned numbers.

    Returns:
        pandas.DataFrame: The filtered DataFrame with unaligned rows removed.
    """

    # Remove rows where 'source_1' column is in the list of unaligned numbers
    dataset = dataset[~dataset["source_1"].isin(list(unaligned_df["timing"]))]

    # Remove rows where 'source_2' column is in the list of unaligned numbers
    dataset = dataset[~dataset["source_2"].isin(list(unaligned_df["timing"]))]

    # Remove rows where 'source_3' column is in the list of unaligned numbers
    dataset = dataset[~dataset["source_3"].isin(list(unaligned_df["timing"]))]

    return dataset


def remove_unaligned_2spk(dataset, unaligned_df):
    """
    Remove unaligned rows from a DataFrame based on a list of numbers.

    Args:
        dataset (pandas.DataFrame): The input DataFrame.
        unaligned_df (pandas.DataFrame): The DataFrame containing the unaligned numbers.

    Returns:
        pandas.DataFrame: The filtered DataFrame with unaligned rows removed.
    """

    # Remove rows where 'source_1' column is in the list of unaligned numbers
    dataset = dataset[~dataset["source_1"].isin(list(unaligned_df["timing"]))]

    # Remove rows where 'source_2' column is in the list of unaligned numbers
    dataset = dataset[~dataset["source_2"].isin(list(unaligned_df["timing"]))]

    return dataset


# Extract start stop times


def extract_start_stop_times(
    metadata_df: pd.DataFrame, source_column: str, dataset: pd.DataFrame
) -> Tuple[List, List]:
    """
    Extracts the start and stop times for each source from a metadata dataframe and a
    dataframe containing unique IDs and start/stop times.

    Args:
        metadata_df (pandas.DataFrame): The metadata dataframe containing the source
        IDs and other columns like (SNR).
        source_column (str): The name of the source column in the metadata dataframe.
        dataset (pandas.DataFrame): The dataframe containing unique IDs and start/stop
        times.

    Returns:
        Tuple[List, List]: A tuple containing two lists: one for start times and
        one for stop times.
    """

    start_times = []
    stop_times = []
    timing = []

    for sourceid in metadata_df[source_column]:
        try:
            start_time = dataset.loc[dataset["unique_id"] == sourceid][
                "start_time"
            ].iloc[0]
        except ValueError:
            start_time = []
            logging.error("No start time found for source ID: %s", sourceid)
        except IndexError:
            start_time = []
            logging.error(
                "Index error found for source ID: %s. The audio ID is likely included in the unaligned text, and no pause start or stop time is available. This audio ID will be discarded later on in another function.",
                sourceid,
            )

        try:
            stop_time = dataset.loc[dataset["unique_id"] == sourceid]["stop_time"].iloc[
                0
            ]
        except ValueError:
            stop_time = []
            logging.error("No stop time found for source ID: %s", sourceid)

        except IndexError:
            stop_time = []
            logging.error(
                "Index error found for source ID: %s. The audio ID is likely included in the unaligned text, and no pause start or stop time is available. This audio ID will be discarded later on in another function.",
                sourceid,
            )

        try:
            time = dataset.loc[dataset["unique_id"] == sourceid]["timing"].iloc[0]
        except ValueError:
            time = []
            logging.error("No stop time found for source ID: %s", sourceid)
        except IndexError:
            time = []
            logging.error(
                "Index error found for source ID: %s. The audio ID is likely included in the unaligned text, and no pause start or stop time is available. This audio ID will be discarded later on in another function.",
                sourceid,
            )

        start_times.append(start_time)
        stop_times.append(stop_time)
        timing.append(time)

    # Speech timing (starting and ending timing)
    speech_timing = []

    for indx, item in enumerate(start_times):
        speech_timing_row = []
        for i in range(len(item)):
            speech_timing_row.append(
                [start_times[indx][i], stop_times[indx][i]]
            )  # append the speech timing

        speech_timing.append(speech_timing_row)

    return start_times, stop_times, timing, speech_timing


# 2spk and 3spk metadata


def update_metadata_2spk(
    metadata_df: pd.DataFrame, dataset: pd.DataFrame, unaligned_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates the metadata dataframe with start and stop times for each source and
    removes unaligned data.

    Args:
        metadata_df (pandas.DataFrame): The metadata dataframe to update.
        dataset (pandas.DataFrame): The dataframe containing unique IDs and start/stop
        times.
        unaligned_df (pandas.DataFrame): The unaligned dataframe to remove from
        the metadata.

    Returns:
        pandas.DataFrame: The updated metadata dataframe.
    """

    # Split the mixture_ID column into source_1, source_2 columns
    metadata_df[["source_1", "source_2"]] = metadata_df["mixture_ID"].str.split(
        r"_(?=\d)", expand=True
    )

    # Extract start and stop times for each source
    (
        source1_start_time,
        source1_stop_time,
        source1_timing,
        source1_speech_timing,
    ) = extract_start_stop_times(metadata_df, "source_1", dataset)
    (
        source2_start_time,
        source2_stop_time,
        source2_timing,
        source2_speech_timing,
    ) = extract_start_stop_times(metadata_df, "source_2", dataset)

    # Add start and stop times; timing to the metadata dataframe
    metadata_df["source1_start_time"] = source1_start_time
    metadata_df["source1_stop_time"] = source1_stop_time
    metadata_df["source1_timing"] = source1_timing
    metadata_df["source1_speech_timing"] = source1_speech_timing
    metadata_df["source2_start_time"] = source2_start_time
    metadata_df["source2_stop_time"] = source2_stop_time
    metadata_df["source2_timing"] = source2_timing
    metadata_df["source2_speech_timing"] = source2_speech_timing

    # Remove unaligned data: remove_unaligned_2spk(metadata_df, unaligned_df)
    metadata_df = remove_unaligned_2spk(metadata_df, unaligned_df)

    # Reset the index
    metadata_df = metadata_df.reset_index(drop=True)

    return metadata_df


def update_metadata_3spk(
    metadata_df: pd.DataFrame, dataset: pd.DataFrame, unaligned_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates the metadata dataframe with start and stop times for each source and
    removes unaligned data.

    Args:
        metadata_df (pandas.DataFrame): The metadata dataframe to update.
        dataset (pandas.DataFrame): The dataframe containing unique IDs and start/stop
        times.
        unaligned_df (pandas.DataFrame): The unaligned dataframe to remove from
        the metadata.

    Returns:
        pandas.DataFrame: The updated metadata dataframe.
    """

    # Split the mixture_ID column into source_1, source_2, and source_3 columns
    metadata_df[["source_1", "source_2", "source_3"]] = metadata_df[
        "mixture_ID"
    ].str.split(r"_(?=\d)", expand=True)

    # Extract start and stop times for each source
    (
        source1_start_time,
        source1_stop_time,
        source1_timing,
        source1_speech_timing,
    ) = extract_start_stop_times(metadata_df, "source_1", dataset)
    (
        source2_start_time,
        source2_stop_time,
        source2_timing,
        source2_speech_timing,
    ) = extract_start_stop_times(metadata_df, "source_2", dataset)
    (
        source3_start_time,
        source3_stop_time,
        source3_timing,
        source3_speech_timing,
    ) = extract_start_stop_times(metadata_df, "source_3", dataset)

    # Add start and stop times; timing to the metadata dataframe
    metadata_df["source1_start_time"] = source1_start_time
    metadata_df["source1_stop_time"] = source1_stop_time
    metadata_df["source1_timing"] = source1_timing
    metadata_df["source1_speech_timing"] = source1_speech_timing
    metadata_df["source2_start_time"] = source2_start_time
    metadata_df["source2_stop_time"] = source2_stop_time
    metadata_df["source2_timing"] = source2_timing
    metadata_df["source2_speech_timing"] = source2_speech_timing
    metadata_df["source3_start_time"] = source3_start_time
    metadata_df["source3_stop_time"] = source3_stop_time
    metadata_df["source3_timing"] = source3_timing
    metadata_df["source3_speech_timing"] = source3_speech_timing

    # Remove unaligned data
    metadata_df = remove_unaligned_3spk(metadata_df, unaligned_df)

    # Reset the index
    metadata_df = metadata_df.reset_index(drop=True)

    return metadata_df


# Part 2: Intersection Functions (stated above)

# Part 3: Generation of OSD Labels (Obtain overlaps)


def create_source1_audio_list(dataset: pd.DataFrame) -> List:
    """
    Creates a list of ranges for source1 audio based on the given DataFrame.

    Args:
        dataset (pandas.DataFrame): The DataFrame containing the source1 start and stop
        time.

    Returns:
        list: A list of tuples containing the ranges for each source1 audio.
    """
    source1_ranges_list = []

    # Iterate over each row in the DataFrame
    for length in range(len(dataset)):
        temp_speaking_ranges = []

        # Iterate over each start and stop time in the current row
        for num in range(len((dataset["source1_start_time"][length]))):
            start_time = dataset["source1_start_time"][length][num]
            stop_time = dataset["source1_stop_time"][length][num]
            start_stop_range = [start_time, stop_time]
            temp_speaking_ranges.append(start_stop_range)

        source1_ranges_list.append(temp_speaking_ranges)

    return source1_ranges_list


def create_source2_audio_list(dataset: pd.DataFrame) -> List:
    """
    Creates a list of ranges for source1 audio based on the given DataFrame.

    Args:
        dataset (pandas.DataFrame): The DataFrame containing the source1 start and stop
        time.

    Returns:
        list: A list of tuples containing the ranges for each source1 audio.
    """
    source2_ranges_list = []

    # Iterate over each row in the DataFrame
    for length in range(len(dataset)):
        temp_speaking_ranges = []

        # Iterate over each start and stop time in the current row
        for num in range(len((dataset["source2_start_time"][length]))):
            start_time = dataset["source2_start_time"][length][num]
            stop_time = dataset["source2_stop_time"][length][num]
            start_stop_range = [start_time, stop_time]
            temp_speaking_ranges.append(start_stop_range)

        source2_ranges_list.append(temp_speaking_ranges)

    return source2_ranges_list


def create_source3_audio_list(dataset: pd.DataFrame) -> list:
    """
    Creates a list of ranges for source1 audio based on the given DataFrame.

    Args:
        dataset (pandas.DataFrame): The DataFrame containing the source1 start and stop
        time.

    Returns:
        list: A list of tuples containing the ranges for each source1 audio.
    """
    source3_ranges_list = []

    # Iterate over each row in the DataFrame
    for length in range(len(dataset)):
        temp_speaking_ranges = []

        # Iterate over each start and stop time in the current row
        for num in range(len((dataset["source3_start_time"][length]))):
            start_time = dataset["source3_start_time"][length][num]
            stop_time = dataset["source3_stop_time"][length][num]
            start_stop_range = [start_time, stop_time]
            temp_speaking_ranges.append(start_stop_range)

        source3_ranges_list.append(temp_speaking_ranges)

    return source3_ranges_list


def generation_label_3mix(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Generates label information for a 3 speaker mix dataset, identifying non-speech,
    1 speaker and 2, 3 speaker overlaps in each audio clip.

    Args:
        dataset: A Pandas DataFrame containing information about the 3-mix audio clips,
        including:
            * 'mixture_ID': Unique identifier for the audio clip.
            * 'audio_total_duration': Total duration of the audio clip in seconds.
            * Additional information about the source audio and labels might be present.

    Returns:
        A Pandas DataFrame with additional columns containing labels for each
        audio clip:
            * 'audio_id': Same as 'mixture_ID' from the input DataFrame.
            * 'source1_ranges_list', 'source2_ranges_list', 'source3_ranges_list':
            Lists of speech ranges for each source in the mix.
            * 'total_time': Duration of the audio clip in seconds.
            * 'speech_times': List of combined speech ranges for all sources
            (no overlaps).
            * 'non_speech': List of silence ranges in the audio clip.
            * 'one_spk', 'two_spk', 'three_spk': Lists of speech ranges for 1, 2, and 3
            speakers, respectively.
            * 'two_or_three_spk': Combined list of speech ranges for 2 or 3 speakers
            (not mutually exclusive with 'one_spk').

    Raises:
        KeyError: If the required columns are not present in the input DataFrame.

    """

    # audio_id = []
    speech = []
    non_speech = []
    one_spk = []
    two_spk = []
    two_or_three_spk = []
    three_spk = []
    label_df = pd.DataFrame()

    # Create lists of ranges for each source
    source1_ranges_list = create_source1_audio_list(dataset)
    source2_ranges_list = create_source2_audio_list(dataset)
    source3_ranges_list = create_source3_audio_list(dataset)

    list_of_all_sources_ranges_flattened = []

    # Duration of each audio clip
    duration_timelines = []
    durations = []

    for indx in range(len(dataset)):
        # create list of all speech ranges for all sources (flattened)
        list_of_sources = []
        list_of_sources = (
            source1_ranges_list[indx]
            + source2_ranges_list[indx]
            + source3_ranges_list[indx]
        )
        list_of_all_sources_ranges_flattened.append(list_of_sources)

        speech_times = combine_overlaps(list_of_sources)
        speech.append(speech_times)

        # Get audio duration timing
        duration = dataset["audio_total_duration"][indx]
        duration_timelines.append([[0, duration]])
        durations.append([0, duration])

    # Get overlap timing, as well as silence timing

    any_intersect_compiled = []
    all_intersect_compiled = []

    for indx in range(len(dataset)):
        any_intersect = []
        all_intersect = []

        for intersection in get_intersect_of_two_list(
            source1_ranges_list[indx], source2_ranges_list[indx]
        ):
            any_intersect.append(intersection)
        for intersection in get_intersect_of_two_list(
            source2_ranges_list[indx], source3_ranges_list[indx]
        ):
            any_intersect.append(intersection)
        for intersection in get_intersect_of_two_list(
            source1_ranges_list[indx], source3_ranges_list[indx]
        ):
            any_intersect.append(intersection)

        for intersection in get_intersect_of_multiple_range(
            [
                source1_ranges_list[indx],
                source2_ranges_list[indx],
                source3_ranges_list[indx],
            ]
        ):
            all_intersect.append(intersection)

        any_intersect_compiled.append(any_intersect)
        all_intersect_compiled.append(all_intersect)

        two_or_three = combine_overlaps(any_intersect)
        two_or_three_spk.append(two_or_three)

        three_spk.append(all_intersect_compiled[indx])

        # Silences
        silences = invert_timings(speech[indx], duration_timelines[indx])

        # Drop duplicate timings in silences
        silences_no_dups = [sublist for sublist in silences if sublist[0] != sublist[1]]
        non_speech.append(list(silences_no_dups))

        overlap_or_no_speech = two_or_three + silences
        sorted_overlap_or_no_speech = sorted(overlap_or_no_speech, key=lambda x: x[0])

        # Obtain one spk and 2 speaker timings
        one_spk_element = invert_timings(
            sorted_overlap_or_no_speech, duration_timelines[indx]
        )
        two_spk_element = invert_timings(all_intersect, two_or_three)

        # Remove duplicate timings (e.g. [4.2, 4.2])
        one_spk_element_no_dups = [
            sublist for sublist in one_spk_element if sublist[0] != sublist[1]
        ]
        two_spk_element_no_dups = [
            sublist for sublist in two_spk_element if sublist[0] != sublist[1]
        ]

        one_spk.append(one_spk_element_no_dups)
        two_spk.append(two_spk_element_no_dups)

    label_df["audio_id"] = dataset["mixture_ID"]
    label_df["source1_ranges_list"] = source1_ranges_list
    label_df["source2_ranges_list"] = source2_ranges_list
    label_df["source3_ranges_list"] = source3_ranges_list
    label_df["total_time"] = durations
    label_df["speech_times"] = speech
    label_df["non_speech"] = non_speech
    label_df["one_spk"] = one_spk
    label_df["two_spk"] = two_spk
    label_df["three_spk"] = three_spk
    label_df["two_or_three_spk"] = two_or_three_spk

    return label_df


def generation_label_2mix(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Generates label information for a 2 speakers mix dataset, identifying 1 spk speech,
    overlapping speech, and silences in each audio clip.

    Args:
        dataset: A Pandas DataFrame containing information about the 2-mix audio clips,
        including:
            * 'mixture_ID': Unique identifier for the audio clip.
            * 'audio_total_duration': Total duration of the audio clip in seconds.
            * Additional information about the source audio and labels might be present.

    Returns:
        A Pandas DataFrame with additional columns containing labels for each audio
        clip:
            * 'audio_id': Same as 'mixture_ID' from the input DataFrame.
            * 'source1_ranges_list', 'source2_ranges_list': Lists of speech ranges for
            each source in the mix.
            * 'total_time': Duration of the audio clip in seconds.
            * 'speech_times': List of combined speech ranges for both sources
            (no overlaps).
            * 'non_speech': List of silence ranges in the audio clip.
            * 'one_spk': List of ranges where only one speaker is active.
            * 'overlaps': List of ranges where both speakers are active
            (potentially overlapped speech).

    Raises:
        KeyError: If the required columns are not present in the input DataFrame.

    """

    speech = []
    non_speech = []
    one_spk = []
    label_df = pd.DataFrame()

    # Create lists of ranges for each source
    source1_ranges_list = create_source1_audio_list(dataset)
    source2_ranges_list = create_source2_audio_list(dataset)

    list_of_all_sources_ranges_flattened = []

    # Duration of each audio clip
    duration_timelines = []
    durations = []

    for indx in range(len(dataset)):
        # create list of all speech ranges for all sources (flattened)
        list_of_sources = []
        list_of_sources = source1_ranges_list[indx] + source2_ranges_list[indx]
        list_of_all_sources_ranges_flattened.append(list_of_sources)

        speech_times = combine_overlaps(list_of_sources)
        speech.append(speech_times)

        # Get audio duration timing
        duration = dataset["audio_total_duration"][indx]
        duration_timelines.append([[0, duration]])
        durations.append([0, duration])

    # Get overlap timing, as well as silence timing
    any_intersect_compiled = []

    for indx in range(len(dataset)):
        any_intersect = []

        for intersection in get_intersect_of_two_list(
            source1_ranges_list[indx], source2_ranges_list[indx]
        ):
            any_intersect.append(intersection)

        any_intersect_compiled.append(any_intersect)

        # Silences: invert timings - remove speech timing from total audio duration to
        # get silences
        silences = invert_timings(speech[indx], duration_timelines[indx])

        # Drop duplicate timings in silences
        silences_no_dups = [sublist for sublist in silences if sublist[0] != sublist[1]]
        non_speech.append(list(silences_no_dups))

        # one_spk
        overlap_or_no_speech = any_intersect + silences
        sorted_overlap_or_no_speech = sorted(overlap_or_no_speech, key=lambda x: x[0])

        # Overlap or no speech
        one_spk_element = invert_timings(
            sorted_overlap_or_no_speech, duration_timelines[indx]
        )

        # Remove duplicate timings (e.g. [4.2, 4.2])
        one_spk_element_no_dups = [
            sublist for sublist in one_spk_element if sublist[0] != sublist[1]
        ]
        one_spk.append(one_spk_element_no_dups)

    label_df["audio_id"] = dataset["mixture_ID"]
    label_df["source1_ranges_list"] = source1_ranges_list
    label_df["source2_ranges_list"] = source2_ranges_list
    label_df["total_time"] = durations
    label_df["speech_times"] = speech
    label_df["non_speech"] = non_speech
    label_df["one_spk"] = one_spk
    label_df["overlaps"] = any_intersect_compiled

    return label_df


def libri2mix_generate_labels_df(
    unaligned_df: pd.DataFrame,
    alignment_dir: str,
    two_mix_metadata: str,
    sampling_rate: float,
    batchsize: int,
    save_data_mixture_name_2mix: str,
) -> Dict:
    """
    Generate labels for Libri2Mix, update metadata, and returns dataframes
    (e.g. Parquet files).

    Args:
    - unaligned_df (pd.DataFrame): DataFrame containing unaligned data.
    - alignment_dir (str): pathway to the directory containing alignment documents.
    - two_mix_metadata (str): pathway to DataFrame containing metadata for two mix.
    - sampling_rate (float): Sampling rate in Hz.
    - batchsize (int): Size of each batch of data to be saved.
    - save_data_mixture_name_2mix (str): Name for the saved file.

    This function performs the following steps:
    1. Loads necessary data and metadata.
    2. Processes and updates metadata for Libri2Mix.
    3. Generates labels for 2mix scenarios.

    # Update metadata for 2 speakers
     df_2mix_osd = update_metadata_2spk(
        duration_2mix_metadata, alignment_df, unaligned_df
    )

    Returns:
    partitions (dict): A dictionary containing partitions of the generated labels
    dataframe.
    Each partition is stored as a DataFrame with keys indicating the part number.

    """

    # Log messages to the console. You can also configure it to log to a file.
    # logging.basicConfig(
    #     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    # logger = logging.getLogger(__name__)

    alignment_df = iterate_files_produce_df(alignment_dir)

    two_mix_metadata = pd.read_csv(two_mix_metadata)

    duration_2mix_metadata = get_duration(
        two_mix_metadata, sampling_rate
    )  # sampling rate = 16000

    # UPDATE METADATA

    # Update metadata for 2 speakers
    df_2mix_osd = update_metadata_2spk(
        duration_2mix_metadata, alignment_df, unaligned_df
    )

    # Generation label 2mix
    df_2mix_osd_labels = generation_label_2mix(df_2mix_osd)

    ## Save as partitions

    partitions = {}
    # Save DataFrame in batches

    batch_size = batchsize  # 100, 800 or 1000
    num_batches = len(df_2mix_osd_labels) // batch_size + 1

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df_2mix_osd_labels))

        batch_df = df_2mix_osd_labels.iloc[start_idx:end_idx]

        batch_df.reset_index(drop=True, inplace=True)

        # Append part number to the file name
        part_number = batch_num + 1

        # Construct a unique key for each partition
        partition_key = f"{save_data_mixture_name_2mix}_part{part_number}_2mix_osd_labels.parquet.gzip"
        partitions[partition_key] = batch_df

    return partitions


# LIBRI3MIX


def libri3mix_generate_labels_df(
    unaligned_df: pd.DataFrame,
    alignment_dir: str,
    three_mix_metadata: str,
    sampling_rate: float,
    batchsize: int,
    save_data_mixture_name_3mix: str,
) -> Dict:
    """
    Generate labels for Libri3Mix, update metadata, and save dataframes into
    Parquet files.

    Args:
    - unaligned_df (pd.DataFrame): DataFrame containing unaligned data.
    - alignment_dir (str): Path to the directory containing alignment documents.
    - three_mix_metadata (str): pathway to dataframe containing metadata for Libri3mix
    (dev/train-100/test).
    - sampling_rate (float): Sampling rate in Hz.
    - batchsize (int): Size of each batch of data to be saved.
    - save_data_mixture_name_3mix (str): Name for the saved file.

    This function performs the following steps:
    1. Loads necessary data and metadata.
    2. Processes and updates metadata for Libri3Mix.
    3. Generates labels for 3mix scenarios.
    4. Returns the resulting dataframe (saved as Parquet file).

    Returns:
    partitions (dict): A dictionary containing partitions of the generated labels
    dataframe. Each partition is stored as a DataFrame with keys indicating the
    part number.

    """
    alignment_df = iterate_files_produce_df(alignment_dir)

    three_mix_metadata = pd.read_csv(three_mix_metadata)

    duration_3mix_metadata = get_duration(
        three_mix_metadata, sampling_rate
    )  # sampling rate = 16000

    # Update metadata for 3 speakers
    df_3mix_osd = update_metadata_3spk(
        duration_3mix_metadata, alignment_df, unaligned_df
    )

    ## GENERATION OF LABELS
    # Generation label 3mix
    df_3mix_osd_labels = generation_label_3mix(df_3mix_osd)

    df_3mix_osd_labels.reset_index(drop=True, inplace=True)

    ## Save as partitions

    partitions = {}
    # Save DataFrame in batches

    batch_size = batchsize  # 100, 800 or 1000
    num_batches = len(df_3mix_osd_labels) // batch_size + 1

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df_3mix_osd_labels))

        batch_df = df_3mix_osd_labels.iloc[start_idx:end_idx]

        batch_df.reset_index(drop=True, inplace=True)

        # Append part number to the file name
        part_number = batch_num + 1

        # Construct a unique key for each partition
        partition_key = f"{save_data_mixture_name_3mix}_part{part_number}_3mix_osd_labels.parquet.gzip"
        partitions[partition_key] = batch_df

    return partitions
