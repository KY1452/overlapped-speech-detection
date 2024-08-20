"""
This is a boilerplate pipeline 'data_600ms_chunks'
generated using Kedro 0.18.14
"""
import json
import math
import os

import librosa
import numpy as np
import pandas as pd


def rename_sparselibrimix_audio_files(sparselibrimix_audio_folder: str) -> json:
    """
    Rename audio files in the SparseLibriMix dataset by including the name of the sparse folder in the filename.

    Args:
        sparselibrimix_audio_folder (str): The path to the root folder of the SparseLibriMix audio dataset.
            This folder should contain the 'mix_clean' and 'mix_noisy' subfolders, which in turn contain the audio files.

    Returns:
        json: a json output with content "True" if this node runs successfully.
    """
    for dirpath, dirnames, filenames in os.walk(sparselibrimix_audio_folder):
        for filename in filenames:
            if os.path.basename(dirpath) in ["mix_clean", "mix_noisy"]:
                if filename.lower().startswith("mix") and filename.endswith((".wav")):
                    print(dirpath)
                    print(filename)
                    # Extract sparse folder
                    sparse_folder = dirpath.split("/")[-3]
                    print(sparse_folder)
                    # Create new file name
                    new_filename = f"{sparse_folder}-{filename}"
                    # Rename the file
                    os.rename(
                        os.path.join(dirpath, filename),
                        os.path.join(dirpath, new_filename),
                    )

    return "True"


def add_audio_amp(
    sparselibrimix_audio_folder: str, audio_id: str, sr_subset: str, subset: str
) -> np.ndarray:
    """
    Generate the amplitude of the audio file specified by the audio_id.

    Args:
        sparselibrimix_audio_folder (str): The path to the folder containing the
        audio files.
        audio_id (str): The ID of the audio file in the format "folder_name/file_name".
        sr_subset (str): sample rate folder path, either "wav8000" or "wav16000".
        subset (str): mix type folder path, either "mix_clean" or "mix_noisy".

    Returns:
        numpy.ndarray: The amplitude data of the audio file.
    """
    folder = audio_id.split("/")[0]
    file = folder + "-" + audio_id.split("/")[1] + ".wav"
    file_path = os.path.join(
        sparselibrimix_audio_folder, folder, sr_subset, subset, file
    )
    amp, _ = librosa.load(
        file_path, sr=None
    )  # sr = None to preserve the original sample rate
    return amp


def generate_chunk_names_per_audio(
    audio_id: str, no_of_chunks: int, chunk_duration: float
) -> list:
    """
    Generates chunk names and timings for an audio file based on the given audio ID and
    number of chunks.

    Args:
        audio_id (str): The ID of the audio file.
        no_of_chunks (int): The number of chunks to generate.
        chunk_duration (float): Duration of each chunk in seconds.

    Returns:
        names: A list of chunk names in the format "{audio_id}_{start_time}_{end_time}".
        timings: A list of chunk timing in the format "[start_time, end_time]".
    """
    names: list = []
    timings: list = []

    # Generate chunk names for each chunk
    for i in range(no_of_chunks):
        start_time: float = round(i * chunk_duration, 1)
        end_time: float = round((i + 1) * chunk_duration, 1)
        names.append(f"{audio_id}_{start_time}_{end_time}")
        timings.append([start_time, end_time])

    return names, timings


def generate_amp_chunks(
    amp: list, no_of_chunks: int, chunk_duration: float, sample_rate: int
) -> list:
    """
    Generate amp chunks based on the given amp and number of chunks.
    Args:
        amp (list): The input amp data as a list of floats
        no_of_chunks (int): The number of chunks to generate.
        sample_rate (int): Sample rate of audio files.

    Returns:
        amp_chunks (list): A list of chunk amps, each chunk represented as a list of floats.
    """
    amp_chunks = []
    chunk_length = int(chunk_duration * sample_rate)

    # Generate chunk amps for each chunk
    for i in range(no_of_chunks):
        start_index = i * chunk_length
        end_index = (i + 1) * chunk_length
        amp_chunks.append(amp[start_index:end_index])

    return amp_chunks


def generate_label_duration(timing: tuple, label: list) -> float:
    """
    Calculate the total duration of overlap between the timing and label intervals.

    Args:
        timing (tuple): The timing interval to compare with label intervals.
        label (list): List of intervals to compare with the timing interval.

    Returns:
        total_label_duration (float): Total duration of overlap between timing and label intervals.
    """
    # Initialize total overlap duration
    total_label_duration = 0.0

    # Iterate through each interval in the overlap list
    for interval in label:
        # Calculate the overlap between target_interval and the current interval
        label_start = max(float(timing[0]), float(interval[0]))
        label_end = min(float(timing[1]), float(interval[1]))

        # Check if there is any overlap
        if label_start < label_end:
            # Calculate the duration of the overlap and add to the total
            label_duration = label_end - label_start
            total_label_duration += label_duration

    return total_label_duration


def generate_chunk_labels(
    chunk_timings: list, non_speech: list, one_spk: list, two_spk: list
) -> list:
    """
    Generate labels for each chunk based on the provided timings and speech categories.

    Args:
        chunk_timings: list of chunk timings
        non_speech: non-speech duration
        one_spk: duration of speech by one speaker
        two_spk: duration of speech by two speakers

    Returns:
        labels: list of labels for each chunk
    """
    labels = []
    for each_chunk in chunk_timings:
        non_speech_duration = generate_label_duration(each_chunk, non_speech)
        one_spk_duration = generate_label_duration(each_chunk, one_spk)
        two_spk_duration = generate_label_duration(each_chunk, two_spk)

        if (
            non_speech_duration > one_spk_duration
            and non_speech_duration > two_spk_duration
        ):
            labels.append(["non_speech"])
        elif (
            one_spk_duration > non_speech_duration
            and one_spk_duration > two_spk_duration
        ):
            labels.append(["one_spk"])
        else:
            labels.append(["two_spk"])

    return labels


def generate_sparselibrimix_all_chunks(
    parquetfile: str,
    rename_done: str,
    sparselibrimix_audio_folder: str,
    sample_rate: int,
    subset: str,
    chunk_duration: float,
) -> pd.DataFrame:
    """
    Generate sparselibrimix chunks from a Parquet file and an audio folder.

    Args:
        parquetfile (str): The path to the Parquet file.
        rename_done (str): The path to the json file indicating the renaming of
        the sparselibrimix audio files are done so that reading the audio file
        will not give an error.
        sparselibrimix_audio_folder (str): The folder containing sparselibrimix
        audio files.
        sample_rate (int): sample rate.
        subset (str): mix type folder path, either "mix_clean" or "mix_noisy".
        chunk_duration (float): Duration of each chunk in seconds.

    Returns:
        pd.DataFrame: A DataFrame containing the generated sparselibrimix chunks.
    """

    # Read the Parquet file into a DataFrame
    label_df = parquetfile
    sr_subset = "wav" + str(sample_rate)

    chunk_df = pd.DataFrame()

    # Read only relevant columns
    chunk_df = label_df

    if rename_done == "True":
        chunk_df["amp"] = label_df.apply(
            lambda row: add_audio_amp(
                sparselibrimix_audio_folder, row["audio_id"], sr_subset, subset
            ),
            axis=1,
        )
    else:
        raise ValueError("Renaming of sparselibrimix not done")

    # In label_df, ["total_time"] appears like [[0.0, 6.992]]. Here we want to extract the total duration only.
    chunk_df["total_time"] = label_df["total_time"].apply(lambda x: float(x[0][1]))
    chunk_df["no_of_chunks"] = chunk_df["total_time"].apply(
        lambda x: int(math.floor(x / chunk_duration))
    )
    chunk_df[["chunk_filenames", "chunk_timings"]] = chunk_df.apply(
        lambda row: pd.Series(
            generate_chunk_names_per_audio(
                row["audio_id"], row["no_of_chunks"], chunk_duration
            )
        ),
        axis=1,
    )
    chunk_df["amp_chunks"] = chunk_df.apply(
        lambda row: generate_amp_chunks(
            row["amp"], row["no_of_chunks"], chunk_duration, sample_rate
        ),
        axis=1,
    )

    chunk_df["chunks_class"] = chunk_df.apply(
        lambda row: generate_chunk_labels(
            row["chunk_timings"], row["non_speech"], row["one_spk"], row["two_spk"]
        ),
        axis=1,
    )
    chunk_df = chunk_df[
        ["audio_id", "amp", "amp_chunks", "chunks_class", "chunk_filenames"]
    ]

    return chunk_df
