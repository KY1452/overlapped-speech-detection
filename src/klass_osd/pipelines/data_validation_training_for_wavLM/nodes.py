"""
This is a boilerplate pipeline 'data_validation_training_for_wavLM'
generated using Kedro 0.18.14
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)


def check_csv_exist(file_path: str) -> None:
    """
    Checks if a CSV file exists at the given file path and attempts to read it.

    Parameters:
    - file_path (str): The file path to the CSV file.

    Returns:
    - Tuple[bool, Optional[pd.DataFrame]]: A tuple where the first element is a
    boolean indicating whether the file was successfully read, and the second
    element is the DataFrame read from the file or None if the file does not exist,
    is empty, could not be parsed, or an error occurred.
    """
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            # Attempt to read the CSV file
            dataset = pd.read_csv(file_path)
            return dataset
        except pd.errors.EmptyDataError:
            logger.error("The file at %s is empty.", file_path)
        except pd.errors.ParserError:
            logger.error("The file at %s could not be parsed.", file_path)
        except Exception as exception:
            # Log any other exception that might occur
            logger.error(
                "An error occurred while reading the file at %s: %s",
                file_path,
                str(exception),
            )
    else:
        logger.error("The file path %s does not exist.", file_path)


def create_wav_metadata(
    hashing_df: pd.DataFrame, audiofile_training_dir: str, split_type: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create metadata for WAV files from the specified audio ID pathway and training
    audio file directory.

    Args:
    - hashing_df(pd.DataFrame): hashing document containing specified audio id
    for training
    - audiofile_training_dir (str): the directory for training audio files
    - split_type: (str): type of dataset used (e.g. training, evaluation, test)

    Returns:
    - metadata_df (pandas DataFrame): metadata for the matched WAV files
    - file_not_found (list): files not found in the specified folder

    """

    train_audio_id = list(hashing_df[hashing_df["split"] == split_type]["audio_id"])

    # Initialize a list to store metadata
    metadata_list = []
    file_not_found = []

    # Iterate through each file name in the list
    for file_name in train_audio_id:
        # Construct the full file path assuming the files are in '.wav' format
        file_path = os.path.join(audiofile_training_dir, f"{file_name}.wav")

        # Check if the file exists to avoid errors
        if os.path.exists(file_path):
            # Extract metadata using soundfile
            info = sf.info(file_path)

            # Append the metadata to the list
            metadata_list.append(
                {
                    "file_name": file_name,
                    "duration": info.duration,
                    "sampling_rate": info.samplerate,
                    "file_type": info.format,
                    "channels": info.channels,
                }
            )
        else:
            logger.warning("File %s does not exist in the specified folder.", file_name)
            file_not_found.append(file_name)

    # Convert the list of metadata into a pandas DataFrame
    metadata_df = pd.DataFrame(metadata_list)

    return metadata_df, file_not_found


# Check metadata to see if .wavfile is of the right properties
def check_metadata(
    metadata_df: pd.DataFrame, max_end_duration: float, target_sampling_rate: int
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Check the metadata for each audio file and identify any issues with the duration,
    file type, sampling rate, and number of channels.

    Args:
    - metadata_df (pd.DataFrame): A pandas DataFrame containing the metadata for
    the audio files.
    - max_end_duration (float): Maximum duration expected of an audio file
    - target_sampling_rate (int): Sampling rate (target)

    Returns:
    - incorrect_duration (list): A list of filenames with inappropriate durations.
    - incorrect_filetype (list): A list of filenames with incorrect file types.
    - incorrect_sampling_rate (list): A list of filenames with incorrect sampling rates.
    - incorrect_channels (list): A list of filenames with incorrect number of channels.
    """
    incorrect_duration = []
    incorrect_filetype = []
    incorrect_sampling_rate = []
    incorrect_channels = []

    for _, row in metadata_df.iterrows():
        # Check if duration is less than or equal to 0.00 and more than max end
        # duration as you've set
        if row["duration"] <= 0.00 and row["duration"] > max_end_duration:
            # Append filename to list
            incorrect_duration.append(row["file_name"])
        # Check for correct file extension
        if row["file_type"] != "WAV":
            incorrect_filetype.append(row["file_name"] + "." + row["file_type"])
        # Check if sampling rate = 16k
        if row["sampling_rate"] != target_sampling_rate:
            incorrect_sampling_rate.append(row["file_name"])
        # Check if number of channels is not 1
        if row["channels"] != 1:
            incorrect_channels.append(row["file_name"])

    return (
        incorrect_duration,
        incorrect_filetype,
        incorrect_sampling_rate,
        incorrect_channels,
    )


def resample_audio_extract_err_rate(
    audiofile_training_dir: str,
    incorrect_sampling_rate: List[str],
    target_sampling_rate: int,
) -> List[str]:
    """
    Extracts and resamples .wav files with incorrect sampling rate to the desired
    target sampling rate.

    Parameters:
        - audiofile_training_dir (str): The directory where the .wav files are located.
        - incorrect_sampling_rate (list): List of filenames with incorrect
        sampling rates.
        - target_sampling_rate (int): Sampling rate (target)


    Returns:
        files_to_be_removed (list): List of filenames that were not successfully
        resampled.
    """

    # Define the directory where the .wav files are located
    files_to_be_removed = []

    # Desired sampling rate
    target_sr = target_sampling_rate

    # Loop through each .wav file
    for filename in incorrect_sampling_rate:
        # Load the audio file
        new_filename = os.path.join(audiofile_training_dir, filename + ".wav")

        try:
            audio_signal, sampling_r = librosa.load(new_filename, sr=None)

            # Resample the audio to the target sampling rate
            y_resampled = librosa.resample(audio_signal, sampling_r, target_sr)

            # Save the resampled audio
            sf.write(new_filename, y_resampled, target_sr)

            logging.info("%s resampled and saved", filename)

        except librosa.util.exceptions.ParameterError as exception:
            files_to_be_removed.append(filename)
            logging.error(
                "Error resampling %s: %s. Sampling rate cannot be converted",
                filename,
                exception,
            )

        except FileNotFoundError as exception:
            files_to_be_removed.append(filename)
            logging.error(
                "%s not found: %s. Sampling rate cannot be converted.",
                filename,
                exception,
            )

        except PermissionError as exception:
            files_to_be_removed.append(filename)
            logging.error(
                "Permission error accessing %s: %s. Sampling rate cannot be converted.",
                filename,
                exception,
            )

        except IOError as exception:
            files_to_be_removed.append(filename)
            logging.error(
                "Error accessing %s: %s. Sampling rate cannot be converted.",
                filename,
                exception,
            )

        except Exception as exception:
            files_to_be_removed.append(filename)
            logging.error(
                "An unexpected error occurred for %s: %s. Sampling rate cannot be converted.",
                filename,
                exception,
            )

    return files_to_be_removed


def sampling_rate_failed_fileids(
    audiofile_training_dir: str,
    incorrect_sampling_rate: List[str],
    target_sampling_rate: int,
) -> List:
    """
    Identifies audio files that failed resampling due to incorrect sampling rates.

    Parameters:
    - audiofile_training_dir (str): Directory containing the training audio files.
    - incorrect_sampling_rate (List[str]): Filenames with incorrect sampling rates.
    - target_sampling_rate (int): Desired sampling rate for audio files.

    Returns:
    unable_convert_sampling_rate (List[str]): List of filenames that could not be
    resampled to the correct sampling rate.

    This function compiles a comprehensive list of files with incorrect
    sampling rates that could not be converted to correct sampling rates
    """

    unable_convert_sampling_rate = resample_audio_extract_err_rate(
        audiofile_training_dir, incorrect_sampling_rate, target_sampling_rate
    )

    return unable_convert_sampling_rate


def create_validation_check_report(
    unable_convert_sampling_rate: List[str],
    incorrect_duration: List[str],
    incorrect_filetype: List[str],
    incorrect_channels: List[str],
    file_not_found: List[str],
) -> List[str]:
    """
    Generates a validation check report summarizing issues identified in audio
    file processing and annotations.

    Parameters:
    - unable_convert_sampling_rate (List[str]): Filenames that could not be
    resampled to the correct sampling rate.
    - incorrect_duration (List[str]): Filenames with incorrect durations.
    - incorrect_filetype (List[str]): Filenames with incorrect file types.
    - incorrect_channels (List[str]): Filenames with an incorrect number of
    audio channels.
    - file_not_found (List[str]): Filenames that were not found during processing.

    Returns:
    - json_contents (List[str]): A list containing a single JSON string that
    summarizes all validation checks.

    This report provides a comprehensive overview of potential issues found
    during the validation checks, including problems with duration, file type,
    audio channels, and sampling rate conversion, making it easier
    to understand and address the identified issues.
    """

    validation_check_report = {
        "unable_convert_samplingrate": unable_convert_sampling_rate,
        "incorrect_duration": incorrect_duration,
        "incorrect_filetype": incorrect_filetype,
        "incorrect_channels": incorrect_channels,
        "file_not_found": file_not_found,
    }

    json_contents = json.dumps(validation_check_report)

    return [json_contents]


def create_combined_audioid_to_remove(
    unable_convert_sampling_rate: List[str],
    incorrect_duration: List[str],
    incorrect_filetype: List[str],
    incorrect_channels: List[str],
    file_not_found: List[str],
) -> List[str]:
    """
    Combines lists of audio IDs flagged for various issues into a single list of
    unique audio IDs to remove.

    Parameters:
    - unable_convert_sampling_rate (List[str]): List of audio IDs that could not
    have their sampling rate correctly converted.
    - incorrect_duration (List[str]): List of audio IDs with incorrect durations.
    - incorrect_filetype (List[str]): List of audio IDs with incorrect file types.
    - incorrect_channels (List[str]): List of audio IDs with incorrect numbers of
    channels.

    Returns:
    - combined_list_audioid_to_remove (List[str]): A list of unique audio IDs that
    have been flagged for removal based on the provided criteria.

    This function consolidates various error flags related to audio file processing
    and annotation into a single, deduplicated list of audio IDs for removal
    or further investigation.
    """
    failed_fileids = (
        unable_convert_sampling_rate
        + incorrect_duration
        + incorrect_filetype
        + incorrect_channels
        + file_not_found
    )
    combined_list_audioid_to_remove = list(set(failed_fileids))

    return combined_list_audioid_to_remove


def revise_hashing_df(
    hashing_df: pd.DataFrame,
    combined_list_audioid_to_remove: List[str],
) -> pd.DataFrame:
    """
    Removes entries from a hashing dataframe based on a list of audio IDs to be removed.

    This function loads a dataframe from a specified csv file. It then filters out the
    rows where the 'audio_id' is present in the provided list of audio IDs to
    be removed. The resulting dataframe, which excludes the specified audio IDs,
    is then returned.

    Parameters:
    hashing_df (pd.DataFrame): The file path to the parquet file containing the
    hashing dataframe.
    combined_list_audioid_to_remove (List[str]): A list of audio IDs to be
    removed from the dataframe.

    Returns:
    pd.DataFrame: A dataframe with the specified audio IDs removed.
    """
    hashing_df = hashing_df[
        ~hashing_df["audio_id"].isin(combined_list_audioid_to_remove)
    ]

    return hashing_df


def validation_check(
    hashing_df: pd.DataFrame,
    split_type: str,
    audiofile_training_dir: str,
    max_end_duration: float,
    target_sampling_rate: int,
) -> Tuple[List[str], List[str]]:
    """
    Perform validation checks on audio files. Return validation report and list of
    erraneous audio ids.

    This function performs several validation checks and audio properties
    verification, to ensure that the audio files meet specified criteria. It identifies
    audio files with issues such as incorrect sampling rates, inappropriate durations,
    incorrect file types, and incorrect number of channels. Based on the validation
    results, it then revises the input dataset (hashing dataset) by removing
    entries related to problematic audio files.

    Parameters:
        hashing_df (pd.DataFrame): hashing dataframe (containing audio IDs).
        split_type (str): type of dataset used (e.g. train, evaluation, test).
        audiofile_training_dir (str): The directory path where the training audio
        files are stored.
        max_end_duration (float): The upper limit for the duration of an audio file
        in seconds. Audio files exceeding this duration will be considered invalid.
        target_sampling_rate (int): The desired sampling rate for the audio files in
        Hertz. Audio files that do not match this sampling rate will be
        flagged as invalid.


    Returns:
        Tuple[List[str], List[str]]:
            - combined_list_audioid_to_remove (list): list of audio ids to remove.
            - json_contents (List[str]): A list containing a single JSON string with
            the validation check report.
    """

    metadata_df, file_not_found = create_wav_metadata(
        hashing_df, audiofile_training_dir, split_type
    )

    (
        incorrect_duration,
        incorrect_filetype,
        incorrect_sampling_rate,
        incorrect_channels,
    ) = check_metadata(metadata_df, max_end_duration, target_sampling_rate)

    unable_convert_sampling_rate = sampling_rate_failed_fileids(
        audiofile_training_dir, incorrect_sampling_rate, target_sampling_rate
    )

    combined_list_audioid_to_remove = create_combined_audioid_to_remove(
        unable_convert_sampling_rate,
        incorrect_duration,
        incorrect_filetype,
        incorrect_channels,
        file_not_found,
    )

    [json_contents] = create_validation_check_report(
        unable_convert_sampling_rate,
        incorrect_duration,
        incorrect_filetype,
        incorrect_channels,
        file_not_found,
    )

    return (combined_list_audioid_to_remove, [json_contents])


def combined_validation_check_return_revised_dataset(
    hashing_df: pd.DataFrame,
    train_split_type: str,
    train_audiofile_training_dir: str,
    eval_split_type: str,
    eval_audiofile_training_dir: str,
    max_end_duration: float,
    target_sampling_rate: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Perform validation checks on audio files and return validation report and list of
    erraneous audio ids.

    Combines list of erraneous audio ids and revises hashing documents to exclude
    erraneous audio ids.

    Parameters:
        - hashing_df (pd.DataFrame): Hashing dataframe (containing audio IDs).
        - train_split_type (str): type of dataset used (e.g. training).
        - train_audiofile_training_dir (str): The directory path where the training
        audio files are stored.
        - eval_split_type (str): type of dataset used (e.g. evaluation).
        - eval_audiofile_training_dir (str): The directory path where the training audio
        files are stored.
        - max_end_duration (float): The upper limit for the duration of an audio file in
        seconds. Audio files exceeding this duration will be considered invalid.
        - target_sampling_rate (int): The desired sampling rate for the audio files in
        Hertz. Audio files that do not match this sampling rate will be flagged
        as invalid.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]:
            - hashing_df (pd.DataFrame): The hashing dataset after validation
            and cleanup.
            - train_json_contents (List[str]): A list containing a single JSON string
            with the validation check report for training dataset.
            - eval_json_contents (List[str]): A list containing a single JSON string
            with the validation check report for evaluation dataset.
    """

    eval_create_combined_audioid_to_remove, [eval_json_contents] = validation_check(
        hashing_df,
        eval_split_type,
        eval_audiofile_training_dir,
        max_end_duration,
        target_sampling_rate,
    )

    train_create_combined_audioid_to_remove, [train_json_contents] = validation_check(
        hashing_df,
        train_split_type,
        train_audiofile_training_dir,
        max_end_duration,
        target_sampling_rate,
    )

    combined_list_audioid_to_remove = (
        eval_create_combined_audioid_to_remove + train_create_combined_audioid_to_remove
    )

    revised_hashing_df = revise_hashing_df(hashing_df, combined_list_audioid_to_remove)

    return revised_hashing_df, [train_json_contents], [eval_json_contents]


def get_histogram_plt(
    new_hashing_df: pd.DataFrame, no_of_spk: int, split_type: str
) -> None:
    """
    Generates histograms for SNR values across different numbers of speakers
    from a specified dataset.

    Parameters:
    - new_hashing_df (pd.DataFrame): New revised dataframe. This document is expected
    to contain audio IDs for training along with their respective SNR values for
    different numbers of speakers.
    - no_of_spk (int): The number of speakers to include in the histograms.
      This determines the number of subplots generated.
    - split_type (str): The type of dataset used
    (e.g., 'training', 'evaluation', 'test').
      This filters the data to only include entries corresponding to the specified
      dataset type.

    Returns:
    - None: The function directly saves the generated histograms as image files and
    does not return any value.


    Note:
    - The function assumes the dataframe contains a 'split' column to filter
    by split_type, and columns named 'one_spk_SNR', 'two_spk_SNR', and 'three_spk_SNR'
    representing the SNR values for one, two, and three speakers, respectively.
    """

    if no_of_spk == 3:
        plots = ["one_spk_SNR", "two_spk_SNR", "three_spk_SNR"]
    elif no_of_spk == 2:
        plots = ["one_spk_SNR", "two_spk_SNR"]

    _, axes = plt.subplots(nrows=1, ncols=no_of_spk, figsize=(15, 4))

    # Read dataframe
    dataframe = new_hashing_df
    dataframe = dataframe[dataframe["split"] == split_type]

    for i in range(no_of_spk):
        # Plot Histogram of SNR
        axes[i].hist(dataframe[plots[i]], bins=20, edgecolor="black")
        axes[i].set_title(f"{split_type} histogram")
        axes[i].set_xlabel(plots[i])
        axes[i].set_ylabel("frequency")
        show_point = dataframe[plots[i]].mean()
        # Annotate the average SNR
        axes[i].annotate(
            f"Average SNR: {show_point:.2f}",
            xy=(show_point, 0),
            xytext=(show_point, axes[i].get_ylim()[1] / 2),
            arrowprops=dict(facecolor="black", shrink=0.05),
            ha="right",
        )

    return plt


def get_gender_chart(
    new_hashing_df: pd.DataFrame, no_of_spk: int, split_type: str
) -> None:
    """
    Generates barchart of gender combination of audio files in the hashing dataset.

    Parameters:
    - new_hashing_df (pd.DataFrame): New revised dataframe. This document is expected
    to contain audio IDs for training along with their respective gender combi values.
    - no_of_spk (int): The number of speakers.
    - split_type (str): The type of dataset used
    (e.g., 'training', 'evaluation', 'test').
    This filters the data to only include entries corresponding to the specified
    dataset type.

    Returns:
    - None: The function directly saves the generated barchart as image files and
    does not return any value.


    Note:
    - The function assumes the dataframe contains a 'split' column to filter by
    split_type, and columns named 'gender_combi'.
    """

    # Read dataframe
    dataframe = new_hashing_df
    dataframe = dataframe[dataframe["split"] == split_type]

    replacement_dict = {
        "FM": "1M1F",
        "MF": "1M1F",
        "FFM": "1M2F",
        "MFF": "1M2F",
        "FMF": "1M2F",
        "MMF": "2M1F",
        "FMM": "2M1F",
        "MFM": "2M1F",
    }

    # Apply the replacements
    dataframe["gender_combi"] = dataframe["gender_combi"].replace(replacement_dict)

    # Calculating the count for each category
    gender_counts = dataframe.groupby("gender_combi").size()

    # Calculating the percentage for each category
    gender_percentage = (gender_counts / gender_counts.sum()) * 100

    # Creating the bar chart
    gender_chart_ax = gender_percentage.plot(kind="bar")
    plt.xlabel("Gender Combination")
    plt.ylabel("Percentage")
    plt.title(f"Percentage Breakdown by Gender Combination ({split_type})")
    plt.xticks(rotation=45)

    # Adding annotations on top of each bar
    for patch in gender_chart_ax.patches:
        gender_chart_ax.annotate(
            f"{patch.get_height():.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    plt.tight_layout()

    return plt
