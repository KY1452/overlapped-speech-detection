import os
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import pandas as pd


def remove_repeated_numbers(input_list: List) -> List:
    """
    Remove sublists with repeated numbers in position 0 and 1.

    Parameters:
    - input_list (list): The input sublist.

    Returns:
    - filtered_list (list): A filtered list without repeated numbers.

    """

    filtered_list = [sublist for sublist in input_list if sublist[0] != sublist[1]]

    return filtered_list


def remove_filtered_cols(col_names: List, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filter away repeated numbers by remove repeated numbers (e.g., start and end time are the same) in cells for specified columns.


    Parameters:
    - col_names (List(str)): List of column names.
    - dataset (pd.DataFrame): DataFrame of metadata.

    Returns:
    - pd.DataFrame: Updated DataFrame.
    """
    for col_name in col_names:
        dataset[col_name] = dataset[col_name].apply(remove_repeated_numbers)

    return dataset


def add_audio_amp_librimix(
    audio_id: str,
    wave_type: str,
    split_type: str,
    mix_type: str,
    num_speaker_mix: str,
    librimix_audio_folder: str,
) -> Tuple[List[float], int]:
    """
    Obtain audio amplitude based on audio file.

    Parameters:
    - audio_id (str): Audio ID.
    - wave_type (str): Wave file type (e.g., 'wav8k', 'wav16k').
    - split_type (str): Split type (e.g., 'dev', 'train-100', 'test').
    - mix_type (str): Mixture type ('both', 'clean').
    - num_speaker_mix (str): Number of speakers in the mix ('Libri2Mix', 'Libri3Mix').
    - librimix_audio_folder (str): pathway to folder containing audio clips

    Returns:
    - amp (Tuple[List[float], int]): A tuple containing the audio amplitude (list of floats) and the sampling rate (int).
    """

    mixture_type = "mix_" + mix_type
    file = audio_id + ".wav"

    file_path = os.path.join(
        librimix_audio_folder,
        num_speaker_mix,
        wave_type,
        "max",
        split_type,
        mixture_type,
        file,
    )
    amp, _ = librosa.load(file_path, sr=None)

    return amp
