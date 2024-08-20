"""
This is a boilerplate pipeline 'wavlm_hf_dataset'
generated using Kedro 0.18.14
"""
import gc
import logging
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import pyrubberband as pyrb
from audiomentations import AddGaussianNoise, AddShortNoises, ApplyImpulseResponse
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import AutoFeatureExtractor


def preprocess_function(
    examples: dict, feature_extractor: str, max_duration: float = 1.0
) -> dict:
    """
    Extract features from a list of audio files using the wav2vec2 feature extractor.

    Args:
        examples (dict): A dictionary containing the input audio examples.
        feature_extractor (str): Feature extractor to use from huggingface to encode data.
        max_duration (float, optional): The maximum duration (in seconds) to consider for each audio example. Defaults to 1.

    Returns:
        dict: A dictionary containing the preprocessed inputs for the machine learning model.
    """
    logger = logging.getLogger(__name__)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
    audio_arrays = [np.array(item) for item in examples["input_values"]]

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_tensors="np",
    )

    logger.info(f"Input dim: {inputs['input_values'].shape}")
    return inputs


def augment_hf_dataset(examples: dict, augment_config: dict) -> dict:
    """
    Function to augment audio huggingface dataset in batches.

    Args:
        examples (dict): A dictionary containing the input audio examples.
        augment_config (dict): Configuration for audio augmentation

    Returns:
        dict: A dictionary containing the augmented inputs for the machine learning model.
    """

    audio_arrays = [np.array(item) for item in examples["input_values"]]
    inputs = [
        augment_dataset(audio_signal, augment_config) for audio_signal in audio_arrays
    ]

    return {"input_values": inputs}


def get_huggingface_dataset(segments_label: dict, dataset_type: str) -> Dataset:
    """
    Generate a HuggingFace dataset from a dictionary of audio_arrays and labels of segments.

    Args:
        segments_label (dict): A dictionary containing the segmented/chunked audio arrays and labels.
        dataset_type (str): The type of the dataset ("train", "evaluation", or "test").

    Returns:
        hf_dataset (Dataset): A HuggingFace dataset.

    """
    if dataset_type in ["train", "augment_train"]:
        hf_dataset = Dataset.from_dict(
            {
                "label": segments_label["label"].astype(int),
                "input_values": segments_label["input_values"].values,
            }
        )
    if dataset_type in ["evaluation", "test", "sparse_test", "augment_eval"]:
        hf_dataset = Dataset.from_dict(
            {
                "audio_id": segments_label["audio_id"].values,
                "label": segments_label["label"].astype(int),
                "input_values": segments_label["input_values"].values,
                "chunk_filenames": segments_label["chunk_filenames"].values,
            }
        )

    return hf_dataset


def get_train_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates training dataset by segmenting and labeling the input dataframe.

    Args:
        df (pd.DataFrame): The input dataframe with four columns: audio id, non_speech_signal, one_spk_signal, two_spk_signal.
        The last three columns contain an array of arrays (representing segments) in each row.

    Returns:
        segments_label_df (pd.DataFrame): The resulting dataframe after segmenting and labeling the input dataframe.
        It contains three columns: 'audio_id' for source audio file, 'label' for the segment label and 'input_values' for the input values.
    """
    segments_label_df = pd.DataFrame(columns=["label", "input_values"])

    # Define the list of column names
    column_names = ["non_speech_signal", "one_spk_signal", "two_spk_signal"]

    for column_name in column_names:
        column_df = pd.DataFrame(df[column_name])
        column_df.rename(columns={column_name: "input_values"}, inplace=True)

        column_df = column_df.explode("input_values")
        if column_name == "non_speech_signal":
            column_df["label"] = 0
        elif column_name == "one_spk_signal":
            column_df["label"] = 1
        elif column_name == "two_spk_signal":
            column_df["label"] = 2
        column_df["audio_id"] = df["audio_id"]
        segments_label_df = pd.concat([segments_label_df, column_df], ignore_index=True)

    segments_label_df.dropna(inplace=True)

    return segments_label_df


def get_eval_test_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a dataset for evaluation and testing.

    Args:
        df: a pandas DataFrame containing the input data with 4 columns: audio_id, amp_chunks (array of arrays), chunks_class (array of arrays), chunk_filenames (array).

    Returns:
        exploded_df: a pandas DataFrame containing the exploded chunks dataset with 4 columns: audio_id, label, input_values (array), chunk_filenames.
    """
    input = df[["audio_id", "amp_chunks"]].explode("amp_chunks")
    chunk_filenames = df["chunk_filenames"].explode()
    label = df["chunks_class"].explode()
    exploded_df = pd.DataFrame(
        {
            "audio_id": input["audio_id"],
            "label": label,
            "input_values": input["amp_chunks"],
            "chunk_filenames": chunk_filenames,
        }
    ).reset_index(drop=True)
    exploded_df["label"] = exploded_df["label"].apply(lambda x: x[0])
    exploded_df["label"] = exploded_df["label"].replace(
        {"two_spk": 2, "one_spk": 1, "non_speech": 0}
    )

    return exploded_df


def augment_dataset(audio_signal: np.ndarray, augment_config: dict) -> np.ndarray:
    """
    Augment the dataset using various audio processing techniques based on the provided configuration.

    Args:
        audio_signal (np.ndarray): Input audio data.
        augment_config (Dict[str, Dict[str, Any]]): Configuration for audio augmentation containing the following keys:
            - 'time_stretch': Dictionary with 'min' and 'max' keys specifying the range for time stretching.
            - 'pitch_shift': Dictionary with 'min' and 'max' keys specifying the range for pitch shifting.
            - 'sr' (int): Sample rate.
            - 'reverbs': Dictionary with 'ir_path' and 'probability' keys for reverb augmentation.
            - 'short_noise': Dictionary with keys for short noise augmentation.
            - 'gaussian_noise': Dictionary with keys for Gaussian noise augmentation.

    Returns:
        augmented_signal (np.ndarray): Augmented audio data.
    """
    time_stretch_factor = random.uniform(
        augment_config["time_stretch"]["min"], augment_config["time_stretch"]["max"]
    )
    signal_after_time = pyrb.time_stretch(
        y=audio_signal, sr=augment_config["sr"], rate=time_stretch_factor
    )

    pitch_shift_factor = random.uniform(
        augment_config["pitch_shift"]["min"], augment_config["pitch_shift"]["max"]
    )
    signal_after_pitch = pyrb.pitch_shift(
        y=signal_after_time, sr=augment_config["sr"], n_steps=pitch_shift_factor
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        reverbs_augment = ApplyImpulseResponse(
            ir_path=augment_config["reverbs"]["ir_path"],
            p=augment_config["reverbs"]["probability"],
        )
        signal_after_reverbs = reverbs_augment(
            samples=signal_after_pitch, sample_rate=augment_config["sr"]
        )

        short_noise_augment = AddShortNoises(
            sounds_path=augment_config["short_noise"]["short_noise_path"],
            min_snr_db=augment_config["short_noise"]["min_snr"],
            max_snr_db=augment_config["short_noise"]["max_snr"],
            noise_rms=augment_config["short_noise"]["noise_rms"],
            min_time_between_sounds=augment_config["short_noise"][
                "min_time_between_sounds"
            ],
            max_time_between_sounds=augment_config["short_noise"][
                "max_time_between_sounds"
            ],
            p=augment_config["short_noise"]["probability"],
        )
        signal_after_short_noise = short_noise_augment(
            samples=signal_after_reverbs, sample_rate=augment_config["sr"]
        )

        gaussian_noise = AddGaussianNoise(
            min_amplitude=augment_config["gaussian_noise"]["min_amp"],
            max_amplitude=augment_config["gaussian_noise"]["max_amp"],
            p=augment_config["gaussian_noise"]["probability"],
        )
        augmented_signal = gaussian_noise(
            samples=signal_after_short_noise, sample_rate=augment_config["sr"]
        )

    return augmented_signal


def curate_hf_dataset(
    input: str or dict,
    output_folder_path: str,
    dataset_type: str,
    encoded: bool,
    feature_extractor: str,
    augment_config: dict = None,
) -> Dataset:
    """
    This function reads Parquet files from the input folder, processes them based on the dataset type,
    creates a Hugging Face dataset, encodes the dataset using a preprocess function, and returns the result.

    Args:
        input (str/dict): The path to the input file or a dict containing partition IDs as keys and partition load functions as values.
        output_folder_path (str): The path to the folder where the curated dataset will be saved.
        dataset_type (str): The type of the dataset ("train", "evaluation", or "test").
        encoded (bool): True for encoded, False for non-encoded.
        feature_extractor (str): Feature extractor to use from huggingface to encode data.
        augment_config (Dict[str, Dict[str, Any]]): Configuration for audio augmentation

    Returns:
        combined_hf_dataset (Dataset): The curated Hugging Face dataset.
    """

    logger = logging.getLogger(__name__)
    logger.info(augment_config)
    combined_hf_dataset = Dataset.from_dict({})

    if dataset_type == "sparse_test":
        hashed_df = input
        segment_dataset = get_eval_test_dataset(hashed_df)
        combined_hf_dataset = get_huggingface_dataset(segment_dataset, dataset_type)

    else:
        for partition_id, partition_load_func in sorted(input.items()):
            logger.info(partition_id)
            # path = os.path.join(input_folder_path, file)
            # hashed_df = pd.read_parquet(path)
            hashed_df = partition_load_func()  # load the actual partition data

            if dataset_type in ["train", "augment_train"]:
                segment_dataset = get_train_dataset(hashed_df)
            if dataset_type in ["evaluation", "test", "augment_eval"]:
                segment_dataset = get_eval_test_dataset(hashed_df)

            hf_dataset = get_huggingface_dataset(segment_dataset, dataset_type)

            if dataset_type in ["augment_train", "augment_eval"]:
                logger.info("Start augmentation")
                hf_dataset = hf_dataset.map(
                    lambda batch: augment_hf_dataset(batch, augment_config),
                    batched=True,
                    num_proc=16,
                )
                logger.info("Dataset augmented")

            combined_hf_dataset = concatenate_datasets(
                [combined_hf_dataset, hf_dataset]
            )
            del hashed_df
            del segment_dataset
            del hf_dataset
            gc.collect()

    if encoded:
        logger.info("Start encoding.")
        combined_hf_dataset = combined_hf_dataset.map(
            preprocess_function,
            batched=True,
            fn_kwargs={"feature_extractor": feature_extractor, "max_duration": 1.0},
        )
        if "attention_mask" in combined_hf_dataset.features:
            combined_hf_dataset = combined_hf_dataset.map(
                remove_columns=["attention_mask"]
            )
        logger.info("Encoding done.")

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    return combined_hf_dataset
