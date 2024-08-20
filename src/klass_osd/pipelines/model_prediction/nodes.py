# """
# This is a boilerplate pipeline 'model_prediction'
# generated using Kedro 0.18.14
# """

# from typing import Any, Callable, Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from datasets import Audio, Dataset
# from sklearn.metrics import classification_report, confusion_matrix
# from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# def compute_predicted_labels(
#     test_data: List[Dict[str, any]],
#     feature_extractor: Callable[[List[float], Dict[str, any]], Dict[str, torch.Tensor]],
#     model: torch.nn.Module,
#     wavlm_training_args_config: Dict[str, any],
# ) -> Tuple[pd.DataFrame, Dict[str, any]]:
#     """
#     Predicts labels for the given test data using a feature extractor and a model.

#     Parameters:
#         test_data (List[Dict[str, any]]): A list of examples containing input values.
#         feature_extractor (Callable): A function that extracts features from
#          input values.
#         model (torch.nn.Module): A pretrained model for label prediction.
#         wavlm_training_args_config (Dict[str, any]): Configuration for
#         training arguments.

#     Returns:
#         metric_df (pd.DataFrame): DataFrame containing predicted labels and
#         ground truth.
#         reports_dict (Dict[str, any): A dictionary containing classification
#         report and confusion matrix.
#     """
#     predicted_classes = []

#     for example in test_data:
#         if len(example["input_values"]) > 400:
#             inputs = feature_extractor(
#                 example["input_values"],
#                 sampling_rate=wavlm_training_args_config.get("sampling_rate"),
#                 return_tensors="pt",
#             )
#             with torch.no_grad():
#                 logits = model(**inputs).logits
#             predicted_class = torch.argmax(logits).item()
#             predicted_classes.append(predicted_class)
#         else:
#             predicted_classes.append("NA")

#     metric_df = pd.DataFrame()
#     metric_df["chunk_filenames"] = test_data["chunk_filenames"]
#     metric_df["true_label"] = test_data["label"]
#     metric_df["predicted_label"] = predicted_classes
#     metric_df = metric_df.replace("NA", np.nan)
#     metric_df = metric_df.dropna()

#     y_true = metric_df["true_label"].astype(int)
#     y_pred = metric_df["predicted_label"].astype(int)
#     classification_rep = classification_report(y_true, y_pred)

#     confusion_mat = confusion_matrix(y_true, y_pred)
#     print(confusion_mat)

#     reports_dict = {
#         "classification_report": classification_rep,
#         "confusion_matrix": confusion_mat.tolist(),
#     }

#     return metric_df, reports_dict


# def prediction_labels_and_report(
#     model_training_dict: Dict,
#     dataset_hf: Dataset,
#     wavlm_save_dir_config: Dict[str, Any],
#     wavlm_training_args_config: Dict[str, Any],
# ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#     """
#     Provide prediction report for evaluation and test datasets using a trained model.

#     Args:
#         - model_training_dict (Dict): dictionary reflecting whether model training was
#         completed successfully (True)
#         - dataset_hf (Dataset): huggingface dataset (stored in pickledataset in kedro).
#         - wavlm_save_dir_config (Dict[str, Any]): Configuration for saving directory.
#         - wavlm_training_args_config (Dict[str, Any]): Configuration for
#         training arguments.

#     Returns:
#         - predicted_label_df (pd.DataFrame): dataframe for predicted labels.
#         - wavlm_report (Dict[str, Any]: prediction report (confusion matrix).
#     """

#     if model_training_dict[0]["training_successful"]:
#         # Import the saved model
#         new_feature_extractor = AutoFeatureExtractor.from_pretrained(
#             wavlm_save_dir_config.get("model_dir")
#         )
#         new_model = AutoModelForAudioClassification.from_pretrained(
#             wavlm_save_dir_config.get("model_dir")
#         )

#         # Predict labels for dataset
#         predicted_label_df, wavlm_report = compute_predicted_labels(
#             dataset_hf,  # huggingface dataset
#             new_feature_extractor,
#             new_model,
#             wavlm_training_args_config,
#         )

#     # raise error
#     else:
#         raise ValueError("Training of model not done / model not saved yet")

#     return predicted_label_df, wavlm_report

"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.18.14
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def batchify(data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Split the data into batches of size `batch_size`.

    Parameters:
        data (List[Dict[str, Any]]): The input data.
        batch_size (int): The size of each batch.

    Returns:
        List[List[Dict[str, Any]]]: A list of batches.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def compute_predicted_labels(
    test_data: List[Dict[str, Any]],
    feature_extractor: Callable[[List[float], Dict[str, Any]], Dict[str, torch.Tensor]],
    model: torch.nn.Module,
    wavlm_training_args_config: Dict[str, Any],
    batch_size: int = 2048  # Adjust the batch size as needed
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Predicts labels for the given test data using a feature extractor and a model.

    Parameters:
        test_data (List[Dict[str, Any]]): A list of examples containing input values.
        feature_extractor (Callable): A function that extracts features from input values.
        model (torch.nn.Module): A pretrained model for label prediction.
        wavlm_training_args_config (Dict[str, Any]): Configuration for training arguments.
        batch_size (int): The size of each batch for processing.

    Returns:
        metric_df (pd.DataFrame): DataFrame containing predicted labels and ground truth.
        reports_dict (Dict[str, Any): A dictionary containing classification report and confusion matrix.
    """
    batches = batchify(test_data, batch_size)
    predicted_classes = []
    chunk_filenames = []
    true_labels = []

    for batch in batches:
        batch_inputs = [example["input_values"] for example in batch if len(example["input_values"]) > 400]
        batch_filenames = [example["chunk_filenames"] for example in batch if len(example["input_values"]) > 400]
        batch_labels = [example["label"] for example in batch if len(example["input_values"]) > 400]

        if batch_inputs:
            inputs = feature_extractor(
                batch_inputs,
                sampling_rate=wavlm_training_args_config.get("sampling_rate"),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=wavlm_training_args_config.get("max_length", 16000)
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            batch_predictions = torch.argmax(logits, dim=-1).tolist()
        else:
            batch_predictions = []

        predicted_classes.extend(batch_predictions)
        chunk_filenames.extend(batch_filenames)
        true_labels.extend(batch_labels)

    metric_df = pd.DataFrame({
        "chunk_filenames": chunk_filenames,
        "true_label": true_labels,
        "predicted_label": predicted_classes
    })

    y_true = metric_df["true_label"].astype(int)
    y_pred = metric_df["predicted_label"].astype(int)
    classification_rep = classification_report(y_true, y_pred)

    confusion_mat = confusion_matrix(y_true, y_pred)
    print(confusion_mat)

    reports_dict = {
        "classification_report": classification_rep,
        "confusion_matrix": confusion_mat.tolist(),
    }

    return metric_df, reports_dict


def prediction_labels_and_report(
    model_training_dict: Dict,
    dataset_hf: Dataset,
    wavlm_save_dir_config: Dict[str, Any],
    wavlm_training_args_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Provide prediction report for evaluation and test datasets using a trained model.

    Args:
        - model_training_dict (Dict): dictionary reflecting whether model training was
        completed successfully (True)
        - dataset_hf (Dataset): huggingface dataset (stored in pickledataset in kedro).
        - wavlm_save_dir_config (Dict[str, Any]): Configuration for saving directory.
        - wavlm_training_args_config (Dict[str, Any]): Configuration for
        training arguments.

    Returns:
        - predicted_label_df (pd.DataFrame): dataframe for predicted labels.
        - wavlm_report (Dict[str, Any]: prediction report (confusion matrix).
    """

    if model_training_dict[0]["training_successful"]:
        # Convert Hugging Face dataset to list of dictionaries
        test_data = dataset_hf.to_pandas().to_dict(orient="records")

        # Import the saved model
        new_feature_extractor = AutoFeatureExtractor.from_pretrained(
            wavlm_save_dir_config.get("model_dir")
        )
        new_model = AutoModelForAudioClassification.from_pretrained(
            wavlm_save_dir_config.get("model_dir")
        )

        # Predict labels for dataset
        predicted_label_df, wavlm_report = compute_predicted_labels(
            test_data,  # List of dictionaries
            new_feature_extractor,
            new_model,
            wavlm_training_args_config,
        )

    # raise error
    else:
        raise ValueError("Training of model not done / model not saved yet")

    return predicted_label_df, wavlm_report