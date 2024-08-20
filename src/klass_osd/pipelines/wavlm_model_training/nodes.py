"""
This is a boilerplate pipeline 'wavlm_model_training'
generated using Kedro 0.18.14
"""
from typing import Any, Callable, Dict, Tuple, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def model_parameters(
    wavlm_model_parameters_config: dict, wavlm_training_args_config: dict
) -> Tuple:
    """
    Initializes a model with the given number of labels and mask time length.

    Args:
        wavlm_model_parameters_config (dict): Configuration for model parameters.
            - num_labels (int): The number of labels for the classification task.
            - mask_time_length (int): The length of the masked time in the model.
        wavlm_training_args_config (dict): Configuration for training arguments.
            - feature_extractor (str): Name or path of the pretrained model for feature
             extraction.
            - audio_classification_model (str): Name or path of the audio classification
             model.

    Returns:
        tuple: A tuple containing the initialized model, the feature extractor, and
        the accuracy.

            - model (AutoModelForAudioClassification): The initialized model for audio
            classification.
            - feature_extractor (AutoFeatureExtractor): The feature extractor for the
            model.
            - f1 (obj): F1 metrics.

    """
    num_labels = wavlm_model_parameters_config.get("num_labels")
    mask_time_length = wavlm_model_parameters_config.get("mask_time_length")

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda") if cuda_available else torch.device("cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        wavlm_training_args_config.get("feature_extractor")
    )

    f1 = evaluate.load("f1")

    model = AutoModelForAudioClassification.from_pretrained(
        wavlm_training_args_config.get("audio_classification_model"),
        num_labels=num_labels,
    ).to(device)

    model.config.mask_time_length = mask_time_length

    return model, feature_extractor, f1


def train_model(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    wavlm_save_dir_config: Dict[str, str],
    model: Any,  # Type of model
    feature_extractor: Any,  # Type of feature_extractor
    compute_metrics: Callable,
    wavlm_training_args_config: Dict[str, Any],  # Type of wavlm_training_args_config
    early_stopping_config: Dict[str, Any],  # Type of early_stopping_config
) -> Dict[str, float]:
    """
    Define training arguments, train and save model.

    Args:
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        wavlm_save_dir_config (dict): Configuration for saving directory.
        model (Model): The model to train.
        feature_extractor (Tokenizer): The feature extractor for the model.
        compute_metrics (Callable): A function to compute evaluation metrics.
        wavlm_training_args_config: Dict[str, Any]: configuration containing parameters
        of wavlm training arguments.
        early_stopping_config: Dict[str, Any]: configuration containing parameters of
        early stopping.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation metrics.

    """
    repo_name = wavlm_training_args_config.get("repo_name")

    training_args = TrainingArguments(
        output_dir=repo_name,
        evaluation_strategy=wavlm_training_args_config.get("evaluation_strategy"),
        save_strategy=wavlm_training_args_config.get("save_strategy"),
        learning_rate=wavlm_training_args_config.get("learning_rate"),
        per_device_train_batch_size=wavlm_training_args_config.get(
            "per_device_train_batch_size"
        ),
        gradient_accumulation_steps=wavlm_training_args_config.get(
            "gradient_accumulation_steps"
        ),
        per_device_eval_batch_size=wavlm_training_args_config.get(
            "per_device_eval_batch_size"
        ),
        num_train_epochs=wavlm_training_args_config.get("num_train_epochs"),
        warmup_ratio=wavlm_training_args_config.get("warmup_ratio"),
        logging_steps=wavlm_training_args_config.get("logging_steps"),
        load_best_model_at_end=wavlm_training_args_config.get("load_best_model_at_end"),
        metric_for_best_model=wavlm_training_args_config.get("metric_for_best_model"),
        seed=wavlm_training_args_config.get("seed"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_config.get(
                    "early_stopping_patience"
                ),
                early_stopping_threshold=early_stopping_config.get(
                    "early_stopping_threshold"
                ),
            )
        ],  # early_stopping_patience, early_stopping_threshold
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    evaluation_result = trainer.evaluate(eval_dataset)

    # Save the model locally
    trainer.save_model(wavlm_save_dir_config.get("model_dir"))

    return evaluation_result


def train_and_save_model(
    wavlm_training_args_config: Dict[str, Any],
    early_stopping_config: Dict[str, Any],
    wavlm_save_dir_config: Dict[str, Any],
    encoded_train_dataset_hf: Dataset,
    encoded_eval_dataset_hf: Dataset,
    wavlm_model_parameters_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Instantiating the pre-trained model, and loading the datasets to be passed into
    function train_model to train, save and evaluate finetuned model.

    Args:
        wavlm_training_args_config (dict): Configuration for training arguments.
        early_stopping_config (dict): Configuration for early stopping.
        wavlm_save_dir_config (dict): Configuration for saving directory.
        encoded_train_dataset_hf (Dataset): training hf dataset.
        encoded_eval_dataset_hf (Dataset): evaluation hf dataset.
        wavlm_model_parameters_config (dict): Configuration for model parameters.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: evaluation results and whether the model
        has trained successfully.
    """

    # Define model
    model, feature_extractor, f1 = model_parameters(
        wavlm_model_parameters_config, wavlm_training_args_config
    )

    def compute_metrics(eval_pred, f1=f1) -> float:
        """
        Compute the metrics for evaluating predictions.

        Parameters:
            eval_pred (object): An object containing the predictions and label ids.

        Returns:
            float: The F1 score computed using the predictions and label ids.
        """
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average="weighted"
        )

    # Train, evaluate and save model
    wavlm_evaluation_result = train_model(
        encoded_train_dataset_hf,
        encoded_eval_dataset_hf,
        wavlm_save_dir_config,
        model,
        feature_extractor,
        compute_metrics,
        wavlm_training_args_config,
        early_stopping_config,
    )

    # To return dictionary if model is successfully trained
    model_success_dict = {"training_successful": True}

    return [wavlm_evaluation_result], [model_success_dict]
