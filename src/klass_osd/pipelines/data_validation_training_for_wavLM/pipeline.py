"""
This is a boilerplate pipeline 'data_validation_training_for_wavLM'
generated using Kedro 0.18.14
"""

import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.data_validation_training_for_wavLM.nodes import (
    combined_validation_check_return_revised_dataset,
    get_gender_chart,
    get_histogram_plt,
)

logger = logging.getLogger(__name__)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=combined_validation_check_return_revised_dataset,
                inputs=[
                    "libri2mix_modeling_dataset_detailed",
                    "params:validation_check_return_revised_datasets.train_group",
                    "params:get_database_yml.train_audio_files_folder_dir",
                    "params:validation_check_return_revised_datasets.eval_group",
                    "params:get_database_yml.eval_audio_files_folder_dir",
                    "params:get_database_yml.max_end_duration",
                    "params:get_database_yml.target_sampling_rate",
                ],
                outputs=[
                    "revised_hashing_df",
                    "train_validation_check_report",
                    "eval_validation_check_report",
                ],
                name="combined_validation_check_return_revised_dataset",
            ),
            node(
                func=get_histogram_plt,
                inputs=[
                    "revised_hashing_df",
                    "params:validation_check_return_revised_datasets.no_of_speakers",
                    "params:validation_check_return_revised_datasets.train_group",
                ],
                outputs="train_snr_histogram_plt",
                name="train_get_histogram_plt",
            ),
            node(
                func=get_histogram_plt,
                inputs=[
                    "revised_hashing_df",
                    "params:validation_check_return_revised_datasets.no_of_speakers",
                    "params:validation_check_return_revised_datasets.eval_group",
                ],
                outputs="eval_snr_histogram_plt",
                name="eval_get_histogram_plt",
            ),
            node(
                func=get_gender_chart,
                inputs=[
                    "revised_hashing_df",
                    "params:validation_check_return_revised_datasets.no_of_speakers",
                    "params:validation_check_return_revised_datasets.train_group",
                ],
                outputs="train_gender_barchart",
                name="train_get_gender_chart",
            ),
            node(
                func=get_gender_chart,
                inputs=[
                    "revised_hashing_df",
                    "params:validation_check_return_revised_datasets.no_of_speakers",
                    "params:validation_check_return_revised_datasets.eval_group",
                ],
                outputs="eval_gender_barchart",
                name="eval_get_gender_chart",
            ),
        ]
    )
