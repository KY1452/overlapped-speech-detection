"""
This is a boilerplate pipeline 'wavlm_hf_dataset'
generated using Kedro 0.18.14
"""

import os

from kedro.config import OmegaConfigLoader
from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.wavlm_hf_dataset.nodes import curate_hf_dataset

cwd = os.getcwd()
conf_path = os.path.join(cwd, "conf")
conf_loader = OmegaConfigLoader(conf_source=conf_path)
parameters = conf_loader["parameters"]


def create_pipeline() -> Pipeline:
    """
    Create a pipeline with the given keyword arguments and return the pipeline object.
    """
    default_datasets = [
        node(
            func=curate_hf_dataset,
            inputs=[
                "librimix_segments_df",
                "params:curate_hf_dataset_train.output_folder_path",
                "params:curate_hf_dataset_train.dataset_type",
                "params:curate_hf_dataset_train.encoded",
                "params:curate_hf_dataset_train.feature_extractor",
            ],
            outputs="hf_train_dataset",
            name="curate_hf_dataset_train",
        ),
        node(
            func=curate_hf_dataset,
            inputs=[
                "libri2mix_test_chunks_df",
                "params:curate_hf_dataset_test.output_folder_path",
                "params:curate_hf_dataset_test.dataset_type",
                "params:curate_hf_dataset_test.encoded",
                "params:curate_hf_dataset_test.feature_extractor",
            ],
            outputs="hf_test_dataset",
            name="curate_hf_dataset_test",
        ),
        node(
            func=curate_hf_dataset,
            inputs=[
                "libri2mix_eval_chunks_df",
                "params:curate_hf_dataset_eval.output_folder_path",
                "params:curate_hf_dataset_eval.dataset_type",
                "params:curate_hf_dataset_eval.encoded",
                "params:curate_hf_dataset_eval.feature_extractor",
            ],
            outputs="hf_eval_dataset",
            name="curate_hf_dataset_eval",
        ),
        node(
            func=curate_hf_dataset,
            inputs=[
                "sparselibrimix_600ms_chunks_clean",
                "params:curate_hf_dataset_sparselibrimix.output_folder_path",
                "params:curate_hf_dataset_sparselibrimix.dataset_type",
                "params:curate_hf_dataset_sparselibrimix.encoded",
                "params:curate_hf_dataset_sparselibrimix.feature_extractor",
            ],
            outputs="hf_sparselibrimix_dataset_clean",
            name="curate_hf_dataset_sparselibrimix_clean",
        ),
        node(
            func=curate_hf_dataset,
            inputs=[
                "sparselibrimix_600ms_chunks_noisy",
                "params:curate_hf_dataset_sparselibrimix.output_folder_path",
                "params:curate_hf_dataset_sparselibrimix.dataset_type",
                "params:curate_hf_dataset_sparselibrimix.encoded",
                "params:curate_hf_dataset_sparselibrimix.feature_extractor",
            ],
            outputs="hf_sparselibrimix_dataset_noisy",
            name="curate_hf_dataset_sparselibrimix_noisy",
        ),
    ]
    if parameters["curate_hf_dataset_augment"]["augmented"]:
        augmented_datasets = [
            node(
                func=curate_hf_dataset,
                inputs=[
                    "librimix_segments_df",
                    "params:curate_hf_dataset_augment.output_folder_path",
                    "params:curate_hf_dataset_augment.dataset_type_train",
                    "params:curate_hf_dataset_augment.encoded",
                    "params:curate_hf_dataset_augment.feature_extractor",
                    "params:curate_hf_dataset_augment.augment_config",
                ],
                outputs="hf_augment_dataset_train",
                name="curate_hf_dataset_augment_train",
            ),
            node(
                func=curate_hf_dataset,
                inputs=[
                    "libri2mix_eval_chunks_df",
                    "params:curate_hf_dataset_augment.output_folder_path",
                    "params:curate_hf_dataset_augment.dataset_type_eval",
                    "params:curate_hf_dataset_augment.encoded",
                    "params:curate_hf_dataset_augment.feature_extractor",
                    "params:curate_hf_dataset_augment.augment_config",
                ],
                outputs="hf_augment_dataset_eval",
                name="curate_hf_dataset_augment_eval",
            ),
        ]
        return Pipeline(default_datasets + augmented_datasets)
    else:
        return Pipeline(default_datasets)
