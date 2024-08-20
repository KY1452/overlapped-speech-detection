"""
This is a boilerplate pipeline 'wavlm_model_training'
generated using Kedro 0.18.14
"""
# import logging

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.wavlm_model_training.nodes import train_and_save_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_and_save_model,
                inputs=[
                    "params:wavlm_training_args_config",
                    "params:early_stopping_config",
                    "params:wavlm_save_dir_config",
                    "hf_train_dataset",
                    "hf_eval_dataset",
                    "params:wavlm_model_parameters_config",
                ],
                outputs=[
                    "wavlm_evaluation_result",
                    "model_training_success",
                ],
                name="wavlm_model_training",
            ),
        ]
    )
