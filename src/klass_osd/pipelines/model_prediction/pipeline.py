"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.model_prediction.nodes import prediction_labels_and_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prediction_labels_and_report,
                inputs=[
                    "model_training_success",
                    "hf_eval_dataset",
                    "params:wavlm_save_dir_config",
                    "params:wavlm_training_args_config",
                ],
                outputs=[
                    "predicted_label_eval_df",
                    "wavlm_eval_report",
                ],
                name="prediction_labels_and_report_eval",
            ),
            # node(
            #     func=prediction_labels_and_report,
            #     inputs=[
            #         "model_training_success",
            #         "hf_test_dataset",
            #         "params:wavlm_save_dir_config",
            #         "params:wavlm_training_args_config",
            #     ],
            #     outputs=[
            #         "predicted_label_test_df",
            #         "wavlm_test_report",
            #     ],
            #     name="prediction_labels_and_report_test",
            # ),
            # node(
            #     func=prediction_labels_and_report,
            #     inputs=[
            #         "model_training_success",
            #         "hf_sparselibrimix_dataset_clean",
            #         "params:wavlm_save_dir_config",
            #         "params:wavlm_training_args_config",
            #     ],
            #     outputs=[
            #         "predicted_label_sparse_clean_df",
            #         "wavlm_sparse_clean_report",
            #     ],
            #     name="prediction_labels_and_report_sparse_clean",
            # ),
            # node(
            #     func=prediction_labels_and_report,
            #     inputs=[
            #         "model_training_success",
            #         "new_dataset_test",
            #         "params:wavlm_save_dir_config",
            #         "params:wavlm_training_args_config",
            #     ],
            #     outputs=[
            #         "predicted_label_new_dataset_df",
            #         "wavlm_new_dataset_report",
            #     ],
            #     name="prediction_labels_and_report_new_dataset",
            # ),            
        ]
    )
