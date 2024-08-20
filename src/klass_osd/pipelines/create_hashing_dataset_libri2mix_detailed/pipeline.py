# """
# This is a boilerplate pipeline 'create_hashing_dataset_libri2mix_detailed'
# generated using Kedro 0.18.14
# """

# from kedro.pipeline import Pipeline, node, pipeline

# from klass_osd.pipelines.create_hashing_dataset_libri2mix_detailed.nodes import (
#     generate_modelling_dataset,
#     get_libri2mix_clean_detailed_datasets,
# )


# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                 func=get_libri2mix_clean_detailed_datasets,
#                 inputs=[
#                     "params:libri2mix_metadata_folder.database_path",
#                     "params:librimix_audio_folder.librimix_audio_folder",
#                     "train360_libri2mix_labels_df",
#                     "train100_libri2mix_labels_df",
#                     "test_libri2mix_labels_df",
#                     "dev_libri2mix_labels_df",
#                     "params:librimix_audio_folder.folder_freq",
#                     "params:librimix_audio_folder.audio_freq",
#                     "params:librimix_audio_folder.folder_audio_mix_length",
#                     "params:librimix_audio_folder.folder_mix_type",
#                     "params:librimix_audio_folder.audio_background",
#                     "params:librimix.datasets",
#                     "params:librimix.libri2mix_clean_overall_detailed_dev",
#                     "params:librimix.libri2mix_clean_overall_detailed_test",
#                     "params:librimix.libri2mix_clean_overall_detailed_train_100",
#                     "params:librimix.libri2mix_clean_overall_detailed_train_360",
#                 ],
#                 outputs="libri2mix_clean_detailed_datasets_detailed",
#                 name="get_libri2mix_clean_detailed_datasets_detailed_node",
#             ),
#             node(
#                 func=generate_modelling_dataset,
#                 inputs="libri2mix_clean_detailed_datasets_detailed",
#                 outputs="libri2mix_modeling_dataset_detailed",
#                 name="generate_modelling_dataset_detailed_node",
#             ),
#         ]
#     )


"""
This is a boilerplate pipeline 'create_hashing_dataset_libri2mix_detailed'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.create_hashing_dataset_libri2mix_detailed.nodes import (
    generate_modelling_dataset,
    get_libri2mix_clean_detailed_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_libri2mix_clean_detailed_datasets,
                inputs=[
                    "params:libri2mix_metadata_folder.database_path",
                    "params:librimix_audio_folder.librimix_audio_folder",
                    "train360_libri2mix_labels_df",
                    "train100_libri2mix_labels_df",
                    "test_libri2mix_labels_df",
                    "dev_libri2mix_labels_df",
                    "params:librimix_audio_folder.folder_freq",
                    "params:librimix_audio_folder.audio_freq",
                    "params:librimix_audio_folder.folder_audio_mix_length",
                    "params:librimix_audio_folder.folder_mix_type",
                    "params:librimix_audio_folder.audio_background",
                    "params:librimix.datasets",
                    "params:librimix.libri2mix_clean_overall_detailed_dev",
                    "params:librimix.libri2mix_clean_overall_detailed_test",
                    "params:librimix.libri2mix_clean_overall_detailed_train_100",
                    "params:librimix.libri2mix_clean_overall_detailed_train_360",
                    "params:part_number",
                ],
                outputs="libri2mix_clean_detailed_datasets_detailed",
                name="get_libri2mix_clean_detailed_datasets_detailed_node",
            ),
            node(
                func=generate_modelling_dataset,
                inputs="libri2mix_clean_detailed_datasets_detailed",
                outputs="libri2mix_modeling_dataset_detailed",
                name="generate_modelling_dataset_detailed_node",
            ),
        ]
    )




