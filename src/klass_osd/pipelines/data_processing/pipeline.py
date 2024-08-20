"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.data_processing.nodes import (  # iterate_sparselibrimix_3,; libri3mix_generate_labels_df,;
    create_unaligned_dataframe,
    iterate_sparselibrimix_2,
    libri2mix_generate_labels_df,
    libri3mix_generate_labels_df,
    iterate_sparselibrimix_3
)


def create_pipeline() -> Pipeline:
    """
    Create a Kedro pipeline for generating time labels for non-speech/one/two/three
    speakers for librimix dataset.

    Args:

    Returns:
        A Kedro pipeline that takes in raw metadata json files and .txt files and
        generates processed labels for Sparselibrimix/librimix dataset.

    """
    return pipeline(
        [
            node(
                func=create_unaligned_dataframe,
                inputs=[
                    "unligned_txt_file",
                ],
                outputs="unaligned_df",
                name="create_unaligned_dataframe",
            ),
            node(
                func=iterate_sparselibrimix_2,
                inputs="sparselibrimix_metadata",
                outputs="sparselibrimix2_processed_data",
                name="iterate_sparselibrimix_2",
            ),
            node(
                func=iterate_sparselibrimix_3,
                inputs="sparselibrimix_metadata",
                outputs="sparselibrimix3_processed_data",
                name="iterate_sparselibrimix_3",
            ),
            node(
                func=libri2mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.dev_alignment_dir",
                    "params:label_generate.dev_two_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.dev_save_data_mixture_name_2mix",
                ],
                outputs="dev_libri2mix_labels_df",
                name="dev_generate_libri2mix_labels",
            ),
            node(
                func=libri2mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.test_alignment_dir",
                    "params:label_generate.test_two_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.test_save_data_mixture_name_2mix",
                ],
                outputs="test_libri2mix_labels_df",
                name="test_generate_libri2mix_labels",
            ),
            node(
                func=libri2mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.train100_alignment_dir",
                    "params:label_generate.train100_two_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.train100_save_data_mixture_name_2mix",
                ],
                outputs="train100_libri2mix_labels_df",
                name="train100_generate_libri2mix_labels",
            ),
            node(
                func=libri2mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.train360_alignment_dir",
                    "params:label_generate.train360_two_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.train360_batchsize",
                    "params:label_generate.train360_save_data_mixture_name_2mix",
                ],
                outputs="train360_libri2mix_labels_df",
                name="train360_generate_libri2mix_labels",
            ),
            node(
                func=libri3mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.dev_alignment_dir",
                    "params:label_generate.dev_three_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.dev_save_data_mixture_name_3mix",
                ],
                outputs="dev_libri3mix_labels_df",
                name="dev_generate_libri3mix_labels",
            ),
            node(
                func=libri3mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.test_alignment_dir",
                    "params:label_generate.test_three_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.test_save_data_mixture_name_3mix",
                ],
                outputs="test_libri3mix_labels_df",
                name="test_generate_libri3mix_labels",
            ),
            node(
                func=libri3mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.train100_alignment_dir",
                    "params:label_generate.train100_three_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.train100_save_data_mixture_name_3mix",
                ],
                outputs="train100_libri3mix_labels_df",
                name="train100_generate_libri3mix_labels",
            ),
            node(
                func=libri3mix_generate_labels_df,
                inputs=[
                    "unaligned_df",
                    "params:label_generate.train360_alignment_dir",
                    "params:label_generate.train360_three_mix_metadata",
                    "params:label_generate.sampling_rate",
                    "params:label_generate.batchsize",
                    "params:label_generate.train360_save_data_mixture_name_3mix",
                ],
                outputs="train360_libri3mix_labels_df",
                name="train360_generate_libri3mix_labels",
            ),
        ]
    )
