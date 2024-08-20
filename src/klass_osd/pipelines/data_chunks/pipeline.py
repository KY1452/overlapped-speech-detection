"""
This is a boilerplate pipeline 'data_chunks'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.data_chunks.nodes import (  # ,save_annotated_chunks_libri3_mix_in_parts,
    save_annotated_chunks_libri2_mix_in_parts,
)


def create_pipeline() -> Pipeline:
    """
    Creates a pipeline for model evaluation.

    Returns:
        Pipeline: The created pipeline.

    Raises:
        None
    """
    return pipeline(
        [
            # node(
            #     func=save_annotated_chunks_libri3_mix_in_parts,
            #     inputs=[
            #         "params:save_annotated_chunks_libri3_mix_in_parts.wave_type_list",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.mixture_type_list",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.split_type_list",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.num_speaker_mix",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.chunk_seconds",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.part_type_list",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.metadata_directory_pathway",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.file_extension",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.librimix_audio_folder",
            #         "hash_audioid_3mix",
            #         "params:save_annotated_chunks_libri3_mix_in_parts.batchsize",
            #     ],
            #     outputs="libri3mix_chunks_df",
            #     name="save_annotated_chunks_libri3_mix_in_parts_output",
            # ),
            node(
                func=save_annotated_chunks_libri2_mix_in_parts,
                inputs=[
                    "params:save_annotated_chunks_libri2_mix_in_parts.wave_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.mixture_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.split_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.num_speaker_mix",
                    "params:save_annotated_chunks_libri2_mix_in_parts.chunk_seconds",
                    "params:save_annotated_chunks_libri2_mix_in_parts.part_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.metadata_directory_pathway",
                    "params:save_annotated_chunks_libri2_mix_in_parts.file_extension",
                    "params:save_annotated_chunks_libri2_mix_in_parts.librimix_audio_folder",
                    "revised_hashing_df",
                    "params:save_annotated_chunks_libri2_mix_in_parts.batchsize",
                ],
                outputs="libri2mix_test_chunks_df",
                name="save_annotated_chunks_libri2_mix_in_parts_output_test",
            ),
            node(
                func=save_annotated_chunks_libri2_mix_in_parts,
                inputs=[
                    "params:save_annotated_chunks_libri2_mix_in_parts.wave_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.mixture_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.train100_split_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.num_speaker_mix",
                    "params:save_annotated_chunks_libri2_mix_in_parts.chunk_seconds",
                    "params:save_annotated_chunks_libri2_mix_in_parts.train100_part_type_list",
                    "params:save_annotated_chunks_libri2_mix_in_parts.metadata_directory_pathway",
                    "params:save_annotated_chunks_libri2_mix_in_parts.file_extension",
                    "params:save_annotated_chunks_libri2_mix_in_parts.librimix_audio_folder",
                    "revised_hashing_df",
                    "params:save_annotated_chunks_libri2_mix_in_parts.batchsize",
                ],
                outputs="libri2mix_eval_chunks_df",
                name="save_annotated_chunks_libri2_mix_in_parts_output_eval",
            ),
        ]
    )
