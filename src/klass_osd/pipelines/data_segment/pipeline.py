"""
This is a boilerplate pipeline 'data_segment'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.data_segment.nodes import save_signals_only_librimix_in_parts


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline for model evaluation.

    Returns:
        Pipeline: The created pipeline.

    Raises:
        None
    """
    return pipeline(
        [
            node(
                func=save_signals_only_librimix_in_parts,
                inputs=[
                    "params:save_signals_only_librimix_in_parts.wave_type_list",
                    "params:save_signals_only_librimix_in_parts.mixture_type_list",
                    "params:save_signals_only_librimix_in_parts.split_type_list",
                    "params:save_signals_only_librimix_in_parts.num_speaker_mix",
                    "revised_hashing_df",
                    "params:save_signals_only_librimix_in_parts.part_type_list",
                    "params:save_signals_only_librimix_in_parts.librimix_audio_folder",
                    "params:save_signals_only_librimix_in_parts.metadata_directory_pathway",
                    "params:save_signals_only_librimix_in_parts.batchsize",
                ],
                outputs="librimix_segments_df",
                name="save_signals_only_librimix_in_parts_output",
            ),
        ]
    )
