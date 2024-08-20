"""
This is a boilerplate pipeline 'data_600ms_chunks'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from klass_osd.pipelines.data_600ms_chunks.nodes import (
    generate_sparselibrimix_all_chunks,
    rename_sparselibrimix_audio_files,
)


def create_pipeline() -> Pipeline:
    """
    Create a pipeline with the given keyword arguments and return the pipeline object.
    """
    return pipeline(
        [
            node(
                func=rename_sparselibrimix_audio_files,
                inputs=[
                    "params:rename_sparselibrimix_audio_files.audio_folder",
                ],
                outputs="rename_done",
                name="rename_sparselibrimix_audio_files",
            ),
            node(
                func=generate_sparselibrimix_all_chunks,
                inputs=[
                    "sparselibrimix2_processed_data",
                    "rename_done",
                    "params:generate_sparselibrimix_all_chunks.audio_folder",
                    "params:generate_sparselibrimix_all_chunks.sample_rate",
                    "params:generate_sparselibrimix_all_chunks.subset_clean",
                    "params:generate_sparselibrimix_all_chunks.chunk_duration",
                ],
                outputs="sparselibrimix_600ms_chunks_clean",
                name="generate_sparselibrimix_all_chunks_clean",
            ),
            node(
                func=generate_sparselibrimix_all_chunks,
                inputs=[
                    "sparselibrimix2_processed_data",
                    "rename_done",
                    "params:generate_sparselibrimix_all_chunks.audio_folder",
                    "params:generate_sparselibrimix_all_chunks.sample_rate",
                    "params:generate_sparselibrimix_all_chunks.subset_noisy",
                    "params:generate_sparselibrimix_all_chunks.chunk_duration",
                ],
                outputs="sparselibrimix_600ms_chunks_noisy",
                name="generate_sparselibrimix_all_chunks_noisy",
            ),
        ]
    )
