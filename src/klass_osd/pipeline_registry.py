# """Project pipelines."""
# from __future__ import annotations

# from typing import Dict

# # import klass_osd.pipelines.data_chunks as data_chunks
# # import klass_osd.pipelines.data_segment as data_segment
# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline, pipeline


# def register_pipelines() -> Dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

"""Project pipelines."""
from __future__ import annotations

from typing import Dict

from kedro.pipeline import Pipeline
from klass_osd.pipelines.create_hashing_dataset_libri2mix_detailed import create_pipeline as create_hashing_dataset_libri2mix_detailed
from klass_osd.pipelines.data_processing import create_pipeline as data_processing
from klass_osd.pipelines.data_validation_training_for_wavLM import create_pipeline as data_validation_training_for_wavLM
from klass_osd.pipelines.data_600ms_chunks import create_pipeline as data_600ms_chunks
from klass_osd.pipelines.data_chunks import create_pipeline as data_chunks
from klass_osd.pipelines.data_segment import create_pipeline as data_segment
from klass_osd.pipelines.wavlm_hf_dataset import create_pipeline as wavlm_hf_dataset
from klass_osd.pipelines.wavlm_model_training import create_pipeline as wavlm_model_training
from klass_osd.pipelines.model_prediction import create_pipeline as model_prediction

def register_pipelines() -> Dict[str, Pipeline]:
    pipelines = {
        "create_hashing_dataset_libri2mix_detailed": create_hashing_dataset_libri2mix_detailed(),
        "data_processing": data_processing(),
        "data_validation_training_for_wavLM": data_validation_training_for_wavLM(),
        "data_600ms_chunks": data_600ms_chunks(),
        "data_chunks": data_chunks(),
        "data_segment": data_segment(),
        "wavlm_hf_dataset": wavlm_hf_dataset(),
        "wavlm_model_training": wavlm_model_training(),
        "model_prediction": model_prediction()
    }
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
