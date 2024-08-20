"""Hooks to add functionality before/after pipelines/nodes."""
import os
from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.framework.project import settings

from klass_osd.utils import logging


class ModelTrackingHooks:
    """Namespace for grouping all model-tracking hooks with MLflow together."""

    @hook_impl
    def before_pipeline_run(self) -> None:
        """Hook implementation to start an MLflow run
        with the session_id of the Kedro pipeline run.

        Args:
            run_params (dict): default parameters fed by kedro library.
            Consist of project config such as run_id

        Returns:
            None
        """

        conf_path = str(Path.cwd() / settings.CONF_SOURCE)
        conf_loader = OmegaConfigLoader(conf_source=conf_path)
        mlflow_setup = conf_loader["credentials"]

        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_setup[
            "MLFLOW_TRACKING_USERNAME"
        ]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_setup[
            "MLFLOW_TRACKING_PASSWORD"
        ]

        mlflow_experiment = mlflow_setup["MLFLOW_EXP_NAME"]
        mlflow_uri = mlflow_setup["MLFLOW_TRACKING_URI"]
        setup_mlflow = mlflow_setup["SETUP_MLFLOW"] == "true"
        mlflow_autolog = mlflow_setup["MLFLOW_AUTOLOG"] == "true"

        self.mlflow_init_status, self.mlflow_run = logging.mlflow_init(
            mlflow_experiment=mlflow_experiment,
            uri=mlflow_uri,
            setup_mlflow=setup_mlflow,
            autolog=mlflow_autolog,
        )

        # logging_utils.mlflow_log(
        #     self.mlflow_init_status,
        #     "log_params",
        #     params={
        #         "duration": duration,
        #         "batch_size": batch_size,
        #     },
        # )

        # conf_path = str(Path.cwd() / settings.CONF_SOURCE)
        # conf_loader = OmegaConfigLoader(conf_source=conf_path)
        # credentials = conf_loader["credentials"]

        # os.environ['MLFLOW_TRACKING_USERNAME']=credentials['MLFLOW_TRACKING_USERNAME']
        # os.environ['MLFLOW_TRACKING_PASSWORD']=credentials['MLFLOW_TRACKING_PASSWORD']

        # mlflow_config_path = "./conf/base/pipelines.yaml"
        # run_name = run_params["pipeline_name"] + '-' + run_params["session_id"]
        # self.mlflow_init_status, self.mlflow_run = mlflow_init(
        #    run_name=run_name, mlflow_config_path=mlflow_config_path
        # )

    # @hook_impl
    # def before_node_run(
    #     self, catalog: DataCatalog, inputs: Dict[str, Any], session_id: str
    # ) -> Dict:
    #     """Hook implementation to convert params to OmegaConf before passing to node

    #         Args:
    #             catalog (DataCatalog): DataCatalog Object consiting datasets from
    #                                      catalog.yaml
    #             inputs (Dict): dictionary of params that go into node as input
    #             session_id (str): session_id of kedro run

    #         Returns:
    #             Dict: dictionary where key is the name on input to replace and the
    #                   value as the value to replace previous value
    #     """
    #     keys = inputs.keys()
    #     param_key = None
    #     for key in keys:
    #         if "params" in key:
    #             param_key = key
    #     if param_key != None:
    #         try:
    #             cfg = OmegaConf.create(inputs[param_key])
    #             return {param_key:cfg}
    #         except:
    #             return
    #     return

    # @hook_impl
    # def after_node_run(
    #     self, node: Node, outputs: Dict[str, Any], inputs: Dict[str, Any]
    # ) -> None:
    #     """Hook implementation to add model tracking after some node runs.
    #     In this example, we will:
    #     * Log the parameters after the data splitting node runs.
    #     * Log the model after the model training node runs.
    #     * Log the model's metrics after the model evaluating node runs.
    #     """
    #     if node._func_name == "split_data":
    #         mlflow.log_params(
    #             {"split_data_ratio": inputs["params:example_test_data_ratio"]}
    #         )

    #     elif node._func_name == "train_model":
    #         model = outputs["example_model"]
    #         mlflow.sklearn.log_model(model, "model")
    #         mlflow.log_params(inputs["parameters"])

    @hook_impl
    def after_pipeline_run(self) -> None:
        """Hook implementation to end the MLflow run and save log files to MLFlow
        after the Kedro pipeline finishes.
        """
        logging.mlflow_end(self.mlflow_init_status, self.mlflow_run, True)

    @hook_impl
    def on_pipeline_error(self) -> None:
        """Hook implementation to end the MLflow run and save log files to MLFLow
        after the Kedro pipeline finishes.
        """
        logging.mlflow_end(self.mlflow_init_status, self.mlflow_run, False)
