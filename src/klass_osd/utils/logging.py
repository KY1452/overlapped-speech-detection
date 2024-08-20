"""Utilities or functions that are useful across all the different
modules in this package can be defined here."""

import logging
import logging.config
import os
import traceback

import mlflow

logger = logging.getLogger(__name__)


def mlflow_init(mlflow_experiment, uri, setup_mlflow=False, autolog=False):
    """Initialise MLflow connection.

    Parameters
    ----------

    setup_mlflow : bool, optional
        Choice to set up MLflow connection, by default False
    autolog : bool, optional
        Choice to set up MLflow's autolog, by default False

    Returns
    -------
    init_success : bool
        Boolean value indicative of success
        of intialising connection with MLflow server.

    mlflow_run : Union[None, `mlflow.entities.Run` object]
        On successful initialisation, the function returns an object
        containing the data and properties for the MLflow run.
        On failure, the function returns a null value.
    """
    init_success = False
    mlflow_run = None
    logger.info(setup_mlflow)
    if setup_mlflow:
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(mlflow_experiment)  # experiment_id="7"

            if autolog:
                mlflow.autolog()

            mlflow.start_run()

            if "MLFLOW_HPTUNING_TAG" in os.environ:
                mlflow.set_tag("hptuning_tag", os.environ.get("MLFLOW_HPTUNING_TAG"))
            if "JOB_UUID" in os.environ:
                mlflow.set_tag("job_uuid", os.environ.get("JOB_UUID"))

            mlflow_run = mlflow.active_run()
            init_success = True
            logger.info("MLflow initialisation has succeeded.")
            logger.info("UUID for MLflow run: %s", mlflow_run.info.run_id)
        except mlflow.exceptions.MlflowException:
            logger.error("MLflow initialization has failed.", exc_info=True)
            traceback.print_exc()

    return init_success, mlflow_run


def mlflow_end(mlflow_init_status: bool, mlflow_run: str, success: bool):
    """End MLflow Run.

    Parameters
    ----------
    mlflow_init_status : bool
        True if mlflow intiialized else False
    mlflow_run : Union[None, `mlflow.entities.Run` object]
        Object containing the data and properties for the MLflow run.
    success : bool, optional
        True if run succeeded else False

    Returns
    -------
    None
    """
    artifact_uri = mlflow.get_artifact_uri()
    logging.info("Artifact URI: %s", artifact_uri)
    mlflow_log(
        mlflow_init_status,
        "log_artifact",
        local_path="info.log",
        artifact_file="info.log",
    )
    mlflow_log(
        mlflow_init_status,
        "log_artifact",
        local_path="error.log",
        artifact_file="error.log",
    )
    mlflow_log(
        mlflow_init_status,
        "log_artifact",
        local_path="debug.log",
        artifact_file="debug.log",
    )
    if success:
        logging.info(
            "Model training with MLflow run ID %s has succeeded.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logging.info(
            "Model training with MLflow run ID %s has failed.",
            mlflow_run.info.run_id,
        )
        mlflow.end_run("FAILED")


def mlflow_log(mlflow_init_status, log_function, **kwargs):
    """Custom function for utilising MLflow's logging functions.

    This function is only relevant when the function `mlflow_init`
    returns a "True" value, translating to a successful initialisation
    of a connection with an MLflow server.

    Parameters
    ----------
    mlflow_init_status : bool
        Boolean value indicative of success of intialising connection
        with MLflow server.
    log_function : str
        Name of MLflow logging function to be used.
        See https://www.mlflow.org/docs/latest/python_api/mlflow.html
    **kwargs
        Keyword arguments passed to `log_function`.
    """
    if mlflow_init_status:
        try:
            method = getattr(mlflow, log_function)
            method(
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in method.__code__.co_varnames
                }
            )
        except mlflow.exceptions.MlflowException as mlflow_error:
            logger.error("MLflow logging error: %s", mlflow_error)
            traceback.print_exc()

        except AttributeError as attribute_error:
            logger.error("AttributeError during MLflow logging: %s", attribute_error)
            traceback.print_exc()

        except TypeError as type_error:
            logger.error("TypeError during MLflow logging: %s", type_error)
            traceback.print_exc()
