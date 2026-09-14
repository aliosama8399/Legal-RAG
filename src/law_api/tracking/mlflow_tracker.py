import os
from contextlib import contextmanager


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    @contextmanager
    def run(self, model_name: str, storage_name: str, **parameters):
        # Tracking is best-effort: an unreachable server must not fail the request,
        # and default mlflow retries would otherwise block for minutes.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        mlflow = None
        try:
            import mlflow as mlflow_module

            mlflow_module.set_tracking_uri(self.tracking_uri)
            mlflow_module.set_experiment(self.experiment_name)
            mlflow_module.start_run(run_name=f"{model_name}-{storage_name}")
            mlflow_module.log_params(
                {"embedding_model": model_name, "storage_provider": storage_name, **parameters}
            )
            mlflow = mlflow_module
        except Exception:
            mlflow = None
        try:
            yield mlflow
        finally:
            if mlflow is not None:
                try:
                    mlflow.end_run()
                except Exception:
                    pass