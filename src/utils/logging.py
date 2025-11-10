"""Logging utilities for Aim and MLflow integration."""

from typing import Dict, Any, Optional
import logging


class AimLogger:
    """Aim logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, run_name: str):
        try:
            from aim import Run
            # Create a new run (Aim will generate a hash automatically)
            # run_hash parameter is only for resuming existing runs
            self.run = Run(experiment=experiment_name)
            # Set the run name for display purposes
            self.run.name = run_name
            self.enabled = True
        except ImportError:
            logging.warning("Aim not available, logging disabled")
            self.enabled = False
            self.run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.enabled and self.run:
            for key, value in params.items():
                self.run.set(key, value, strict=False)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled and self.run:
            for key, value in metrics.items():
                self.run.track(value, name=key, step=step or 0)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        if self.enabled and self.run:
            self.run['config'] = config


class MLflowLogger:
    """MLflow logger for experiment tracking."""
    
    def __init__(self, experiment_name: str, run_name: str):
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            self.run = mlflow.start_run(run_name=run_name)
            self.enabled = True
        except ImportError:
            logging.warning("MLflow not available, logging disabled")
            self.enabled = False
            self.run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.enabled:
            import mlflow
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            import mlflow
            mlflow.log_metrics(metrics, step=step or 0)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        if self.enabled:
            import mlflow
            mlflow.log_dict(config, "config.yaml")
    
    def end_run(self):
        """End the MLflow run."""
        if self.enabled:
            import mlflow
            mlflow.end_run()


class PrintLogger:
    """Simple print-based logger."""
    
    def __init__(self):
        self.enabled = True
    
    def log_params(self, params: Dict[str, Any]):
        """Print hyperparameters."""
        if self.enabled:
            print("Parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Print metrics."""
        if self.enabled:
            step_str = f" (step {step})" if step is not None else ""
            print(f"Metrics{step_str}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
    
    def log_config(self, config: Dict[str, Any]):
        """Print configuration."""
        if self.enabled:
            print("Configuration:")
            print(config)


def make_loggers(cfg) -> Dict[str, Any]:
    """Create loggers based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary of logger instances
    """
    loggers = {}
    experiment_name = cfg.get('experiment_name', 'seqbench')
    run_name = cfg.get('run_name', 'default')
    
    logger_config = cfg.get('loggers', {})
    
    if logger_config.get('aim', False):
        loggers['aim'] = AimLogger(experiment_name, run_name)
    
    if logger_config.get('mlflow', False):
        loggers['mlflow'] = MLflowLogger(experiment_name, run_name)
    
    if logger_config.get('print', True):
        loggers['print'] = PrintLogger()
    
    return loggers

