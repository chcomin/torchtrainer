"""Some customizations for the lightning library"""

import contextlib
import os
from typing import override

from lightning.fabric.loggers.csv_logs import _ExperimentWriter as _FabricExperimentWriter
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.csv_logs import CSVLogger as CSVLoggerL
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from lightning.pytorch.loggers.wandb import WandbLogger

with contextlib.suppress(ImportError):
    import wandb


class CustomExperimentWriter(ExperimentWriter):
    """Lightning ExperimentWriter with different filename and no existing directory checks."""

    NAME_METRICS_FILE = "log.csv"

    def _check_log_dir_exists(self):
        pass

class CSVLogger(CSVLoggerL):
    """Lightning CSVLogger with a CustomExperimentWriter."""

    @property
    @override
    @rank_zero_experiment
    def experiment(self) -> _FabricExperimentWriter:
        r"""Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~lightning.pytorch.core.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = CustomExperimentWriter(log_dir=self.log_dir)
        return self._experiment
    
def setup_wandb_lightning(args, run_path):
    """Setup wandb for logging experiments."""
    
    wandb_project = args.wandb_project
    experiment_name = args.experiment_name
    run_name = args.run_name
    group = args.wandb_group

    if group == "":
        group = None

    wandb_run_name = f"{experiment_name}/{run_name}"

    os.environ["WANDB_SILENT"] = "True"
    # Find previous run with the same name and delete it
    api = wandb.Api()
    runs = api.runs(wandb_project, {"config.run_name": wandb_run_name})
    try:
        num_runs = len(runs) 
    except (ValueError, TypeError):
        # Project does not exist
        num_runs = 0

    if num_runs > 1:
        print(f"Warning, more than one run with name {wandb_run_name} found in project "
              "{wandb_project} in the wandb database. Deleting only the last run.")
    if num_runs >= 1:
        runs[-1].delete()   

    WandbLogger(
        name = wandb_run_name,
        dir = str(run_path),
        project = wandb_project,
        notes = args.meta,
        group = group,
        # track hyperparameters and run metadata
        config=args
    )