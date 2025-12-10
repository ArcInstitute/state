import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch.optim import Optimizer

from ..models import PerturbationModel
from .batch_speed_monitor import BatchSpeedMonitorCallback
from .model_flops_utilization import ModelFLOPSUtilizationCallback
from .cumulative_flops import CumulativeFLOPSCallback

__all__ = ["PerturbationModel", "BatchSpeedMonitorCallback", "ModelFLOPSUtilizationCallback", "CumulativeFLOPSCallback"]


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        pl_module.log("train/gradient_norm", gradient_norm(pl_module))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm
