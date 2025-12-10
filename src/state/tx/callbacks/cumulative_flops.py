import logging
from typing import Any, Optional

import torch
from lightning import LightningModule, Trainer
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CumulativeFLOPSCallback(Callback):
    """
    PyTorch Lightning callback to track cumulative FLOPS during training.

    - Measures FLOPs once on the first training batch using `measure_flops`.
    - Tracks cumulative FLOPs and logs at validation frequency.
    - Logs cumulative_flops to trainer loggers (e.g., W&B, CSV) at validation cadence.

    Args:
        use_backward: If True, include backward pass FLOPs in the measurement.
    """

    def __init__(
        self,
        *,
        use_backward: bool = False,
    ) -> None:
        super().__init__()
        self.use_backward = use_backward

        self._flops_per_batch: Optional[int] = None
        self._measured: bool = False
        self._cumulative_flops: int = 0
        self._batch_count: int = 0

    def _trainstep_forward_backward(self, model: LightningModule, batch: Any) -> torch.Tensor:
        """Encapsulate calling StateTransitionPerturbationModel.training_step and backward.

        This intentionally targets StateTransitionPerturbationModel's signature and
        performs both forward and backward to capture full FLOPs.

        !!WARNING!!
        This has only been tested with StateTransitionPerturbationModel. Behavior with any other model has not been verified.
        """
        model.zero_grad(set_to_none=True)
        loss: torch.Tensor = model.training_step(batch, 0, padded=True)  # type: ignore
        if self.use_backward:
            loss.backward()
        return loss

    def _measure_flops_once(self, trainer: Trainer, pl_module: Any, batch: Any) -> None:
        if self._measured:
            return

        model = pl_module

        def forward_fn():
            return self._trainstep_forward_backward(model, batch)

        self._flops_per_batch = int(measure_flops(model, forward_fn=forward_fn))
        logger.info(f"CumulativeFLOPSCallback: Measured FLOPs per batch: {self._flops_per_batch}")

        model.zero_grad(set_to_none=True)
        self._measured = True

    def on_train_batch_start(self, trainer: Trainer, pl_module: Any, batch: dict, batch_idx: int) -> None:
        if not self._measured and batch_idx == 0 and trainer.current_epoch == 0:
            self._measure_flops_once(trainer, pl_module, batch)

    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: Any, batch: dict, batch_idx: int) -> None:
        if self._flops_per_batch is None:
            return

        self._batch_count += 1
        self._cumulative_flops += self._flops_per_batch

        # Log cumulative FLOPs after every training batch
        pl_module.log(
            "cumulative_flops",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        logger.info(f"CumulativeFLOPSCallback: Logged cumulative FLOPs: {self._cumulative_flops}")

    def on_validation_start(self, trainer: Trainer, pl_module: Any) -> None:
        if self._flops_per_batch is None:
            return

        # Log cumulative FLOPs at validation frequency for W&B panel alignment
        pl_module.log(
            "cumulative_flops_val_sync",
            float(self._cumulative_flops),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
