import logging
import time
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule, Trainer
from lightning.fabric.utilities.throughput import Throughput, measure_flops
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelFLOPSUtilizationCallback(Callback):
    """
    PyTorch Lightning callback to measure and log Model FLOPS Utilization (MFU).

    - Measures FLOPs once on the first training batch using `measure_flops`.
    - Tracks rolling throughput metrics via `Throughput` with a window equal to
      the user input window size.
    - Logs MFU to the trainer loggers (e.g., W&B) at the same cadence as other metrics.

    Args:
        available_flops: Theoretical peak flops for device in TFLOPS, example: enter 60e12 for 60 TFLOPS.
        use_backward: If True, include backward pass FLOPs in the measurement.
        logging_interval: The interval at which to log MFU.
        cell_set_len: The length of the cell set.
        window_size: The size of the rolling window.
    """

    def __init__(
        self,
        *,
        available_flops: Optional[float] = None,
        use_backward: bool = False,
        logging_interval: int = 50,
        cell_set_len: Optional[int] = None,
        window_size: int = 20,
    ) -> None:
        super().__init__()
        self.available_flops = available_flops
        print(f"ModelFLOPSUtilizationCallback: Using available flops: {self.available_flops}")
        self.use_backward = use_backward
        print(f"ModelFLOPSUtilizationCallback: Using use_backward: {self.use_backward}")
        self.logging_interval = logging_interval
        print(f"ModelFLOPSUtilizationCallback: Using logging interval: {self.logging_interval}")
        self.cell_set_len = cell_set_len
        print(f"ModelFLOPSUtilizationCallback: Using cell set length: {self.cell_set_len}")

        self._throughput: Optional[Throughput] = None
        self._window_size: int = window_size
        print(f"ModelFLOPSUtilizationCallback: Using window size: {self._window_size}")
        self._flops_per_batch: Optional[int] = None
        self._measured: bool = False
        self._train_start_time: Optional[float] = None
        self._cell_sets_len: Optional[int] = None
        # Cumulative counters since training start
        self._cumulative_time: float = 0.0
        self._cumulative_batches: int = 0
        self._cumulative_samples: int = 0

    def setup(self, trainer: Trainer, pl_module: Any, stage: str) -> None:
        # Initialize throughput tracker
        world_size = getattr(trainer, "num_devices")
        assert isinstance(world_size, int), f"world_size must be an integer, got {type(world_size)}"
        assert world_size > 0, f"world_size must be greater than 0, got {world_size}"
        print(f"ModelFLOPSUtilizationCallback: Initializing throughput tracker with world_size: {world_size}")

        self._throughput = Throughput(
            available_flops=self.available_flops,
            world_size=world_size,
            window_size=self._window_size,
        )
        # Reset cumulative counters on setup
        self._cumulative_time = 0.0
        self._cumulative_batches = 0
        self._cumulative_samples = 0

    def _infer_batch_size(self, batch: Any) -> int:
        """Infer the logical batch size.

        In the cell-load pipeline, the sampler yields flattened batches of size
        batch_size * cell_set_len. Divide the leading dimension by cell_set_len to recover the true batch size.
        """
        batch_size = batch["pert_cell_emb"].shape[0]
        return batch_size // self.cell_set_len

    def _trainstep_forward_backward(self, model: LightningModule, batch: Any) -> torch.Tensor:
        """Encapsulate calling StateTransitionPerturbationModel.training_step and backward.

        This intentionally targets StateTransitionPerturbationModel's signature and
        performs both forward and backward to capture full FLOPs.

        !!WARNING!!
        This has only been tested with StateTransitionPerturbationModel. Behavior with any other model has not been verified.
        """
        # Clean gradients before measuring
        model.zero_grad(set_to_none=True)
        # Call training_step with the expected signature
        loss: torch.Tensor = model.training_step(batch, 0, padded=True)  # type: ignore
        # Backward to include backward-pass FLOPs
        if self.use_backward:
            loss.backward()
        return loss

    def _measure_flops_once(self, trainer: Trainer, pl_module: Any, batch: Any) -> None:
        if self._measured:
            return

        model = pl_module

        # Measure FLOPs using a single callable that runs training_step and backward
        def forward_fn():
            return self._trainstep_forward_backward(model, batch)

        self._flops_per_batch = int(measure_flops(model, forward_fn=forward_fn))
        print(f"ModelFLOPSUtilizationCallback: Measured FLOPs per batch: {self._flops_per_batch}")
        pl_module.log("flops_per_batch", self._flops_per_batch, prog_bar=False, on_step=True, on_epoch=False)

        # Clear gradients before real training continues (safety)
        model.zero_grad(set_to_none=True)

        # Expose on the module for visibility/debugging
        setattr(pl_module, "flops_per_batch", self._flops_per_batch)
        self._measured = True

    def on_train_batch_start(self, trainer: Trainer, pl_module: Any, batch: dict, batch_idx: int) -> None:
        # Only calculate FLOPs on the first batch of the first epoch
        if not self._measured and batch_idx == 0 and trainer.current_epoch == 0:
            self._measure_flops_once(trainer, pl_module, batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._train_start_time = time.time()

    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: Any, batch: dict, batch_idx: int) -> None:
        if self._train_start_time is None or self._throughput is None:
            return

        samples = self._infer_batch_size(batch)

        # Update cumulative totals since training start
        self._cumulative_batches += 1
        self._cumulative_samples += samples

        # Log at a cadence controled by the logging_interval
        if batch_idx % self.logging_interval == 0 and batch_idx > 0:
            # Synchronize CUDA if available to ensure accurate timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Cumulative duration since training start
            self._cumulative_time = time.time() - self._train_start_time

            if batch_idx == self.logging_interval:
                flops = self._flops_per_batch * (self.logging_interval + 1)  # type: ignore
            else:
                flops = self._flops_per_batch * self.logging_interval  # type: ignore

            # Update throughput tracker
            self._throughput.update(
                time=self._cumulative_time,
                batches=self._cumulative_batches,
                samples=self._cumulative_samples,
                flops=flops,  # type: ignore
            )

            metrics: Dict[str, float] = self._throughput.compute()
            # Prefer global MFU when available, otherwise device MFU
            mfu = metrics.get("global/mfu", metrics.get("device/mfu", None))
            if mfu is not None:
                mfu = 100 * mfu
                pl_module.log("mfu (%)", mfu, prog_bar=True, on_step=True, on_epoch=False)

            # Log cell_sets (cell_sentences) per second
            cell_sets_per_sec = metrics.get("global/samples_per_sec", metrics.get("device/samples_per_sec", None))
            if cell_sets_per_sec is not None:
                pl_module.log(
                    "cell_sets_per_sec",
                    cell_sets_per_sec,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=False,
                )
