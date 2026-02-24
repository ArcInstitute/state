import time

from lightning.pytorch.callbacks import Callback


class BatchSpeedMonitorCallback(Callback):
    """
    Callback that logs the number of batches processed per second to wandb.
    """

    def __init__(self, logging_interval=50):
        """
        Args:
            logging_interval: Log the speed every N batches
        """
        super().__init__()
        self.logging_interval = logging_interval
        self.batch_start_time = None
        self.batch_times = []
        self.last_logged_batch = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record the start time of the batch."""
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Calculate and log the batch processing speed.
        """
        if self.batch_start_time is None:
            return

        # Calculate time taken for this batch
        batch_end_time = time.time()
        batch_time = batch_end_time - self.batch_start_time
        self.batch_times.append(batch_time)

        # Log every logging_interval batches
        if batch_idx % self.logging_interval == 0 and batch_idx > 0:
            # Calculate batches per second over the last interval
            if len(self.batch_times) > 0:
                avg_batch_time = sum(self.batch_times) / len(self.batch_times)
                batches_per_second = 1.0 / avg_batch_time if avg_batch_time > 0 else 0

                # Log to wandb
                pl_module.log("batches_per_second", batches_per_second)

            # Reset for next interval
            self.batch_times = []
            self.last_logged_batch = batch_idx
