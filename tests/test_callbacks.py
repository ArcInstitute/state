from typing import Any, cast
import pytest

from state.tx.callbacks import model_flops_utilization as mfu
from state.tx.callbacks.model_flops_utilization import ModelFLOPSUtilizationCallback
from state.tx.callbacks.cumulative_flops import CumulativeFLOPSCallback
import torch


class FakeTrainer:
    def __init__(self, num_devices: int = 1, current_epoch: int = 0):
        self.num_devices = num_devices
        self.current_epoch = current_epoch


class FakeModel(torch.nn.Module):
    def __init__(self, in_dim: int = 8, out_dim: int = 8) -> None:
        super().__init__()
        # Keep operations simple and deterministic for FLOPs counting
        self.weight = torch.nn.Parameter(torch.ones(in_dim, out_dim))
        self.logged = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matmul: (1 x in_dim) @ (in_dim x out_dim) -> (1 x out_dim)
        return x @ self.weight

    def training_step(self, batch, idx, padded: bool = True) -> torch.Tensor:
        # Derive a batch size from the fake batch to influence FLOPs
        # Our tests use cell_set_len=5 and fake_batch has shape (batch_size * cell_set_len, ...)
        bsz = batch["pert_cell_emb"].shape[0] // 5
        x = torch.ones(bsz, self.weight.shape[0], requires_grad=True)
        y = self.forward(x)
        return y.sum()

    def log(self, name, value, *, prog_bar=False, on_step=False, on_epoch=False, sync_dist=False):
        self.logged.append(
            {
                "name": name,
                "value": value,
                "prog_bar": prog_bar,
                "on_step": on_step,
                "on_epoch": on_epoch,
                "sync_dist": sync_dist,
            }
        )


@pytest.fixture
def fake_model():
    # Function-scoped fake model implementing the minimal interface used by the callback
    # Use 1x1 matmul so forward FLOPs are exactly 2 (multiply + add)
    return FakeModel(in_dim=1, out_dim=1)


@pytest.fixture
def trainer():
    return FakeTrainer(num_devices=2, current_epoch=0)


class _Arr:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def fake_batch():
    # Create a flattened batch where total rows = batch_size * cell_set_len
    # We'll use batch_size=4 and cell_set_len=5 consistently in tests
    return {"pert_cell_emb": _Arr((20, 3))}


def test_measure_flops_once_only_first_batch_and_epoch(fake_model, fake_batch):
    cb = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=False, logging_interval=1, window_size=10)
    trainer = FakeTrainer(num_devices=1, current_epoch=0)
    # Initialize throughput to avoid None checks elsewhere
    cb.setup(cast(Any, trainer), fake_model, stage="fit")

    # First batch, first epoch -> should measure exactly once
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=0)
    first_logs = [e for e in fake_model.logged if e["name"] == "flops_per_batch"]
    assert cb._measured is True and len(first_logs) == 1

    # Subsequent batch in same epoch -> no re-measure
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=1)
    assert len([e for e in fake_model.logged if e["name"] == "flops_per_batch"]) == 1

    # First batch of a later epoch -> still no re-measure because already measured
    trainer.current_epoch = 1
    cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=0)
    assert len([e for e in fake_model.logged if e["name"] == "flops_per_batch"]) == 1


def test_measure_flops_once_counts_forward_and_backward_flops(fake_model, fake_batch):
    # Compare forward-only vs forward+backward FLOPs
    trainer = FakeTrainer(num_devices=1, current_epoch=0)

    # Forward-only
    cb_fwd = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=False)
    cb_fwd._measured = False
    cb_fwd._flops_per_batch = None
    cb_fwd._measure_flops_once(cast(Any, trainer), fake_model, fake_batch)

    # Forward + backward
    cb_bwd = ModelFLOPSUtilizationCallback(cell_set_len=5, use_backward=True)
    cb_bwd._measured = False
    cb_bwd._flops_per_batch = None
    cb_bwd._measure_flops_once(cast(Any, trainer), fake_model, fake_batch)

    # Expect backward ≈ 2x forward for matmul (dX and dW), so total ≈ forward + 2*forward = 3x forward
    assert cb_fwd._flops_per_batch is not None and cb_bwd._flops_per_batch is not None
    fwd = cast(int, cb_fwd._flops_per_batch)
    bwd = cast(int, cb_bwd._flops_per_batch)
    assert bwd == 3 * fwd
    # Ensure it was logged on the model
    assert any(e["name"] == "flops_per_batch" and e["value"] == cb_bwd._flops_per_batch for e in fake_model.logged)


def test_mfu_is_calculated_correctly(fake_model, fake_batch):
    # Setup callback with small window for faster MFU computation
    cb = ModelFLOPSUtilizationCallback(
        cell_set_len=5,
        use_backward=False,
        logging_interval=5,
        available_flops=1000,
        window_size=3,
    )
    trainer = FakeTrainer(num_devices=1, current_epoch=0)
    cb.setup(cast(Any, trainer), fake_model, stage="fit")

    # Set known FLOPs per batch to avoid measurement
    cb._measured = True
    cb._flops_per_batch = 1000

    # Simulate training with 1 second per batch
    start_time = mfu.time.time()

    for batch_idx in range(16):
        if batch_idx == 0:
            cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=batch_idx)
            cb._train_start_time = start_time
        else:
            cb.on_train_batch_start(cast(Any, trainer), fake_model, fake_batch, batch_idx=batch_idx)

        # Mock time progression
        current_time = start_time + (batch_idx + 1) * 1.0
        original_time = mfu.time.time
        mfu.time.time = lambda: current_time

        try:
            cb.on_train_batch_end(cast(Any, trainer), fake_model, outputs=None, batch=fake_batch, batch_idx=batch_idx)
        finally:
            mfu.time.time = original_time

    # Verify MFU was logged with reasonable values
    mfu_logs = [e for e in fake_model.logged if e["name"] == "mfu (%)"]
    assert len(mfu_logs) >= 1

    for mfu_log in mfu_logs:
        assert 50 <= mfu_log["value"] <= 150

    # Verify samples per second was logged with reasonable values
    sps_logs = [e for e in fake_model.logged if e["name"] == "cell_sets_per_sec"]
    assert len(sps_logs) >= 1

    for sps_log in sps_logs:
        assert 2 <= sps_log["value"] <= 6


class TestCumulativeFLOPSCallback:
    def test_cumulative_flops_calculation_accuracy(self, fake_model, fake_batch):
        """Test that cumulative FLOPs calculation is accurate."""
        cb = CumulativeFLOPSCallback(use_backward=False)
        trainer = FakeTrainer(num_devices=1, current_epoch=0)

        # Set known FLOPs per batch to avoid measurement
        cb._measured = True
        cb._flops_per_batch = 1000

        # Simulate 5 training batches
        for batch_idx in range(5):
            cb.on_train_batch_end(cast(Any, trainer), fake_model, outputs=None, batch=fake_batch, batch_idx=batch_idx)

        # Check cumulative FLOPs
        assert cb._cumulative_flops == 5000
        assert cb._batch_count == 5

    def test_cumulative_flops_batch_logging(self, fake_model, fake_batch):
        """Test that cumulative FLOPs are logged after every training batch."""
        cb = CumulativeFLOPSCallback(use_backward=False)
        trainer = FakeTrainer(num_devices=1, current_epoch=0)

        # Set known FLOPs per batch
        cb._measured = True
        cb._flops_per_batch = 500

        # Simulate some training batches
        for batch_idx in range(3):
            cb.on_train_batch_end(cast(Any, trainer), fake_model, outputs=None, batch=fake_batch, batch_idx=batch_idx)

        # Should have cumulative FLOPs and logged after each batch
        assert cb._cumulative_flops == 1500

        # Check that cumulative_flops was logged 3 times (once per batch)
        cumulative_logs = [log for log in fake_model.logged if log["name"] == "cumulative_flops"]
        assert len(cumulative_logs) == 3
        assert cumulative_logs[0]["value"] == 500.0  # After batch 0
        assert cumulative_logs[1]["value"] == 1000.0  # After batch 1
        assert cumulative_logs[2]["value"] == 1500.0  # After batch 2

        # Verify logging parameters
        for log in cumulative_logs:
            assert log["on_step"] is True
            assert log["on_epoch"] is False
