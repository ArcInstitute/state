from lightning.pytorch.callbacks import Callback


class CPABestModelTracker(Callback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_model = None
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        if self.best_score is None:
            self.best_score = trainer.callback_metrics[self.monitor]
            self.best_model = pl_module.state_dict()
        else:
            if self.mode == "min":
                if trainer.callback_metrics[self.monitor] < self.best_score:
                    self.best_score = trainer.callback_metrics[self.monitor]
                    self.best_model = pl_module.state_dict()
            else:
                if trainer.callback_metrics[self.monitor] > self.best_score:
                    self.best_score = trainer.callback_metrics[self.monitor]
                    self.best_model = pl_module.state_dict()

    def on_train_end(self, trainer, pl_module):
        pl_module.load_state_dict(self.best_model)
        print(f"Best model loaded with {self.monitor} = {self.best_score}")
        return self.best_model
