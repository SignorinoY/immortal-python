from pytorch_lightning.callbacks import Callback
from .function import plot_classes_preds


class ImagePredictionLogger(Callback):
    def __init__(self, samples, classes, nums=4):
        super().__init__()
        self.images, self.labels = samples[:nums]
        self.classes = classes

    def on_validation_epoch_end(self, trainer, pl_module):
        images = self.images.to(device=pl_module.device)
        labels = self.labels.to(device=pl_module.device)
        trainer.logger.experiment.add_figure(
            "Predictions vs. Actuals",
            plot_classes_preds(pl_module, images, labels, self.classes),
            pl_module.current_epoch,
        )
