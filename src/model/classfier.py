from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class FashionMNISTClassfier(pl.LightningModule):

    classes = (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot",
    )

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(1, 1, 28, 28)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        self.train_acc(preds, y)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        _, preds = torch.max(y_hat, 1)
        self.val_acc(preds, y)
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def test_epoch_end(self, outputs):
        class_probs = []
        class_preds = []
        for output in outputs:
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        for class_index in range(len(self.classes)):
            tensorboard_preds = test_preds == class_index
            tensorboard_probs = test_probs[:, class_index]
            self.logger.experiment.add_pr_curve(
                self.classes[class_index], tensorboard_preds, tensorboard_probs,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parser
