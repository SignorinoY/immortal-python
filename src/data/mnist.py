from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data/", batch_size: int = 32, num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def prepare_data(self):
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train, self.val = random_split(full, [55000, 5000])
            self.dims = tuple(self.train[0][0].shape)
        if stage == "test" or stage is None:
            self.test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )
            self.dims = tuple(self.test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
