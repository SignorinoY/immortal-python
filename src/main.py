from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.mnist import FashionMNISTDataModule
from model.classfier import FashionMNISTClassfier


def main(args):
    data = FashionMNISTDataModule(args.data_dir, args.batch_size, args.num_workers)
    model = FashionMNISTClassfier()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.model_dir,
        filename="fashion-mnist-v0-{epoch:02d}-{val_loss:.2f}",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=5)
    logger = TensorBoardLogger(args.log_dir, name="classifier")
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, early_stop_callback], logger=logger
    )

    trainer.fit(model, datamodule=data)
    trainer.test(datamodule=data)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)
