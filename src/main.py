from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.mnist import FashionMNISTDataModule
from model.classfier import FashionMNISTClassfier
from util.logger import ImagePredictionLogger


def main(args):
    data = FashionMNISTDataModule(args.data_dir, args.batch_size, args.num_workers)
    data.prepare_data()
    data.setup()
    samples = next(iter(data.val_dataloader()))
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

    model = FashionMNISTClassfier()
    checkpoint_callback = ModelCheckpoint(
        monitor="Validation Loss",
        dirpath=args.model_dir,
        filename="fashion-mnist-v1-{epoch:02d}-{Validation Loss:.2f}",
    )
    early_stop_callback = EarlyStopping(monitor="Validation Loss", patience=5)
    image_prediction_callback = ImagePredictionLogger(samples, classes)

    logger = TensorBoardLogger(args.log_dir, name="classifier")

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stop_callback, image_prediction_callback],
        logger=logger,
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
