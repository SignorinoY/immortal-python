from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.mnist import FashionMNISTDataModule
from model.classfier import FashionMNISTClassfier
from util.logger import ImagePredictionLogger


def main():
    pl.seed_everything(10086)

    # args
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--model_dir", type=str, default="./model/")
    parser.add_argument("--log_dir", type=str, default="./log/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FashionMNISTClassfier.add_model_specific_args(parser)
    args = parser.parse_args()

    # data
    data = FashionMNISTDataModule(
        args.data_dir, args.batch_size, args.num_workers, args.pin_memory
    )
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

    # model
    model = FashionMNISTClassfier(args.learning_rate)
    early_stop_callback = EarlyStopping(monitor="val_loss")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir,
        filename="fashion-mnist-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
    )
    image_prediction_callback = ImagePredictionLogger(samples, classes)
    logger = TensorBoardLogger(save_dir=args.log_dir, name="classifier", log_graph=True)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[early_stop_callback, checkpoint_callback, image_prediction_callback],
        logger=logger,
    )

    # training
    trainer.fit(model, datamodule=data)

    # testing
    trainer.test(datamodule=data)


if __name__ == "__main__":
    main()
