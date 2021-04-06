# Immortal Python

An example project to illustrate the use of GPU servers for computing is based on Pytorch.

## Installation

Use the package manager [conda](https://www.anaconda.com/) to install required packages.

```bash
conda env create --file immortal.yml
```

## Usage

```bash
conda activate immortal
python src/main.py --gpus 1 --learning_rate 1e-4 --batch_size 256 --num_workers 4 --pin_memory True
```

```bash
tensorboard --logdir ./log/
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)