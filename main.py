import argparse

from model import UNet
from preprocess import read_data
from train import train


def get_config():
    """
    Parses the command-line arguments and sets configurations

    :return: config, a dictionary that stores all arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["train", "test", "predict"],
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        nargs=2,
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of epochs", dest="n_epochs"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate",
        dest="learning_rate",
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


def main(cfg):
    model = UNet(cfg)
    
    if cfg["mode"] == "train":
        n_epochs = cfg["n_epochs"]
        pass


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
