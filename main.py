import argparse
from os import read
from sklearn.model_selection import train_test_split
from wrapt.wrappers import transient_function_wrapper

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
        required=True,
        help="Input directory",
        dest="input_dir",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs/",
        help="Output directory",
        dest="output_dir",
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
    input_dir = cfg["input_dir"]

    if cfg["mode"] == "train":
        img_dir = input_dir + "images"
        mask_dir = input_dir + "masks"

        inputs, masks = read_data(img_dir, mask_dir)
        train_inputs, val_inputs, train_masks, val_masks = train_test_split(
            inputs, masks, test_size=0.1
        )
        n_epochs = cfg["n_epochs"]

        for i in range(n_epochs):
            print("Epoch ", i + 1)
            train(model, train_inputs, train_masks)

        # Save the model weights
        model_path = cfg["output_dir"] + "model_weights"
        model.save_weights(model_path)
        print("Model successfully saved!")

    if cfg["mode"] == "test":
        pass

    else:
        pass


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
