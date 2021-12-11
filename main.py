import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from model import UNet
from utils import read_data, visualize
from train import train
from test import test, predict


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
        choices=["train", "test"],
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
        "-e", "--epochs", type=int, default=5, help="Number of epochs", dest="n_epochs"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "--class-weight", type=int, default=5, help="Class weight", dest="class_weight"
    )

    args = parser.parse_args()
    config = {}

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    return config


def main(cfg):
    model = UNet(cfg)
    input_dir = cfg["input_dir"]
    img_dir = input_dir + "/images"
    mask_dir = input_dir + "/masks"
    model_path = cfg["output_dir"] + "model_weights"

    if cfg["mode"] == "train":
        inputs, masks = read_data(img_dir, mask_dir)
        train_inputs, val_inputs, train_masks, val_masks = train_test_split(
            inputs, masks, test_size=0.2
        )
        n_epochs = cfg["n_epochs"]

        print("Training U-Net...")
        for i in range(n_epochs):
            print("Epoch:", i + 1)
            train(model, train_inputs, train_masks)
            mean_IoU = test(model, val_inputs, val_masks)
            print("Mean Intersection-Over-Union: {}".format(mean_IoU))

        # Save the model weights
        tf.keras.models.save_model(model, model_path)
        print("Model successfully saved!")

    else:  # mode == test
        test_inputs, test_masks = read_data(img_dir, mask_dir)
        # Load the model weights
        model = tf.keras.models.load_model(model_path)
        print("Model successfully loaded!")

        # Make predictions
        mean_IoU = test(model, test_inputs, test_masks)
        print("Test set IoU score: {}".format(mean_IoU))
        predictions = predict(model, test_inputs)

        # Visualize model performance
        vis_idx = random.randint(0, len(test_inputs) - 1)
        vis_list = [test_inputs[vis_idx], test_masks[vis_idx], predictions[vis_idx]]
        visualize(vis_list)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
