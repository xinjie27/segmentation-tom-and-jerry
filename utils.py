import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def read_data(img_dir, mask_dir):
    """
    Reads in data from given file paths.

    :param img_dir: directory containing all image files (in .jpg format)
    :param mask_dir: directory containing all mask files (in .npy format)
    :return:
        images - 4-D array of shape (num_inputs, height, width, num_channels)
        masks - 4-D array of shape (num_inputs, height, width, 1)
    """
    img_list = glob.glob(img_dir + "/*.jpg")
    mask_list = glob.glob(mask_dir + "/*.npy")

    images = np.array(
        [np.array(Image.open(fname).resize([256, 192])) for fname in img_list]
    ).astype(np.float32)
    masks = np.array(
        [
            np.array(Image.fromarray(np.load(fname)).resize([256, 192]))
            for fname in mask_list
        ]
    )
    masks = np.expand_dims(masks, axis=-1)

    # Normalize the pixel values
    images /= 255.0

    return images, masks


def visualize(vis_list):
    """
    Visualizes the segmentation result.

    :param vis_list: either [image, true_mask] or [image, true_mask, pred_mask]
    :return: None
    """
    plt.figure(figsize=(15, 15))
    titles = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(vis_list)):
        plt.subplot(1, len(vis_list), i + 1)
        plt.title(titles[i])
        plt.imshow(tf.keras.utils.array_to_img(vis_list[i]))
        plt.axis("off")
    plt.show()
