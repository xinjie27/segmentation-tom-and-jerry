import numpy as np
import glob
from PIL import Image


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
        [np.array(Image.open(fname).resize([128, 128])) for fname in img_list]
    ).astype(np.float32)
    masks = np.array(
        [
            np.array(Image.fromarray(np.load(fname)).resize([128, 128]))
            for fname in mask_list
        ]
    )

    masks = np.expand_dims(masks, axis=-1)

    # Normalize the pixel values
    images /= 255.0

    return images[:40], masks[:40]
