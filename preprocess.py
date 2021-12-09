import numpy as np
import glob
from PIL import Image


def read_data(img_dir, mask_dir):
    """
    Reads in data from given file paths.

    :param img_dir:
    :param mask_dir:
    :return:
    """
    img_list = glob.glob(img_dir + "/*.jpg")
    mask_list = glob.glob(mask_dir + "*.jpg")

    images = np.array([np.array(Image.open(fname)) for fname in img_list])
    masks = np.array([np.load(fname) for fname in mask_list])
    # Normalize the pixel values
    images /= 255.

    return images, masks
