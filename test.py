from re import L
import numpy as np
import tensorflow as tf


def test(model, test_inputs, test_masks):
    """
    Tests the model on test images and masks to compute the mean IoU score.

    :param model: a tf.keras.Model instance that has been trained
    :param train_inputs: 4-D Tensor of shape (num_examples, height, width, num_channels)
    :param train_masks: 4-D Tensor of shape (num_examples, height, width, 1)
    :return: a float that ranges from 0 to 1
    """
    num_inputs = test_inputs.shape[0]

    num_batches = 0
    IoU = tf.keras.metrics.MeanIoU(num_classes=2)
    acc_IoU = 0

    for i in range(0, num_inputs, model.batch_size):
        batch_inputs = test_inputs[i : i + model.batch_size, :, :, :]
        batch_masks = test_masks[i : i + model.batch_size, :, :, :]
        probs = model(batch_inputs)
        predictions = tf.where(probs < 0.5, 0, 1)

        IoU.update_state(batch_masks, predictions)
        acc_IoU += IoU.result().numpy()
        IoU.reset_state()
        num_batches += 1

    mean_IoU = acc_IoU / num_batches

    return mean_IoU


def _DSC(y_pred, y_true, smooth=1e-4):
    """
    Computes the Dice similarity coefficient (F1 score).

    :param y_pred:
    :param y_true:
    :param smooth: a smoothing factor, 1e-4 by default
    :return: a float that ranges from 0 to 1
    """
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.reshape(y_true, [-1])

    intersection = tf.math.reduce_sum(y_pred_f * y_true_f)
    dice_coeff = (2.0 * intersection + smooth) / (
        tf.math.reduce_sum(y_pred_f) + tf.math.reduce_sum(y_true_f) + smooth
    )

    return dice_coeff


def _IoU(y_pred, y_true):
    """
    Computes the intersection over union (IoU).
    """
    y_pred_f = tf.reshape(y_pred, [-1])
    y_true_f = tf.reshape(y_true, [-1])

    intersection = tf.reduce_sum(y_pred_f * y_true_f)
    union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f) - intersection

    return intersection / union
