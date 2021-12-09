import numpy as np
import tensorflow as tf


def test(model, test_inputs, test_masks):
    """
    Tests the model on test images and masks to compute the IoU and Dice coefficient scores.

    :param model: a tf.keras.Model instance that has been trained
    :param train_inputs: 4-D Tensor of shape (num_examples, height, width, num_channels)
    :param train_masks: 4-D Tensor of shape (num_examples, height, width, 1)
    :return: average accuracy across all test examples
    """
    num_inputs = test_inputs.shape[0]

    for i in range(0, num_inputs, model.batch_size):
        batch_inputs = test_inputs[i: i + model.batch_size, :, :, :]
        batch_masks = test_masks[i: i + model.batch_size, :, :, :]
        predictions = model(batch_inputs)
        
        pass


def _IoU(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    return intersection / union

def _dice_coeff(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    pass