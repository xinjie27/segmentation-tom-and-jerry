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
        predictions = predict(model, batch_inputs)

        IoU.update_state(batch_masks, predictions)
        acc_IoU += IoU.result().numpy()
        IoU.reset_state()
        num_batches += 1

    mean_IoU = acc_IoU / num_batches

    return mean_IoU


def predict(model, inputs):
    """
    Makes predictions based on the model.

    :param model: a tf.keras.Model that has been trained
    :param inputs: 4-D Tensor of shape (num_examples, height, width, num_channels)
    :return: predictions, 4-D Tensor of shape (num_examples, height, width, 1)
    """
    probs = model(inputs)
    predictions = tf.where(probs < 0.5, 0, 1)
    return predictions
