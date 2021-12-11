import numpy as np
import tensorflow as tf


def train(model, train_inputs, train_masks):
    """
    Trains the model for one epoch.

    :param model: a tf.keras.Model instance
    :param train_inputs: 4-D Tensor of shape (num_inputs, height, width, num_channels)
    :param train_masks: 4-D Tensor of shape (num_inputs, height, width, 1)
    :return: None
    """
    num_inputs = train_inputs.shape[0]

    # Shuffle training inputs and masks
    indices = tf.random.shuffle(range(num_inputs))
    train_inputs = tf.gather(train_inputs, indices)
    train_masks = tf.gather(train_masks, indices)

    for i in range(0, num_inputs, model.batch_size):
        batch_inputs = train_inputs[i : i + model.batch_size, :, :, :]
        batch_masks = train_masks[i : i + model.batch_size, :, :, :]
        batch_inputs, batch_masks = _augment(batch_inputs, batch_masks)

        with tf.GradientTape() as tape:
            batch_pred = model(batch_inputs)
            loss = model.loss(batch_pred, batch_masks)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def _augment(images, masks):
    """
    Applies image augmentation on a pair of image batch and mask batch, including:
    1. Random flip horizontally
    2. Random flip vertically
    3. Random brightness
    4. Random hue

    :param images: 4-D Tensor of shape (height, width, num_channels, batch_size)
    :param masks: 4-D Tensor of shape (height, width, 1, batch_size)
    :return:
        images - a batch of randomly augmented images
        masks - a batch of randomly augmented masks
    """
    masks = tf.cast(masks, tf.float32)
    concat_batch = tf.concat([images, masks], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_batch)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

    images = maybe_flipped[:, :, :, :-1]
    masks = tf.cast(maybe_flipped[:, :, :, -1:], tf.int32)

    images = tf.image.random_brightness(images, max_delta=0.2)
    images = tf.image.random_hue(images, max_delta=0.1)

    return images, masks
