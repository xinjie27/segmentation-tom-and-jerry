import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Model, layers


class ConvBlock(Model):
    """
    This class contains two 3x3 convolution layers, each followed by batch
    normalization and ReLU activation function.
    (Conv2D -> BN -> ReLU) * 2

    :param num_filters: number of output feature channels
    """

    def __init__(self, num_filters):
        super(ConvBlock, self).__init__()
        self.kernel_size = 3
        self.conv1 = layers.Conv2D(num_filters, self.kernel_size, padding="same")
        self.conv2 = layers.Conv2D(num_filters, self.kernel_size, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")

    @tf.function
    def call(self, inputs):
        """
        Runs a forward pass.

        :param inputs: input batch of feature maps
        :return: output feature maps
        """
        conv1_out = self.conv1(inputs)
        bn1_out = self.bn(conv1_out)
        act1_out = self.act(bn1_out)
        conv2_out = self.conv2(act1_out)
        bn2_out = self.bn(conv2_out)
        act2_out = self.act(bn2_out)
        return act2_out


class EncoderBlock(Model):
    """
    The encoder block (contracting part) consists of a ConvBlock and a 2D max
    pooling layer.

    :param num_filters: number of output feature channels
    """

    def __init__(self, num_filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(num_filters)
        self.pool = layers.MaxPool2D((2, 2))

    @tf.function
    def call(self, inputs):
        """
        Runs a forward pass.

        :param inputs: input batch of feature maps
        :return:
        conv_out - output of ConvBlock, used in skip connection
        pool_out - reduced feature maps passed to the next block as input
        """
        conv_out = self.conv_block(inputs)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class DecoderBlock(Model):
    """
    The decoder block (expansive part) consists of a transposed convolution
    layer, a skip connection, and a ConvBlock.

    :param num_filters: number of output feature channels
    """

    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        self.conv_t = layers.Conv2DTranspose(num_filters, 2, strides=2, padding="same")
        self.concat = layers.Concatenate()
        self.conv_block = ConvBlock(num_filters)

    @tf.function
    def call(self, inputs, skip_features):
        """
        Runs a forward pass.

        :param inputs: input batch of feature maps
        :param skip_features: the first output of the encoder block that are
            fetched through the skip connection
        :return: output feature maps
        """
        ("here at call start")
        conv_t_out = self.conv_t(inputs)
        ("here at convT end")
        concat_out = self.concat([skip_features, conv_t_out])
        conv_out = self.conv_block(concat_out)
        return conv_out


class UNet(Model):
    """
    This class implements the U-Net architecture.
    """

    def __init__(self, cfg):
        super(UNet, self).__init__()

        # Hyperparameters
        self.lr = cfg["learning_rate"]
        self.batch_size = cfg["batch_size"]
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Model architecture
        self.input_layer = layers.InputLayer(input_shape=(128, 128, 3))
        self.encoder1 = EncoderBlock(64)
        self.encoder2 = EncoderBlock(128)
        self.encoder3 = EncoderBlock(256)
        self.encoder4 = EncoderBlock(512)
        self.bridge_conv = ConvBlock(1024)
        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64)
        self.out_conv = layers.Conv2D(1, 1, padding="same", activation="sigmoid")

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        s1, x1 = self.encoder1(x)
        s2, x2 = self.encoder2(x1)
        s3, x3 = self.encoder3(x2)
        s4, x4 = self.encoder4(x3)

        x5 = self.bridge_conv(x4)

        d1 = self.decoder1(x5, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        probs = self.out_conv(d4)
        return probs

    def loss(self, probs, masks):
        loss = tf.keras.losses.SparseCategoricalCrossentropy
        return loss(masks, probs)
        # return tf.keras.metrics.binary_crossentropy(masks, probs)
