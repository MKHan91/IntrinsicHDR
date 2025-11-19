import tensorflow as tf

class DequantizationNet(tf.keras.Model):
    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train

        # Encoder
        self.conv1 = tf.keras.layers.Conv2D(16, 7, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(16, 7, padding='same')

        self.down1_conv1 = tf.keras.layers.Conv2D(32, 5, padding='same')
        self.down1_conv2 = tf.keras.layers.Conv2D(32, 5, padding='same')
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.down2_conv1 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.down2_conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.down3_conv1 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.down3_conv2 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.pool3 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        self.down4_conv1 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.down4_conv2 = tf.keras.layers.Conv2D(256, 3, padding='same')
        self.pool4 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

        # Decoder
        self.up1_upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up1_conv1 = tf.keras.layers.Conv2D(128, 3, padding='same')
        self.up1_conv2 = tf.keras.layers.Conv2D(128, 3, padding='same')

        self.up2_upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up2_conv1 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.up2_conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')

        self.up3_upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up3_conv1 = tf.keras.layers.Conv2D(32, 3, padding='same')
        self.up3_conv2 = tf.keras.layers.Conv2D(32, 3, padding='same')

        self.up4_upsample = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up4_conv1 = tf.keras.layers.Conv2D(16, 3, padding='same')
        self.up4_conv2 = tf.keras.layers.Conv2D(16, 3, padding='same')

        # Output
        self.out_conv = tf.keras.layers.Conv2D(3, 3, padding='same', activation='tanh')

    def _down(self, x, conv1, conv2, pool):
        x = pool(x)
        x = tf.nn.leaky_relu(conv1(x), alpha=0.1)
        x = tf.nn.leaky_relu(conv2(x), alpha=0.1)
        return x

    def _up(self, x, skip, upsample, conv1, conv2):
        x = upsample(x)
        x = tf.nn.leaky_relu(conv1(x), alpha=0.1)
        x = tf.nn.leaky_relu(conv2(tf.concat([x, skip], axis=-1)), alpha=0.1)
        return x

    def call(self, inputs, training=False):
        # Encoder
        x = tf.nn.leaky_relu(self.conv1(inputs), alpha=0.1)
        s1 = tf.nn.leaky_relu(self.conv2(x), alpha=0.1)
        s2 = self._down(s1, self.down1_conv1, self.down1_conv2, self.pool1)
        s3 = self._down(s2, self.down2_conv1, self.down2_conv2, self.pool2)
        s4 = self._down(s3, self.down3_conv1, self.down3_conv2, self.pool3)
        x = self._down(s4, self.down4_conv1, self.down4_conv2, self.pool4)

        # Decoder
        x = self._up(x, s4, self.up1_upsample, self.up1_conv1, self.up1_conv2)
        x = self._up(x, s3, self.up2_upsample, self.up2_conv1, self.up2_conv2)
        x = self._up(x, s2, self.up3_upsample, self.up3_conv1, self.up3_conv2)
        x = self._up(x, s1, self.up4_upsample, self.up4_conv1, self.up4_conv2)

        output = inputs + self.out_conv(x)
        return output
