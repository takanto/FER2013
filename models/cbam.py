import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = layers.GlobalMaxPooling2D()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.se = tf.keras.Sequential([
            layers.Conv2D(channel // reduction, 1, use_bias=False),
            layers.ReLU(),
            layers.Conv2D(channel, 1, use_bias=False)
        ])
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(tf.expand_dims(max_result, axis=1))
        avg_out = self.se(tf.expand_dims(avg_result, axis=1))
        output = self.sigmoid(max_out + avg_out)
        return output
    
class EfficientChannelAttention(layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super(EfficientChannelAttention, self).__init__(**kwargs)
        self.k_size = k_size

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.conv = layers.Conv1D(filters=1, kernel_size=self.k_size, padding='same', kernel_initializer='glorot_uniform', use_bias=False,)

    def call(self, x):
        _, H, W, C = x.shape
        squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        attn = self.conv(squeeze)
        attn = tf.squeeze(attn, axis=1)
        attn = tf.math.sigmoid(attn)
        attn = tf.expand_dims(attn, axis=1)
        x =  x * attn
        x = tf.reshape(x, (-1, H*W, C))
        return x


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size, padding='same')

    def call(self, x):
        max_result = tf.reduce_max(x, axis=3, keepdims=True)
        avg_result = tf.reduce_mean(x, axis=3, keepdims=True)
        result = tf.concat([max_result, avg_result], axis=3)
        output = self.conv(result)
        output = layers.Activation('sigmoid')(output)
        return output


class ChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super(ChannelBlockAttentionBlock, self).__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        x_skip = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x_skip
    
class EfficientChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, kernel_size=49):
        super(EfficientChannelBlockAttentionBlock, self).__init__()
        self.ca = EfficientChannelAttention()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        x_skip = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x_skip