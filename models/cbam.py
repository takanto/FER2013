import tensorflow as tf
from tensorflow.keras import layers

class ChannelAttention(layers.Layer):
    def __init__(self, num_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.mlp = tf.keras.Sequential([
            layers.Dense(num_channels // self.reduction_ratio, activation='relu'),
            layers.Dense(num_channels, activation='sigmoid')
        ])

    def call(self, x):
        _, H, W, C = x.shape
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
                #avg_pool = tf.reshape(avg_pool, (tf.shape(avg_pool)[0], -1))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
                #max_pool = tf.reshape(max_pool, (tf.shape(max_pool)[0], -1))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = tf.expand_dims(tf.expand_dims(channel_att_sum, axis=1), axis=1)
        x = x * scale
        x = tf.reshape(x, (-1, H, W, C))
        return x
    
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
        return x

class BasicConv(layers.Layer):
    def __init__(self, out_planes, kernel_size, stride=1, padding='same', dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = layers.Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=dilation, groups=groups, use_bias=bias)
        self.bn = layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if bn else None
        self.relu = layers.ReLU() if relu else None

    def call(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(layers.Layer):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def call(self, x):
        max_pool = tf.reduce_max(x, axis=3, keepdims=True)
        mean_pool = tf.reduce_mean(x, axis=3, keepdims=True)
        return tf.concat([max_pool, mean_pool], axis=3)


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, kernel_size, padding='same', relu=False)

    def call(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = tf.sigmoid(x_out)
        x = x * scale
        return x


class ChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, num_channels=512, reduction=16, kernel_size=49):
        super(ChannelBlockAttentionBlock, self).__init__()
        self.num_channels = num_channels
        self.reduction = reduction
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.ca = ChannelAttention(num_channels=self.num_channels, reduction_ratio=self.reduction)
        self.sa = SpatialAttention(kernel_size=self.kernel_size)

    def call(self, x):
        x_skip = x
        out = self.ca(x)
        out = self.sa(out)
        return out + x_skip
    
class EfficientChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, kernel_size=49):
        super(EfficientChannelBlockAttentionBlock, self).__init__()
        self.ca = EfficientChannelAttention()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        x_skip = x
        out = self.ca(x)
        out = self.sa(out)
        return out + x_skip