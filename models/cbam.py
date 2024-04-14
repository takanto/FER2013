import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.saving.register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.num_channels = num_channels
        self.conv = tf.keras.layers.Conv2D(num_channels//ratio, kernel_size=3, padding='same', use_bias=False)
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_in = self.avg_pool(x)[:, tf.newaxis, tf.newaxis, :]
        avg_in = self.conv(avg_in)
        avg_in = self.relu(avg_in)
        avg_out = self.conv2(avg_in)

        max_in = self.max_pool(x)[:, tf.newaxis, tf.newaxis, :]
        max_in = self.conv(max_in)
        max_in = self.relu(max_in)
        max_out = self.conv2(max_in)

        out = avg_out + max_out
        attn = self.sigmoid(out)
        return x*attn
    
    def get_config(self):
        return {
            'avg_pool': self.avg_pool,
            'max_pool': self.max_pool,
            'conv': self.conv,
            'relu': self.relu,
            'conv2': self.conv2,
            'sigmoid': self.sigmoid
        }
    
@tf.keras.saving.register_keras_serializable()
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
    
    def get_config(self):
        return {
            'conv': self.conv,
        }

@tf.keras.saving.register_keras_serializable()
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
    
    def get_config(self):
        return {
            'conv': self.conv,
            'bn': self.bn,
            'relu': self.relu
        }

@tf.keras.saving.register_keras_serializable()
class ChannelPool(layers.Layer):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def call(self, x):
        max_pool = tf.reduce_max(x, axis=3, keepdims=True)
        mean_pool = tf.reduce_mean(x, axis=3, keepdims=True)
        return tf.concat([max_pool, mean_pool], axis=3)


@tf.keras.saving.register_keras_serializable()
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
    
    def get_config(self):
        return {
            'compress': self.compress,
            'spatial': self.spatial,
        }

@tf.keras.saving.register_keras_serializable()
class ChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, num_channels=512, reduction=16, kernel_size=49):
        super(ChannelBlockAttentionBlock, self).__init__()
        self.num_channels = num_channels
        self.reduction = reduction
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.ca = ChannelAttention(num_channels=self.num_channels, ratio=self.reduction)
        self.sa = SpatialAttention(kernel_size=self.kernel_size)

    def call(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out
    
    def get_config(self):
        return {
            'ca': self.ca,
            'sa': self.sa,
        }
    
    
@tf.keras.saving.register_keras_serializable()
class EfficientChannelBlockAttentionBlock(layers.Layer):
    def __init__(self, kernel_size=49):
        super(EfficientChannelBlockAttentionBlock, self).__init__()
        self.ca = EfficientChannelAttention()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out
    
    def get_config(self):
        return {
            'ca': self.ca,
            'sa': self.sa,
        }