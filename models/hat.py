import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

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
        x = tf.reshape(x, (-1, H*W, C))
        return x
    
    def get_config(self):
        return {
            'conv': self.conv,
        }
    
@tf.keras.saving.register_keras_serializable()
class ChannelAttention(layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.k_size = k_size

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.conv1 = layers.Conv1D(filters=1, kernel_size=self.k_size, padding='same', kernel_initializer='glorot_uniform', use_bias=False,)
        self.conv2 = layers.Conv1D(filters=1, kernel_size=self.k_size, padding='same', kernel_initializer='glorot_uniform', use_bias=False,)
    
    def call(self, x):
        _, H, W, C = x.shape
        squeeze = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        attn = self.conv1(squeeze)
        attn = tf.squeeze(attn, axis=1)
        attn = tf.math.sigmoid(attn)
        attn = tf.expand_dims(attn, axis=1)
        attn = self.conv2(attn)
        x =  x * attn
        x = tf.reshape(x, (-1, H*W, C))
        return x
    
    def get_config(self):
        return {
            'conv1': self.conv1,
            'conv2': self.conv2,
        }
    

@tf.keras.saving.register_keras_serializable()
class WindowbasedMultiheadedSelfAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super(WindowbasedMultiheadedSelfAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            name='wattn_bias',
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            shape=relative_position_index.shape,
            dtype=tf.int32,
            trainable=False,
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat, axis=0,)
        
        relative_position_bias = tf.reshape(relative_position_bias, (num_window_elements, num_window_elements, -1),)
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), dtype=tf.float32)
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
            attn = tf.keras.activations.softmax(attn, axis=-1)
        else:
            attn = tf.keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
    
    def get_config(self):
        return {
            'qkv': self.qkv,
            'proj': self.proj,
            'relative_position_bias_table': self.relative_position_bias_table.numpy(),
            'relative_position_index': self.relative_position_index.numpy()
        }
    


@tf.keras.saving.register_keras_serializable()
class OverlapCrossAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, overlap_ratio, qkv_bias=True, dropout_rate=0.0, **kwargs,):
        super(OverlapCrossAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.overlap_ratio = overlap_ratio
        self.q = layers.Dense(dim, use_bias=qkv_bias)
        self.kv = layers.Dense(dim*2, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)
        
        self.M = window_size
        self.Mo = int((1 + overlap_ratio) * window_size)

        num_window_elements = ((self.M+self.Mo) - 2) * (2 * self.Mo) + 1
        self.relative_position_bias_table = self.add_weight(
            name='oca_bias',
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(self.M)
        coords_w = np.arange(self.Mo)
        coords_matrix_h = np.meshgrid(coords_h, coords_h, indexing="ij")
        coords_matrix_w = np.meshgrid(coords_w, coords_w, indexing="ij")
        coords_h = np.stack(coords_matrix_h)
        coords_w = np.stack(coords_matrix_w)
        coords_flatten_h = coords_h.reshape(2, -1)
        coords_flatten_w = coords_w.reshape(2, -1)
        relative_coords = coords_flatten_h[:, :, None] - coords_flatten_w[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.Mo - 1
        relative_coords[:, :, 1] += self.Mo - 1
        relative_coords[:, :, 0] *= 2 * self.Mo - 1
        relative_position_index = relative_coords.sum(-1)
        

        self.relative_position_index = tf.Variable(
            initial_value=relative_position_index,
            shape=relative_position_index.shape,
            dtype=tf.int32,
            trainable=False,
        )
        
    def call(self, q_w, kv_w):
        _, q_size, q_channels = q_w.shape
        _, kv_size, kv_channels = kv_w.shape
        head_dim_q = q_channels // self.num_heads
        head_dim_kv = kv_channels // self.num_heads

        x_q = self.q(q_w)
        x_q = tf.reshape(x_q, (-1, q_size, self.num_heads, head_dim_q))
        q = tf.transpose(x_q, (0, 2, 1, 3))
        x_kv = self.kv(kv_w)
        x_kv = tf.reshape(x_kv, (-1, kv_size, 2, self.num_heads, head_dim_kv))
        x_kv = tf.transpose(x_kv, (2, 0, 3, 1, 4))
        k, v = x_kv[0], x_kv[1]
        q = q * self.scale
        k = tf.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.M * self.Mo
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat, axis=0,)
        relative_position_bias = tf.reshape(relative_position_bias, (self.M**2, self.Mo**2, -1),)
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        attn = tf.keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, (-1, q_size, q_channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv
    
    def get_config(self):
        return {
            'q': self.q,
            'kv': self.kv,
            'proj': self.proj,
            'relative_position_bias_table': self.relative_position_bias_table.numpy(),
            'relative_position_index': self.relative_position_index.numpy()
        }
    

@tf.keras.saving.register_keras_serializable()
class EfficientHybridAttentionBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, qkv_bias=True, dropout_rate=0.0,**kwargs,):
        super(EfficientHybridAttentionBlock, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.eca = EfficientChannelAttention()
        self.attn = WindowbasedMultiheadedSelfAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, height, width, channels))
        cattn = self.eca(x)
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size, self.window_size)
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        cattn = self.drop_path(cattn, training=training)
        x = x_skip + x + cattn
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
    def get_config(self):
        return {
            'eca': self.eca,
            'attn': self.attn,
            'mlp': self.mlp,
            'attn_mask': self.attn_mask.numpy()
        }
    
@tf.keras.saving.register_keras_serializable()
class HybridAttentionBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, qkv_bias=True, dropout_rate=0.0,**kwargs,):
        super(HybridAttentionBlock, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.eca = ChannelAttention()
        self.attn = WindowbasedMultiheadedSelfAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                shape=attn_mask.shape,
                dtype=attn_mask.dtype,
                trainable=False,
            )

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, height, width, channels))
        cattn = self.eca(x)
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size, self.window_size)
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        cattn = self.drop_path(cattn, training=training)
        x = x_skip + x + cattn
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
    def get_config(self):
        return {
            'eca': self.eca,
            'attn': self.attn,
            'mlp': self.mlp,
            'attn_mask': self.attn_mask.numpy()
        }

@tf.keras.saving.register_keras_serializable()
class OverlapCrossAttentionBlock(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, overlap_ratio=0, num_mlp=1024, qkv_bias=True, dropout_rate=0.0,**kwargs,):
        super(OverlapCrossAttentionBlock, self).__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.num_mlp = num_mlp

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = OverlapCrossAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            overlap_ratio=overlap_ratio,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(tf.keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.overlap_ratio = 0
            self.window_size = min(self.num_patch)

    def call(self, x, training=False):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, height, width, channels))

        Mo = int((1+self.overlap_ratio)*self.window_size)
        pad_size = int(self.overlap_ratio*self.window_size / 2)
        x_pad = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        q_windows = window_partition(x, self.window_size, self.window_size)
        kv_windows = window_partition(x_pad, Mo, self.window_size)
        q_windows = tf.reshape(
            q_windows, (-1, self.window_size * self.window_size, channels)
        )
        kv_windows = tf.reshape(
            kv_windows, (-1, Mo * Mo, channels)
        )
        attn_windows = self.attn(q_windows, kv_windows)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels),
        )
        x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x
    
    def get_config(self):
        return {
            'attn': self.attn,
            'mlp': self.mlp,
        }


@tf.keras.saving.register_keras_serializable()
class EfficientResidualHybridAttentionGroup(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=1, overlap_ratio=0, num_mlp=1024, hab_num=3, qkv_bias=True, dropout_rate=0.0,**kwargs,):
        super(EfficientResidualHybridAttentionGroup, self).__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.overlap_ratio = overlap_ratio
        self.num_mlp = num_mlp
        self.hab_num = hab_num

        self.hab = EfficientHybridAttentionBlock(dim=dim, num_patch=num_patch, num_heads=num_heads, window_size=window_size, 
                                        shift_size=shift_size, num_mlp=num_mlp, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.ocab = OverlapCrossAttentionBlock(dim=dim, num_patch=num_patch, num_heads=num_heads, window_size=window_size, 
                                               overlap_ratio=overlap_ratio, num_mlp=num_mlp, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.conv = layers.Conv2D(filters=dim, kernel_size=(3,3), padding='same', kernel_initializer='glorot_uniform', use_bias=False,)

    def call(self, x):
        h, w = self.num_patch
        _, size, channels = x.shape
        x_skip = x

        for _ in range(self.hab_num):
          x = self.hab(x)

        x = self.ocab(x)

        x = tf.reshape(x, (-1, h, w, channels))
        x = self.conv(x)
        x = tf.reshape(x, (-1, size, channels))
        
        x = x_skip + x
        return x
    
    def get_config(self):
        return {
            'hab': self.hab,
            'ocab': self.ocab,
            'conv': self.conv
        }
    

@tf.keras.saving.register_keras_serializable()
class ResidualHybridAttentionGroup(layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=1, overlap_ratio=0, num_mlp=1024, hab_num=3, qkv_bias=True, dropout_rate=0.0,**kwargs,):
        super(ResidualHybridAttentionGroup, self).__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.overlap_ratio = overlap_ratio
        self.num_mlp = num_mlp
        self.hab_num = hab_num

        self.hab = HybridAttentionBlock(dim=dim, num_patch=num_patch, num_heads=num_heads, window_size=window_size, 
                                        shift_size=shift_size, num_mlp=num_mlp, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.ocab = OverlapCrossAttentionBlock(dim=dim, num_patch=num_patch, num_heads=num_heads, window_size=window_size, 
                                               overlap_ratio=overlap_ratio, num_mlp=num_mlp, qkv_bias=qkv_bias, dropout_rate=dropout_rate)
        self.conv = layers.Conv2D(filters=dim, kernel_size=(3,3), padding='same', kernel_initializer='glorot_uniform', use_bias=False,)

    def call(self, x):
        h, w = self.num_patch
        _, size, channels = x.shape
        x_skip = x

        for _ in range(self.hab_num):
          x = self.hab(x)

        x = self.ocab(x)

        x = tf.reshape(x, (-1, h, w, channels))
        x = self.conv(x)
        x = tf.reshape(x, (-1, size, channels))
        
        x = x_skip + x
        return x
    
    def get_config(self):
        return {
            'hab': self.hab,
            'ocab': self.ocab,
            'conv': self.conv
        }



def window_partition(x, window_size, stride):
    batch_size, height, width, channels = x.shape
    patch_num_y = (height - window_size) // stride + 1
    patch_num_x = (width - window_size) // stride + 1
    
    windows = []
    for i in range(patch_num_y):
        for j in range(patch_num_x):
            window = x[:, i * stride:i * stride + window_size, j * stride:j * stride + window_size, :]
            windows.append(window)
    
    windows = tf.concat(windows, axis=0)
    return windows




def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,(-1, patch_num_y, patch_num_x, window_size, window_size, channels,),)
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (-1, height, width, channels))
    return x



def patch_extract(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=(1, patch_size[0], patch_size[1], 1),
        strides=(1, patch_size[0], patch_size[1], 1),
        rates=(1, 1, 1, 1),
        padding="VALID",
    )
    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


@tf.keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch)
        return self.proj(patch) + self.pos_embed(pos)
    
    def get_config(self):
        return {
            'proj': self.proj,
            'pos_embed': self.pos_embed,
        }



@tf.keras.saving.register_keras_serializable()
class PatchMerging(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = tf.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, (-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)
    
    def get_config(self):
        return {
            'linear_trans': self.linear_trans,
        }