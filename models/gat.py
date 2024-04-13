import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.saving.register_keras_serializable()
class GraphAttention(layers.Layer):
    def __init__(self, num_features, num_filters, in_drop, coef_drop, residual=False):
        super(GraphAttention, self).__init__()
        self.num_features = num_features
        self.num_filters = num_filters
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        

    def build(self, input_shape):
        self.conv1d = layers.Conv1D(self.num_filters, kernel_size=1, use_bias=False)
        self.a = layers.Conv1D(1, kernel_size=1)
        self.leaky_relu = layers.LeakyReLU(0.2)
        self.bias = tf.Variable(initial_value=tf.zeros(shape=(self.num_filters)))

    def call(self, x, A):
        if self.in_drop != 0.0:
            x = tf.nn.dropout(x, 1.0 - self.in_drop)

        x_skip = x

        x = self.conv1d(x)
        x_i = self.a(x)
        x_j = self.a(x)
        x_ij = x_i+tf.transpose(x_j, [0,2,1])
        e = self.leaky_relu(x_ij)
        a = tf.nn.softmax(e+A)

        if self.coef_drop != 0.0:
            a = tf.nn.dropout(a, 1.0 - self.coef_drop)
        if self.in_drop != 0.0:
            x = tf.nn.dropout(x, 1.0 - self.in_drop)
        
        x = tf.matmul(a, x)
        x = x + self.bias
        
        if self.residual:
            if x_skip.shape[-1] != x.shape[-1]:
                x = x + self.conv1d(x_skip)
            else:
                x = x + x_skip
        return x
    
    def get_config(self):
        return {
            'conv1d': self.conv1d,
            'a': self.a,
            'leaky_relu': self.leaky_relu,
            "bias": self.bias.numpy(),
        }
    
@tf.keras.saving.register_keras_serializable()
class GraphAttentionV2(layers.Layer):
    def __init__(self, num_features, num_filters, in_drop, coef_drop, residual=False):
        super(GraphAttentionV2, self).__init__()
        self.num_features = num_features
        self.num_filters = num_filters
        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.residual = residual
        

    def build(self, input_shape):
        self.conv1d = layers.Conv1D(self.num_filters, kernel_size=1, use_bias=False)
        self.a = layers.Conv1D(1, kernel_size=1)
        self.leaky_relu = layers.LeakyReLU(0.2)
        self.bias = tf.Variable(initial_value=tf.zeros(shape=(self.num_filters)))

    def call(self, x, A):
        if self.in_drop != 0.0:
            x = tf.nn.dropout(x, 1.0 - self.in_drop)

        x_skip = x

        x = self.conv1d(x)
        x = self.leaky_relu(x)
        x_i = self.a(x)
        x_j = self.a(x)
        x_ij = x_i+tf.transpose(x_j, [0,2,1])
        a = tf.nn.softmax(x_ij+A)

        if self.coef_drop != 0.0:
            a = tf.nn.dropout(a, 1.0 - self.coef_drop)
        if self.in_drop != 0.0:
            x = tf.nn.dropout(x, 1.0 - self.in_drop)
        
        x = tf.matmul(a, x)
        x = x + self.bias
        
        if self.residual:
            if x_skip.shape[-1] != x.shape[-1]:
                x = x + self.conv1d(x_skip)
            else:
                x = x + x_skip
        return x
    
    def get_config(self):
        return {
            'conv1d': self.conv1d,
            'a': self.a,
            'leaky_relu': self.leaky_relu,
            "bias": self.bias.numpy(),
        }
    
@tf.keras.saving.register_keras_serializable()
class DiscreteMultiHeadGraphAttention(layers.Layer):
    def __init__(self, num_heads, num_features, num_filters, threshold, in_drop, coef_drop, residual=False):
        super(DiscreteMultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.threshold = threshold
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttention(num_features, num_filters, in_drop, coef_drop, residual))

    def call(self, x, distance_matrix):
        A = tf.cast(distance_matrix<self.threshold, tf.float32)
        head_outputs = [head(x, A) for head in self.attention_heads]
        return tf.reduce_mean(head_outputs, axis=0)
    
    def get_config(self):
        return {
            'attention_heads': self.attention_heads,
        }

@tf.keras.saving.register_keras_serializable() 
class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, num_heads, num_features, num_filters, threshold, in_drop, coef_drop, residual=False):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.threshold = threshold
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttention(num_features, num_filters, in_drop, coef_drop, residual))

    def call(self, x, distance_matrix):
        A = distance_matrix
        head_outputs = [head(x, A) for head in self.attention_heads]
        return tf.reduce_mean(head_outputs, axis=0)
    
    def get_config(self):
        return {
            'attention_heads': self.attention_heads,
        }
    
@tf.keras.saving.register_keras_serializable()
class DiscreteMultiHeadGraphAttentionV2(layers.Layer):
    def __init__(self, num_heads, num_features, num_filters, threshold, in_drop, coef_drop, residual=False):
        super(DiscreteMultiHeadGraphAttentionV2, self).__init__()
        self.num_heads = num_heads
        self.threshold = threshold
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttentionV2(num_features, num_filters, in_drop, coef_drop, residual))

    def call(self, x, distance_matrix):
        A = tf.cast(distance_matrix<self.threshold, tf.float32)
        head_outputs = [head(x, A) for head in self.attention_heads]
        return tf.reduce_mean(head_outputs, axis=0)
    
    def get_config(self):
        return {
            'attention_heads': self.attention_heads,
        }
    
@tf.keras.saving.register_keras_serializable()
class MultiHeadGraphAttentionV2(layers.Layer):
    def __init__(self, num_heads, num_features, num_filters, in_drop, coef_drop, residual=False):
        super(MultiHeadGraphAttentionV2, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = []
        for _ in range(num_heads):
            self.attention_heads.append(GraphAttentionV2(num_features, num_filters, in_drop, coef_drop, residual))

    def call(self, x, distance_matrix):
        A = distance_matrix
        head_outputs = [head(x, A) for head in self.attention_heads]
        return tf.reduce_mean(head_outputs, axis=0)
    
    def get_config(self):
        return {
            'attention_heads': self.attention_heads,
        }