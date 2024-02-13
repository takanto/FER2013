import tensorflow as tf
from tensorflow.keras import layers

class DistanceGraphConvolution(layers.Layer):
    def __init__(self, num_features, num_filters):
        super(DistanceGraphConvolution, self).__init__()
        self.num_features = num_features
        self.num_filters = num_filters
        self.W = self.add_weight(shape=(num_features, num_filters),
                                       initializer='glorot_uniform',
                                       trainable=True)

    def call(self, x, distance_matrix):
        # Inputs: [batch_size, num_nodes, num_features]
        # Adjacency matrix: [batch_size, num_nodes, num_nodes]

        # Perform graph convolution
        A = tf.exp(-0.2*distance_matrix)
        D = tf.reduce_sum(A, axis=-1)
        D = tf.linalg.diag(D)
        D_inv_sqrt = tf.linalg.inv(tf.sqrt(D))
        A = tf.matmul(tf.matmul(D_inv_sqrt, A), D_inv_sqrt)
        x = tf.matmul(A, x)
        x = tf.matmul(x, self.W)
        return x
    
class GraphConvolution(layers.Layer):
    def __init__(self, num_features, num_filters, threshold):
        super(GraphConvolution, self).__init__()
        self.num_features = num_features
        self.num_filters = num_filters
        self.threshold = threshold
        self.W = self.add_weight(shape=(num_features, num_filters),
                                       initializer='glorot_uniform',
                                       trainable=True)

    def call(self, x, distance_matrix):
        A = tf.cast(distance_matrix<self.threshold, tf.float32)
        D = tf.reduce_sum(A, axis=-1)
        D = tf.linalg.diag(D)
        D_inv_sqrt = tf.linalg.inv(tf.sqrt(D))
        A = tf.matmul(tf.matmul(D_inv_sqrt, A), D_inv_sqrt)
        x = tf.matmul(A, x)
        x = tf.matmul(x, self.W)
        return x