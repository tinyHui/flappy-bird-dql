import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def build_network():
    # initial input weights and bias
    W1_conv = weight_variable([8, 8, 4, 32])
    b1_conv = bias_variable([32])

    W2_conv = weight_variable([4, 4, 32, 64])
    b2_conv = bias_variable([64])

    W3_conv = weight_variable([3, 3, 64, 64])
    b3_conv = bias_variable([64])

    W1_fc = weight_variable([256, 256])
    b1_fc = bias_variable([256])

    W2_fc = weight_variable([256, 2])
    b2_fc = bias_variable([2])

    # input layer
    input_placeholder = tf.placeholder("float", [None, 80, 80, 4])

    h1_conv = tf.nn.relu(conv2d(input_placeholder, W1_conv, 4) + b1_conv)
    h1_pool = max_pool_2x2(h1_conv)

    h2_conv = tf.nn.relu(conv2d(h1_pool, W2_conv, 2) + b2_conv)
    h2_pool = max_pool_2x2(h2_conv)

    h3_conv = tf.nn.relu(conv2d(h2_pool, W3_conv, 1) + b3_conv)
    h3_pool = max_pool_2x2(h3_conv)

    # fully connected ReLU layer
    h3_pool_flat = tf.reshape(h3_pool, [-1, 256])
    h_fc = tf.nn.relu(tf.matmul(h3_pool_flat, W1_fc) + b1_fc)

    # matrix multiplier to get final action
    output_data = tf.matmul(h_fc, W2_fc) + b2_fc

    return input_placeholder, output_data
