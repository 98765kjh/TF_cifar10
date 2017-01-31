import tensorflow as tf
import input_data

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)

training_epoch = 1
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

W1 = tf.get_variable("W1", shape=[28*28, 500], initializer=xavier_init(28*28, 500))
W2 = tf.get_variable("W2", shape=[500, 256], initializer=xavier_init(500, 256))
W3 = tf.get_variable("W3", shape=[256, 128], initializer=xavier_init(256, 128))
W4 = tf.get_variable("W4", shape=[128, 10], initializer=xavier_init(128, 10))

b1 = tf.Variable(tf.zeros([500]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([128]))
b4 = tf.Variable(tf.zeros([10]))

_L2 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L2 = tf.nn.dropout(_L2, dropout_rate)

_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W2), b2))
L3 = tf.nn.dropout(_L3, dropout_rate)

_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W3), b3))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypo = tf.add(tf.matmul(L4, W4), b4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypo, Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, "./save/10_save.ckpt")
    print ("Restore Finished!")

    correct_prediction = tf.equal(tf.argmax(hypo, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels, dropout_rate:1}))

