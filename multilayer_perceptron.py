import tensorflow as tf
import math
tf.set_random_seed(0)


# network Parameters
n_hidden = [10000, 2500, 500, 100]  # layers and their number of neurons
img_size = 128                      # input image shape: 128*128
n_input = img_size**2               # data input: 16384
n_classes = 50                      # total classes (bird species)

# weights initialised with small random values between -0.2 and +0.2
W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden[0]], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([n_hidden[0], n_hidden[1]], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([n_hidden[1], n_hidden[2]], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([n_hidden[2], n_hidden[3]], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([n_hidden[3], n_classes], stddev=0.1))

# biases initialised with small *positive* values (0.1)
B1 = tf.Variable(tf.ones([n_hidden[0]]) / 10)
B2 = tf.Variable(tf.ones([n_hidden[1]]) / 10)
B3 = tf.Variable(tf.ones([n_hidden[2]]) / 10)
B4 = tf.Variable(tf.ones([n_hidden[3]]) / 10)
B5 = tf.Variable(tf.zeros([n_classes]))

# input X: 128x128 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, img_size, img_size, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)
# probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)
# step for variable learning rate
step = tf.placeholder(tf.int32)

# model with dropout at each layer
XX = tf.reshape(X, [-1, n_input])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)
Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Y = tf.nn.softmax(tf.matmul(Y4d, W5) + B5)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, exponential decay from 0.003->0.0001
lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1 / math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


def training(data):
    # init session
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    # TODO ustalić ile iteracji
    for i in range(10000):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = data # TODO tutaj ma pobierać 100 (?) obrazów uczących
        train_data = {X: batch_X, Y_: batch_Y, pkeep: 0.75, step: i}

        # the backpropagation training step
        sess.run(train_step, train_data)

