import tensorflow as tf


class PerceptronModel:
    def __init__(self, img_features, labels, learning_rate, is_training, config):
        self._is_training = is_training
        self._config = config

        self._summaries = []
        self._valid_summaries = []

        initializer = tf.truncated_normal_initializer(0, self._config.weights_init_stddev)

        with tf.variable_scope('Model', initializer=initializer):
            self._model_output, summaries = self._create_model(img_features)
            self._summaries.extend(summaries)
            self._prediction = tf.nn.softmax(self._model_output, axis=1)

        with tf.variable_scope('ModelLoss'):
            self._loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self._model_output, labels=labels)
            self._loss = tf.reduce_mean(self._loss)

            self._summaries.append(tf.summary.scalar('Loss', self._loss))
            self._valid_summaries.append(self._summaries[-1])

        with tf.name_scope('AccuracyStat'):
            is_correct = tf.equal(tf.argmax(self._prediction, 1),
                                  tf.argmax(labels, 1))
            self._accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            self._summaries.append(tf.summary.scalar('Accuracy', self._accuracy))
            self._valid_summaries.append(self._summaries[-1])

        self._summaries.append(tf.summary.scalar('LearningRate', learning_rate))

        # update batchnorm EMAs before updating weights (in one feed)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Model')):
            with tf.name_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate)

                # clip gradients to avoid nans
                gradients, variables = zip(*optimizer.compute_gradients(
                    self._loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model')))
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self._train_op = optimizer.apply_gradients(zip(gradients, variables))

        self._summaries = tf.summary.merge(self._summaries)
        self._valid_summaries = tf.summary.merge(self._valid_summaries)

        self._init = tf.global_variables_initializer()

        tf.get_default_graph().finalize()

    def _layer_wrapper(self, layer, activation):
        if self._config.enable_batchnorm:
            layer = tf.layers.batch_normalization(layer, training=self._is_training)
        return activation(layer)

    def _create_model(self, img_features):
        summary = []
        r = img_features
        for l in self._config.hidden_size:
            r = self._layer_wrapper(tf.layers.dense(r, l, use_bias=False),
                                    activation=self._config.activation_fun)
        r = tf.layers.dense(r, self._config.model_output_shape[0])
        return r, summary

    def train_op(self):
        return [self._summaries, self._train_op]

    def valid_op(self):
        return [self._valid_summaries]

    def eval_op(self):
        return [self._prediction]

    def accuracy_op(self):
        return [self._accuracy]

    def init_fun(self, sess):
        sess.run(self._init)

    @staticmethod
    def feed_lr():
        return True
