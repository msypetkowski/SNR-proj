import argparse
from itertools import islice
from pathlib import Path

import tensorflow as tf

import data
from config import BaseConfig


class Config(BaseConfig):
    model_img_features_shape = (BaseConfig.hog_feature_size,)
    model_labels_shape = (BaseConfig.classes_count,)
    model_output_shape = model_labels_shape
    batch_size = 32
    validation_size = 1000


config = Config


def parse_arguments():
    parser = argparse.ArgumentParser(description='GAN test')
    parser.add_argument('-m', '--model-dir', help='Model dir',
                        dest='modelDir', type=Path, required=True)
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='modelName', type=str, required=True)
    return parser.parse_args()


class Model:

    def __init__(self, img_features, labels, learning_rate, is_training):
        self._is_training = is_training

        self._summaries = []
        self._valid_summaries = []

        initializer = tf.truncated_normal_initializer(0, 0.02)

        with tf.variable_scope('Model', initializer=initializer):
            self._model_output, summaries = self._create_model(img_features)
            self._summaries.extend(summaries)

        with tf.variable_scope('ModelLoss'):
            self._loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self._model_output, labels=labels, name="XEntWithLogits")
            self._loss = tf.reduce_mean(self._loss)
            self._summaries.append(tf.summary.scalar('Loss', self._loss))
            self._valid_summaries.append(self._summaries[-1])

        self._summaries.append(tf.summary.scalar('LearningRate', learning_rate))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Model')):
            with tf.name_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(
                    self._loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model')))
                gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self._optimizer = optimizer.apply_gradients(zip(gradients, variables))

        self._summaries = tf.summary.merge(self._summaries)
        self._valid_summaries = tf.summary.merge(self._valid_summaries)

    def _batchnorm_wrapper(self, layer, activation=tf.nn.relu):
        return activation(tf.layers.batch_normalization(layer, training=self._is_training))

    def _create_model(self, img_features):
        summary = []
        # TODO: experiment
        layers = [512, 256, 128, 64]
        r = img_features
        for l in layers:
            r = self._batchnorm_wrapper(tf.layers.dense(r, l, use_bias=False))
        r = tf.layers.dense(r, config.model_output_shape[0])
        return r, summary

    def train_op(self):
        return [self._summaries, self._optimizer]

    def valid_op(self):
        return [self._valid_summaries]

    def eval_op(self):
        return [self._model_output]


def main(args):
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + config.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + config.model_labels_shape, name='ImgLabels')
        learning_rate = tf.placeholder(tf.float32, name='LearningRate')

        model = Model(img_features, labels, learning_rate, is_training=True)
        sess.run(tf.global_variables_initializer())

        # setup saving summaries and checkpoints
        model_dir = str(args.modelDir.joinpath(args.modelName))
        writer = tf.summary.FileWriter(model_dir)
        writer.add_graph(sess.graph)
        writer_save_ratio = 5

        validation_dir = str(args.modelDir.joinpath(args.modelName + '_validation'))
        validation_writer = tf.summary.FileWriter(validation_dir)
        validation_writer.add_graph(sess.graph)

        saver = tf.train.Saver()
        saver_save_ratio = 100
        validation_ratio = 10

        # setup input data
        train_data, test_data = data.get_train_validation_raw(0.1)
        batch_generator = iter(data.batch_generator(train_data, config.batch_size))
        validation_set = list(islice(data.batch_generator(
                         test_data, batch_size=config.validation_size), 1))[0]

        # training loop
        lr = 0.1
        i = 0
        try:
            for i in range(100000):
                print(i)
                lr *= 0.5 ** (1 / 200)
                img, lbl = next(batch_generator)
                summaries = sess.run(model.train_op(), {
                    img_features: img,
                    labels: lbl,
                    learning_rate: lr,
                })[0]

                if i % validation_ratio == 0:
                    img, lbl = validation_set
                    summaries = sess.run(model.valid_op(), {
                        img_features: img,
                        labels: lbl,
                        learning_rate: lr,
                    })[0]
                    validation_writer.add_summary(summaries, i)
                    writer.flush()
                    validation_writer.flush()

                if i % writer_save_ratio == 0:
                    writer.add_summary(summaries, i)
                if i % saver_save_ratio == 0:
                    saver.save(sess, model_dir, global_step=i)

        except KeyboardInterrupt:
            saver.save(sess, model_dir, global_step=i)


if __name__ == '__main__':
    main(parse_arguments())
