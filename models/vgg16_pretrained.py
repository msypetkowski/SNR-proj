""" Ripped from: https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
VGG16 model pretrained on ImageNet

wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz

"""
from pathlib import Path
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from .base_model import BaseModel


class VGG16PretrainedModel(BaseModel):
    def __init__(self, images, labels, is_training, config):
        self._images = images
        self._labels = labels
        self._is_training = is_training
        self._config = config
        self._summaries = []
        # self._valid_summaries = []

        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=config.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=config.classes_count, is_training=is_training,
                                   dropout_keep_prob=config.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = config.model_path
        assert(model_path.is_file())

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
        self._init_fn = tf.contrib.framework.assign_from_checkpoint_fn(str(model_path), variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        self._fc8_init = tf.variables_initializer(fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        # tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        fc8_optimizer = tf.train.GradientDescentOptimizer(config.learning_rate1)
        fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(config.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        lbl = tf.to_int32(tf.argmax(labels, 1))
        # print(prediction.shape)
        # print(labels.shape)
        correct_prediction = tf.equal(prediction, lbl)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._train_op = fc8_train_op
        self._full_train_op = full_train_op
        self._prediction = tf.nn.softmax(logits)
        self._accuracy=accuracy

        self._summaries.append(tf.summary.scalar('Accuracy', accuracy))
        # self._valid_summaries.append(self._summaries[-1])
        self._summaries.append(tf.summary.scalar('Loss', loss))
        # self._valid_summaries.append(self._summaries[-1])
        self._summaries = tf.summary.merge(self._summaries)
        # self._valid_summaries = tf.summary.merge(self._valid_summaries)

        # tf.get_default_graph().finalize()

    def train_op(self):
        return [self._summaries, self._train_op]

    # for finetuning
    def full_train_op(self):
        return [self._summaries, self._full_train_op]

    def valid_fun(self, sess, images, labels):
        accuracy = self.accuracy_fun(sess, images, labels)
        summary = tf.Summary()
        summary.value.add(tag="Accuracy", simple_value=accuracy)
        return summary

    def init_fun(self, sess):
        self._init_fn(sess)
        sess.run(self._fc8_init)

    @staticmethod
    def feed_lr():
        return False
