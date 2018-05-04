import argparse
import time
from pathlib import Path

import tensorflow as tf

import config
import data
import models


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multilayer perceptron training for birds recogntion.')
    parser.add_argument('-m', '--model-dir', help='Model dir',
                        dest='model_dir', type=Path, required=True)
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=True)
    parser.add_argument('-t', '--model-type', help='Model type (perceptron, my_conv or vgg16_pretrained)',
                        dest='model_type', type=str, default='perceptron')
    return parser.parse_args()


def get_model_and_config(args):
    if args.model_type == 'perceptron':
        conf = config.PerceptronConfig
        Model = models.perceptron.PerceptronModel
    elif args.model_type == 'my_conv':
        conf = config.MyConvConfig
        Model = models.my_conv.MyConvModel
    elif args.model_type == 'vgg16_pretrained':
        conf = config.VGG16PretrainedConfig
        Model = models.vgg16_pretrained.VGG16PretrainedModel
    else:
        raise ValueError("Model type {} not supported".format(args.model_type))
    return Model, conf


def print_parameters_stat():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total trainable parameters: ', total_parameters)


def main(args):
    Model, conf = get_model_and_config(args)
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
        if Model.feed_lr():
            learning_rate = tf.placeholder(tf.float32, name='LearningRate')
        is_training = tf.placeholder(tf.bool, shape=(), name='IsTraining')

        if Model.feed_lr():
            model = Model(img_features, labels, learning_rate, is_training=is_training, config=conf)
        else:
            model = Model(img_features, labels, is_training=is_training, config=conf)
        model.init_fun(sess)

        # setup saving summaries and checkpoints
        model_dir = str(args.model_dir.joinpath(args.model_name))
        writer = tf.summary.FileWriter(model_dir)
        writer.add_graph(sess.graph)
        validation_dir = str(args.model_dir.joinpath(args.model_name + '_validation'))
        validation_writer = tf.summary.FileWriter(validation_dir)
        saver = tf.train.Saver()

        print_parameters_stat()

        # setup input data
        train_data, test_data = data.get_train_validation_raw(conf.validation_raw_examples_ratio)
        batch_generator = iter(data.batch_generator(train_data, conf.batch_size, use_hog=conf.use_hog))
        validation_set = data.get_unaugmented(test_data, use_hog=conf.use_hog)

        # measure time
        start_time = time.time()
        data_generation_time_sum = 0

        # training loop
        if model.feed_lr():
            lr = conf.initial_lr
        i = 0
        try:
            for i in range(conf.max_training_steps):
                if model.feed_lr():
                    lr *= conf.lr_decay

                # generate next batch
                start_generation_time = time.time()
                img, lbl = next(batch_generator)
                data_generation_time_sum += time.time() - start_generation_time

                feed_dict = {
                    img_features: img,
                    labels: lbl,
                    is_training: True,
                }
                if model.feed_lr():
                    feed_dict[learning_rate] = lr
                summaries = sess.run(model.train_op(), feed_dict)[0]

                if i % conf.save_train_summaries_ratio == 0:
                    writer.add_summary(summaries, i)
                if i % conf.save_weights_ratio == 0:
                    saver.save(sess, model_dir, global_step=i)

                if i % conf.save_validation_summaries_ratio == 0:
                    print('step', i)
                    img, lbl = validation_set
                    img = img[:64] # TODO: remove
                    lbl = lbl[:64] # TODO: remove
                    summaries = sess.run(model.valid_op(), {
                        img_features: img,
                        labels: lbl,
                        is_training: False,
                    })[0]
                    validation_writer.add_summary(summaries, i)
                    writer.flush()
                    validation_writer.flush()

        except KeyboardInterrupt:
            pass

        saver.save(sess, model_dir, global_step=i)

        # last validation
        img, lbl = validation_set
        summaries = sess.run(model.valid_op(), {
            img_features: img,
            labels: lbl,
            is_training: False,
        })[0]
        validation_writer.add_summary(summaries, i)
        writer.flush()
        validation_writer.flush()

        print('final accuracy:', sess.run(model.accuracy_op(), {
            img_features: img,
            labels: lbl,
            is_training: False,
        })[0])

        t = time.time() - start_time
        print('global training time:', str(t) + "s")
        print('waited for batch generation:', str(data_generation_time_sum) + "s")
        print('ratio:', data_generation_time_sum / t)


if __name__ == '__main__':
    main(parse_arguments())
