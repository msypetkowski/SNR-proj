import argparse
import time
from pathlib import Path

import tensorflow as tf

import config
import data
import perceptron


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multilayer perceptron training for birds recogntion.')
    parser.add_argument('-m', '--model-dir', help='Model dir',
                        dest='model_dir', type=Path, required=True)
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=True)
    parser.add_argument('-t', '--model-type', help='Model type (currently only perceptron is supported)',
                        dest='model_type', type=str, default='perceptron')
    return parser.parse_args()


def get_model_and_config(args):
    if args.model_type == 'perceptron':
        conf = config.PerceptronConfig
        Model = perceptron.PerceptronModel
    else:
        raise ValueError("Model type not supported")
    return Model, conf


def main(args):
    Model, conf = get_model_and_config(args)
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
        learning_rate = tf.placeholder(tf.float32, name='LearningRate')
        is_training = tf.placeholder(tf.bool, shape=(), name='IsTraining')

        model = Model(img_features, labels, learning_rate, is_training=is_training, config=conf)
        sess.run(tf.global_variables_initializer())

        # setup saving summaries and checkpoints
        model_dir = str(args.model_dir.joinpath(args.model_name))
        writer = tf.summary.FileWriter(model_dir)
        writer.add_graph(sess.graph)
        validation_dir = str(args.model_dir.joinpath(args.model_name + '_validation'))
        validation_writer = tf.summary.FileWriter(validation_dir)
        saver = tf.train.Saver()

        # setup input data
        train_data, test_data = data.get_train_validation_raw(conf.validation_raw_examples_ratio)
        batch_generator = iter(data.batch_generator(train_data, conf.batch_size))
        validation_set = data.get_unaugmented(test_data)

        # measure time
        start_time = time.time()
        data_generation_time_sum = 0

        # training loop
        lr = conf.initial_lr
        i = 0
        try:
            for i in range(conf.max_training_steps):
                lr *= conf.lr_decay

                # generate next batch
                start_generation_time = time.time()
                img, lbl = next(batch_generator)
                data_generation_time_sum += time.time() - start_generation_time

                summaries = sess.run(model.train_op(), {
                    img_features: img,
                    labels: lbl,
                    learning_rate: lr,
                    is_training: True,
                })[0]

                if i % conf.save_train_summaries_ratio == 0:
                    writer.add_summary(summaries, i)
                if i % conf.save_weights_ratio == 0:
                    saver.save(sess, model_dir, global_step=i)

                if i % conf.save_validation_summaries_ratio == 0:
                    print('step', i)
                    img, lbl = validation_set
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
