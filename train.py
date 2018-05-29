#!/bin/env python3
import argparse
import json
import time
from collections import defaultdict

import tensorflow as tf

import data
import eval
from models import get_model_and_config
from visualization import draw_plots


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multilayer perceptron training for birds recogntion.')
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=True)
    parser.add_argument('-t', '--model-type', help='Model type (Perceptron, MyConv or VGG16Pretrained)',
                        dest='model_type', type=str, default='MyConv')
    parser.add_argument('-p', '--draw-plot', help='Save model training and validation summaries plot'
                                                  'to files model_namel.{png, eps, json}',
                        dest='draw_plot', action='store_true', default=False)
    parser.add_argument('-c', '--continue', help='Step to continue from.',
                        dest='restore', type=int, default=None)
    return parser.parse_args()


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


def append_values_from_summaries(target, summaries, iteration):
    summary_proto = tf.Summary().FromString(summaries)
    val = summary_proto.value
    while val:
        v = val.pop()
        target[v.tag].append((iteration, v.simple_value))


def main(args):
    model_cls, conf = get_model_and_config(args)
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
        learning_rate = tf.placeholder(tf.float32, name='Lr')
        is_training = tf.placeholder(tf.bool, shape=(), name='IsTraining')

        model = model_cls(img_features, labels, learning_rate, is_training=is_training, config=conf)
        saver = tf.train.Saver()
        if args.restore is None:
            model.init_fun(sess)
        else:
            checkpoints = eval.list_model_checkpoints(args.model_name, conf.model_dir)
            chkp = checkpoints[-1]  # use latest checkpoint
            print('using checkpoint:', chkp)
            saver.restore(sess, chkp)

        # setup saving summaries and checkpoints
        model_dir = str(conf.model_dir.joinpath(args.model_name))
        writer = tf.summary.FileWriter(model_dir)
        writer.add_graph(sess.graph)
        validation_dir = str(conf.model_dir.joinpath(args.model_name + '_validation'))
        validation_writer = tf.summary.FileWriter(validation_dir)

        print_parameters_stat()

        # setup input data
        train_data, test_data = data.get_train_validation_raw(conf.validation_raw_examples_ratio)
        batch_generator = iter(data.batch_generator(train_data, conf.batch_size, use_hog=conf.use_hog))
        validation_set = data.get_unaugmented(test_data, use_hog=conf.use_hog)

        # measure time
        start_time = time.time()
        data_generation_time_sum = 0

        plot_values = defaultdict(list)
        plot_valid_values = defaultdict(list)

        # training loop
        lr = conf.initial_lr
        if args.restore is None:
            i = 0
        else:
            i = args.restore
        try:
            for i in range(i, conf.max_training_steps + i):
                lr *= conf.lr_decay

                # generate next batch
                start_generation_time = time.time()
                img, lbl = next(batch_generator)
                data_generation_time_sum += time.time() - start_generation_time

                feed_dict = {img_features: img, labels: lbl, is_training: True, learning_rate: lr}
                if i == conf.finetune_begin:
                    print('starting finetuning')
                    lr = conf.initil_finetune_lr
                if i < conf.finetune_begin:
                    summaries = sess.run(model.train_op(), feed_dict)[0]
                else:
                    summaries = sess.run(model.full_train_op(), feed_dict)[0]

                if i % conf.save_train_summaries_ratio == 0:
                    writer.add_summary(summaries, i)
                    if args.draw_plot:
                        append_values_from_summaries(plot_values, summaries, i)
                if i % conf.save_weights_ratio == 0 and i > 0:
                    saver.save(sess, model_dir, global_step=i)

                if i % conf.save_validation_summaries_ratio == 0:
                    print('step', i)
                    img, lbl = validation_set
                    summaries = model.valid_fun(sess, img, lbl)
                    validation_writer.add_summary(summaries, i)
                    writer.flush()
                    validation_writer.flush()

                    if args.draw_plot:
                        append_values_from_summaries(plot_valid_values, summaries, i)


        except KeyboardInterrupt:
            pass

        saver.save(sess, model_dir, global_step=i)

        # last validation
        img, lbl = validation_set
        summaries = model.valid_fun(sess, img, lbl)
        validation_writer.add_summary(summaries, i)
        writer.flush()
        validation_writer.flush()

        print('final accuracy:', model.accuracy_fun(sess, img, lbl))

        t = time.time() - start_time
        print('global training time:', str(t) + "s")
        print('waited for batch generation:', str(data_generation_time_sum) + "s")
        print('ratio:', data_generation_time_sum / t)

        # save plots (accuracy, learning rate and model loss)
        if args.draw_plot:
            plot_name = args.model_name
            draw_plots(plot_name, plot_values, plot_valid_values, args.model_name)
            with open(plot_name + '.json', 'w') as f:
                f.write(json.dumps((plot_values, plot_valid_values)))


if __name__ == '__main__':
    main(parse_arguments())
