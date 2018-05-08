import argparse
import time
from pathlib import Path

import tensorflow as tf

import config
import data
import models
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multilayer perceptron training for birds recogntion.')
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=True)
    parser.add_argument('-t', '--model-type', help='Model type (perceptron, my_conv or vgg16_pretrained)',
                        dest='model_type', type=str, default='perceptron')
    parser.add_argument('-p', '--plot-name', help='Save model training and validation summaries plot as plot_name.png',
                        dest='plot_name', type=str, default=None)
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


def draw_plots(args, plot_values, plot_valid_values):
    for i, key in enumerate(set(plot_values.keys(), plot_valid_values.keys())):
        plt.subplot(311 + i)
        plt.title(k)
        plt.xlabel('Iteration')
        plt.ylabel(k)
        plt.grid(True)

        # TODO: add legend in case there are both train and valid sumaries for the same key
        for plt_values in (plot_values, plot_valid_values):
            if key in plt_values:
                k, values = plt_values[key]
                plt.plot(*zip(values))

    plt.tight_layout()
    plt.savefig(args.plot_name + '.png')
    plt.show()


def main(args):
    Model, conf = get_model_and_config(args)
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
        learning_rate = tf.placeholder(tf.float32, name='LearningRate')
        is_training = tf.placeholder(tf.bool, shape=(), name='IsTraining')

        model = Model(img_features, labels, learning_rate, is_training=is_training, config=conf)
        model.init_fun(sess)

        # setup saving summaries and checkpoints
        model_dir = str(conf.model_dir.joinpath(args.model_name))
        writer = tf.summary.FileWriter(model_dir)
        writer.add_graph(sess.graph)
        validation_dir = str(conf.model_dir.joinpath(args.model_name + '_validation'))
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

        plot_values = []
        plot_valid_values = []

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

                feed_dict = {
                    img_features: img,
                    labels: lbl,
                    is_training: True,
                }
                feed_dict[learning_rate] = lr
                if i == conf.finetune_begin:
                    print('starting finetuning')
                    lr = conf.initil_finetune_lr
                if i < conf.finetune_begin:
                    summaries = sess.run(model.train_op(), feed_dict)[0]
                else:
                    summaries = sess.run(model.full_train_op(), feed_dict)[0]

                if i % conf.save_train_summaries_ratio == 0:
                    writer.add_summary(summaries, i)
                    if args.plot_name:
                        # TODO: add plot_values here
                        pass
                if i % conf.save_weights_ratio == 0 and i > 0:
                    saver.save(sess, model_dir, global_step=i)

                if i % conf.save_validation_summaries_ratio == 0:
                    print('step', i)
                    img, lbl = validation_set
                    summaries = model.valid_fun(sess, img, lbl)
                    validation_writer.add_summary(summaries, i)
                    writer.flush()
                    validation_writer.flush()

                    if args.plot_name:
                        # TODO: get values from summary object - like following:
                        # summary_proto = tf.Summary().FromString(summaries)
                        # for entry in summary_proto.value:
                        #     # i teraz z entry pobież co trzeba i gdzieś zapisz

                        plot_valid_values['Accuracy'].append((i, model.accuracy_fun(sess, img, lbl)))
                        plot_valid_values['LearningRate'].append((i,lr))
                        # plot_values['Loss'].append((i,loss))


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
        if args.plot_name:
            draw_plots(args, plot_values, plot_valid_values)

if __name__ == '__main__':
    main(parse_arguments())
