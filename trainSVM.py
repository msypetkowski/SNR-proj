#!/bin/env python3
import argparse

import tensorflow as tf
import numpy as np
from sklearn import svm
import data
from eval import list_model_checkpoints
from models import get_model_and_config
from data_preview import decorate_images
from visualization import show_image, merge_images


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train SVM after given layer (by tensor name).')
    parser.add_argument('-n', '--model-name', help='Base model name (of model to restore)',
                        dest='model_name', type=str, required=True)
    parser.add_argument('-t', '--model-type', help='Base model type (Perceptron, MyConv or VGG16Pretrained)',
                        dest='model_type', type=str, default='MyConv')
    parser.add_argument('-l', '--tensor-name', help='Tensorflow tensor name after which SVM should be placed',
                        dest='tensor_name', type=str, required=True)
    parser.add_argument('-k', '--kernel-type', help="SVM kernel type - one of:"
                        "‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed",
                        dest='kernel_type', type=str, default='rbf')
    parser.add_argument('-c', '--samples-count', help="Training samples count",
                        dest='training_samples_count', type=int, default=5000)
    return parser.parse_args()


def get_model(model_cls, conf):
    img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
    labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
    return model_cls(img_features, labels, 0, is_training=False, config=conf)


def main(args):
    model_cls, conf = get_model_and_config(args)
    with tf.Session() as sess:
        model = get_model(model_cls, conf)

        # restore latest checkpoint
        saver = tf.train.Saver()
        checkpoints = list_model_checkpoints(args.model_name, conf.model_dir)
        chkp = checkpoints[-1]  # use latest checkpoint
        print('using checkpoint:', chkp)
        saver.restore(sess, chkp)

        # find node that will be used as last layer before SVM
        graph = tf.get_default_graph()
        nn_output = graph.get_tensor_by_name(args.tensor_name)

        # prepare data
        train_data, test_data = data.get_train_validation_raw(conf.validation_raw_examples_ratio)
        batch_generator = iter(data.batch_generator(train_data, args.training_samples_count, use_hog=conf.use_hog))
        train_images, train_labels = next(batch_generator)
        test_images, test_labels = data.get_unaugmented(test_data, use_hog=conf.use_hog)

        # get train and test features
        train_features = model.eval_fun(sess, train_images, tensor=nn_output)
        test_features = model.eval_fun(sess, test_images, tensor=nn_output)
        assert(len(train_features) == args.training_samples_count)
        # print(len(train_features[0]))

        # train SVM with the features
        classifier = svm.SVC(kernel=args.kernel_type, decision_function_shape='ovo')
        classifier.fit(train_features, np.argmax(train_labels, 1))

        # measure and show the results on testset
        prediction = np.array(classifier.predict(test_features))
        is_correct = [np.argmax(lbl) == pred for lbl, pred in zip(test_labels, prediction)]
        images = decorate_images(test_images, is_correct)
        print('accuracy:', sum(is_correct) / len(is_correct))
        show_image(merge_images(images, scale=0.2))


if __name__ == '__main__':
    main(parse_arguments())
