"""
Data and model configuration.
"""

import glob
from pathlib import Path

import tensorflow as tf


class BaseConfig:
    data_root_path = Path.home().joinpath('snrdata')
    data_bboxes = data_root_path.joinpath('bounding_boxes.txt')
    data_classes = data_root_path.joinpath('classes.txt')
    data_labels = data_root_path.joinpath('image_class_labels.txt')
    images_paths = list(map(Path, glob.glob(
        str(data_root_path.joinpath('SET_C')) + '/**/*.jpg')))
    classes_count = 50
    random_seed = 123
    hog_feature_size = 3780
    raw_image_size = 196
    feed_ready_image_size = 128  # for convolution and visualization


class PerceptronConfig(BaseConfig):
    model_img_features_shape = (BaseConfig.hog_feature_size,)
    model_labels_shape = (BaseConfig.classes_count,)
    model_output_shape = model_labels_shape

    # training config
    batch_size = 128
    initial_lr = 0.01
    lr_decay = 0.5 ** (1 / 2000)  # decrease lr by 50% in 2000 iterations
    max_training_steps = 20000
    save_weights_ratio = 500
    save_train_summaries_ratio = 5
    save_validation_summaries_ratio = 20

    # validation config
    validation_raw_examples_ratio = 0.1

    # model config
    hidden_size = [256, 128, 64]
    weights_init_stddev = 0.02
    enable_batchnorm = True
    activation_fun = tf.nn.relu
    # activation_fun = tf.sigmoid


config = BaseConfig
