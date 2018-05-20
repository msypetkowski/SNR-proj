"""
Data and models configuration.
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
    # raw_image_size = 196
    raw_image_size = 256

    # feed_ready_image_size = 128  # for convolution and visualization
    feed_ready_image_size = 224

    model_dir = Path.home().joinpath('snrmodels')

    finetune_begin = float('inf')  # no finetuning by default


class BaseTrainingLoopConfig(BaseConfig):
    save_train_summaries_ratio = 5
    save_validation_summaries_ratio = 20
    validation_raw_examples_ratio = 0.1
    max_training_steps = 20000
    save_weights_ratio = 1000


class PerceptronConfig(BaseTrainingLoopConfig):
    # training config
    batch_size = 128
    initial_lr = 0.01
    lr_decay = 0.5 ** (1 / 2000)  # decrease lr by 50% in 2000 iterations

    # model config
    model_img_features_shape = (BaseConfig.hog_feature_size,)
    model_labels_shape = (BaseConfig.classes_count,)
    model_output_shape = model_labels_shape
    use_hog = True
    hidden_size = [16, 128, 64]
    weights_init_stddev = 0.02
    enable_batchnorm = True
    activation_fun = tf.nn.relu
    # activation_fun = tf.sigmoid


# 1695858 parameters
class MyConvConfigMedium(PerceptronConfig):
    # training config
    batch_size = 128
    max_training_steps = 5000
    save_weights_ratio = 1000
    initial_lr = 0.03
    lr_decay = 0.5 ** (1 / 1000)  # decrease lr by 50% in 1000 iterations

    # model config
    model_img_features_shape = (*[BaseConfig.feed_ready_image_size] * 2, 3)
    use_hog = False
    use_max_pool = True
    # when use_max_pool enabled, given stride is used by max polling layers
    hidden_size = [
        # filter size   strides     output depth
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((3, 3),        (2, 2),     128),
        ((2, 2),        (1, 1),     128),
        ((2, 2),        (1, 1),     128),  # 4x4
    ]
    dense_hidden_size = [512]


# 1527538 parameters
class MyConvConfigLessLayers(MyConvConfigMedium):
    hidden_size = [
        # filter size   strides     output depth
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (4, 4),     64), #  larger strides
        # ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((3, 3),        (2, 2),     128),
        # ((2, 2),        (1, 1),     128),
        ((2, 2),        (1, 1),     128),  # 4x4
    ]


# 1966706 parameters
class MyConvConfigMoreLayers(MyConvConfigMedium):
    hidden_size = [
        # filter size   strides     output depth
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (1, 1),     64),  # additional
        ((5, 5),        (2, 2),     64),
        ((5, 5),        (1, 1),     64),  # additional
        ((3, 3),        (2, 2),     128),
        ((2, 2),        (1, 1),     128),
        ((2, 2),        (1, 1),     128),  # additional
        ((2, 2),        (1, 1),     128),  # 4x4
    ]


MyConvConfig = MyConvConfigMedium


class VGG16PretrainedConfig(BaseTrainingLoopConfig):
    model_img_features_shape = (*[224] * 2, 3)
    model_labels_shape = (BaseConfig.classes_count,)

    use_hog = False
    batch_size = 32
    dropout_keep_prob = 0.5
    weight_decay = 5e-4
    model_path = Path('vgg_16.ckpt')

    initial_lr = 5e-3
    initil_finetune_lr = 5e-5
    lr_decay = 0.5 ** (1 / 500)

    max_training_steps = 2500
    finetune_begin = 1500
