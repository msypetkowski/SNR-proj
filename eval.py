import argparse
import glob
from pathlib import Path

import tensorflow as tf

import data
from train import Model, config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Multilayer perceptron evaluation on given images.')
    parser.add_argument('-m', '--model-dir', help='Model dir',
                        dest='model_dir', type=Path, required=True)
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=True)
    # TODO: images parameter
    return parser.parse_args()


def list_model_checkpoints(model_name, model_dir):
    checkpoints = glob.glob(str(model_dir) + f'/{model_name}-*.meta', recursive=True)
    checkpoints = [c[:-5] for c in checkpoints]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    print('Model checkpoints:', end='\n\t')
    print('\t\n'.join(checkpoints))
    return checkpoints


def predict_classes(args, images_features):
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + config.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + config.model_labels_shape, name='ImgLabels')
        model = Model(img_features, labels, 0, is_training=False)

        saver = tf.train.Saver()
        checkpoints = list_model_checkpoints(args.model_name, args.model_dir)
        chkp = checkpoints[-1]  # use latest checkpoint
        print('using checkpoint:', chkp)
        saver.restore(sess, chkp)

        return sess.run(model.eval_op(), {
            img_features: images_features,
            # labels: lbl,
            # learning_rate: lr,
        })[0]


def main(args):
    raise NotImplementedError


if __name__ == '__main__':
    main(parse_arguments())
