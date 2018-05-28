import glob

import tensorflow as tf

from train import parse_arguments
from models import get_model_and_config


def list_model_checkpoints(model_name, model_dir):
    checkpoints = glob.glob(str(model_dir) + f'/{model_name}-*.meta', recursive=True)
    checkpoints = [c[:-5] for c in checkpoints]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    print('Model checkpoints:', end='\n\t')
    print('\t\n'.join(checkpoints))
    return checkpoints


def predict_classes(args, images_features, lbl=None):
    model_cls, conf = get_model_and_config(args)
    with tf.Session() as sess:
        img_features = tf.placeholder(tf.float32, (None,) + conf.model_img_features_shape, name='ImgFeatures')
        labels = tf.placeholder(tf.float32, (None,) + conf.model_labels_shape, name='ImgLabels')
        model = model_cls(img_features, labels, 0, is_training=False, config=conf)

        saver = tf.train.Saver()
        checkpoints = list_model_checkpoints(args.model_name, conf.model_dir)
        chkp = checkpoints[-1]  # use latest checkpoint
        print('using checkpoint:', chkp)
        saver.restore(sess, chkp)

        if lbl is not None:
            print('accuracy:', model.accuracy_fun(sess, images_features, lbl))
        return model.eval_fun(sess, images_features)


def main(_):
    raise NotImplementedError


if __name__ == '__main__':
    main(parse_arguments())
