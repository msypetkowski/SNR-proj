"""
TODO: more visualizations - hog images, training and test set,
precission-recall plot, etc.
"""
import argparse
from pathlib import Path

from data import *
from eval import predict_classes
from train import get_model_and_config
from utils import group
from visualization import merge_images_with_border, show_image, make_border


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Birds recognition system.')

    # data preview
    parser.add_argument('--raw', help='Veiw raw images',
                        dest='view_raw', action='store_true', default=False)
    parser.add_argument('--test', help='View whole testset',
                        dest='view_test', action='store_true', default=False)
    parser.add_argument('--train', help='View trainset',
                        dest='view_train', action='store_true', default=False)
    parser.add_argument('--aug', help='Show augmentation on an example',
                        dest='augmentation', action='store_true', default=False)

    parser.add_argument('--hog', help='View hog feature vectors (generic parameter)',
                        dest='view_hog', action='store_true', default=False)
    parser.add_argument('-c', '--view-count', help='How much images should be viewed (generic parameter)',
                        dest='view_count', type=int, default=36)

    # additional parameters for --test
    parser.add_argument('-e', '--eval', help='Evaluate with trained nn and show true/false answers.',
                        dest='eval', action='store_true', default=False)
    parser.add_argument('-m', '--model-dir', help='Model dir',
                        dest='model_dir', type=Path, required=False)
    parser.add_argument('-n', '--model-name', help='Model name',
                        dest='model_name', type=str, required=False)
    parser.add_argument('-t', '--model-type', help='Model type (currently only perceptron is supported)',
                        dest='model_type', type=str, default='perceptron')

    return parser.parse_args()


def view_raw(args):
    data = read_raw(args.view_count)
    for name, _, cls, bbox in data:
        print(name, cls, bbox)
    for d in group(data, 36):
        # display raw images
        _, images, _, _ = zip(*d)
        show_image(merge_images_with_border([cv2.resize(i, (150, 150)) for i in images]))

        # display prepared images
        # images, _ = zip(*starmap(lambda *args1: process_one_example(*args1, use_hog=args.view_hog), d))
        # show_image(merge_images_with_border([cv2.resize(i, (150, 150)) for i in images]))


def view_test(args):
    _, conf = get_model_and_config(args)
    _, raw_test_data = get_train_validation_raw(conf.validation_raw_examples_ratio)
    images, _ = get_unaugmented(raw_test_data, use_hog=args.view_hog)
    features, labels = get_unaugmented(raw_test_data, use_hog=True)
    if args.eval:
        if not args.model_dir or not args.model_name:
            raise ValueError("Reguired model-dir and model-name for testset evaluation.")
        prediction = predict_classes(args, features, labels)
        is_correct = [np.argmax(lbl) == np.argmax(pred) for lbl, pred in zip(labels, prediction)]
        assert len(is_correct) == len(images)
        print('good answers:', sum(is_correct))
        print('accuracy:', sum(is_correct) / len(is_correct))
        images = [make_border(img, color=(int(cor) * 255, 0, int(not cor) * 255))
                  for img, cor in zip(images, is_correct)]
        show_image(merge_images(images, scale=0.5))
    else:
        # show_image(merge_images_with_border([cv2.resize(i, (100, 100)) for i in images]))
        show_image(merge_images_with_border(images))


def view_train(args):
    _, conf = get_model_and_config(args)
    train_data, _ = get_train_validation_raw(conf.validation_raw_examples_ratio)
    train_data = batch_generator(train_data, conf.batch_size, use_hog=args.view_hog)
    viewed = 0
    for images, _ in train_data:
        show_image(merge_images_with_border(images))
        viewed += conf.batch_size
        if viewed >= args.view_count:
            break


def view_aug(args):
    data = [read_raw(2)[1]]
    images, _ = next(iter(batch_generator(data, args.view_count, use_hog=args.view_hog)))
    show_image(get_unaugmented(data, use_hog=args.view_hog)[0][0])
    show_image(merge_images_with_border(images))


def main(args):
    if args.view_test:
        view_test(args)

    if args.view_train:
        view_train(args)

    if args.view_raw:
        view_raw(args)

    if args.augmentation:
        view_aug(args)


if __name__ == '__main__':
    main(parse_arguments())
