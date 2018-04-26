"""
TODO: more visualizations - hog images, training and test set,
precission-recall plot, etc.
"""
import argparse
from itertools import starmap

import cv2
from data import *
from visualization import merge_images_with_border, show_image
from utils import group
from main2 import config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Birds recognition system.')

    # data preview
    parser.add_argument('-r', '--raw', help='Veiw raw images',
                        dest='view_raw', action='store_true', default=False)
    parser.add_argument('-t', '--test', help='View whole testset',
                        dest='view_test', action='store_true', default=False)
    parser.add_argument('-n', '--train', help='View trainset',
                        dest='view_train', action='store_true', default=False)
    parser.add_argument('-a', '--augmentation', help='Show augmentation on an example',
                        dest='augmentation', action='store_true', default=False)

    parser.add_argument('--hog', help='View hog feature vectors (generic parameter)',
                        dest='view_hog', action='store_true', default=False)
    parser.add_argument('-c', '--view-count', help='How much images should be viewed (generic parameter)',
                        dest='view_count', type=int, default=36)

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
    _, test_data = get_train_validation_raw(config.validation_raw_examples_ratio)
    test_data = get_unaugmented(test_data, use_hog=args.view_hog)
    images, _ = test_data
    show_image(merge_images_with_border([cv2.resize(i, (100, 100)) for i in images]))
    # show_image(merge_images_with_border(images))


def view_train(args):
    train_data, _ = get_train_validation_raw(config.validation_raw_examples_ratio)
    train_data = batch_generator(train_data, config.batch_size, use_hog=args.view_hog)
    viewed = 0
    for images, _ in train_data:
        show_image(merge_images_with_border(images))
        viewed += config.batch_size
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
