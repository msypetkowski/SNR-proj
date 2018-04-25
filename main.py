"""
TODO: model, training loop etc.
"""
import argparse
from itertools import starmap

import cv2
from data import *
from visualization import merge_images, show_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Birds recognition system.')

    # data preview
    parser.add_argument('-v', '--view-data', help='Display test set instead of training',
                        dest='view_data', action='store_true', default=False)
    parser.add_argument('--view-count', help='How much images should be viewed',
                        dest='view_count', type=int, default=16)

    return parser.parse_args()


def data_preview(args):
    data = read_raw(args.view_count)
    data = map_classes(data)

    for name, _, cls, bbox in data:
        print(name, cls, bbox)

    # display data
    _, images, _, _ = zip(*data)
    show_image(merge_images([cv2.resize(i, (200, 200)) for i in images]))

    # display prepared images (not hog features)
    images, _ = zip(*starmap(lambda *args: process_one_example(*args, use_hog=False), data))
    show_image(merge_images([cv2.resize(i, (200, 200)) for i in images]))

    # display hog
    show_image(merge_images([get_hog_features(i) for i in images]))


def main(args):
    if args.view_data:
        data_preview(args)


if __name__ == '__main__':
    main(parse_arguments())
