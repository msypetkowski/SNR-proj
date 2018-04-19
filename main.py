"""
TODO: model, training loop etc.
"""
import argparse

import cv2
from data import read_raw
from visualization import merge_images, show_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Birds recognition system.')
    parser.add_argument('-v', '--view-data', help='Display test set instead of training',
                        dest='view_data', action='store_true', default=False)
    parser.add_argument('--view-count', help='How much images should be viewed',
                        dest='view_count', type=int, default=16)
    return parser.parse_args()


def main(args):
    if args.view_data:
        raw = read_raw(args.view_count)
        _, images, _, _ = zip(*raw)
        show_image(merge_images([cv2.resize(i, (100, 100)) for i in images]))
        for name, _, cls, bbox in raw:
            print(name, cls, bbox)


if __name__ == '__main__':
    main(parse_arguments())
