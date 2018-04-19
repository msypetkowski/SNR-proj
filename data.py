"""
Utilities for reading and preparing data.
"""
from itertools import islice

import cv2

from config import config


def to_filename(name):
    return ''.join(name.split('-')) + '.jpg'


def read_table(filename, attr_count):
    with open(filename, 'r') as f:
        lines = [l.split(' ') for l in f.read().split('\n')]
        # ignore lines with different columns count
        return [l for l in lines if len(l) == attr_count]


def read_bboxes():
    """ Returns dict str : bbox
    """
    return {to_filename(name): list(map(int, rect))
            for name, *rect in read_table(config.data_bboxes, 5)}


def read_classes():
    """ Returns dict str : class
    """
    return {to_filename(name): int(cls)
            for name, cls in read_table(config.data_labels, 2)}


def read_raw(count=-1):
    """ Returns list of tuples (hash, tensor, class, bounding_box)
    """
    if count == -1:
        count = len(config.images_paths)
    bboxes = read_bboxes()
    classes = read_classes()
    return [(p.name, cv2.imread(str(p)), bboxes[p.name], classes[p.name])
            for p in islice(config.images_paths, count)]


def get_hog_features(img):
    # TODO: check and adjust hog parameters, input image size itp.
    img = cv2.resize(img, (128, 128))
    hog = cv2.HOGDescriptor()
    return hog.compute(img)
