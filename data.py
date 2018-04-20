"""
Utilities for reading and preparing data.
"""
from itertools import islice
import random

import cv2
import numpy as np

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
    """ Returns list of tuples (image_filename, tensor, class, bounding_box)
    """
    if count == -1:
        count = len(config.images_paths)
    bboxes = read_bboxes()
    classes = read_classes()
    return [(p.name, cv2.imread(str(p)), classes[p.name], bboxes[p.name])
            for p in islice(config.images_paths, count)]


def get_class_mapping(raw_data):
    """ Returns dict int -> int.
    Classes in raw_data may have values from non-continous range.
    """
    _, _, classes, _ = zip(*raw_data)
    return {cls:i for i,cls in enumerate(np.unique(classes))}


def map_classes(raw_data):
    """ Changes classes, so that they are subsequent numbers
    from 0 to config.classes_count
    """
    mapping = get_class_mapping(raw_data)
    return [(fn, img, mapping[cls], bbox) for fn, img, cls, bbox in raw_data]


def process_one_example(name, image, cls, bbox, rand=random.Random()):
    x, y, w, h = bbox
    img = image[y:y+w, x:x+w]
    label = np.zeros(config.classes_count)
    assert 0 <= cls < config.classes_count
    label[cls] = 1.0
    return img, cls


def example_generator(raw_data, random_seed=123):
    """ Yields examples - pairs (prepared_image, cls_vec)
    Does data augmentation.
    """
    rand = random.Random(123)
    rand.shuffle(raw_data)
    for ex in raw_data:
        yield process_one_example(*ex)


def get_hog_features(img):
    # TODO: check and adjust hog parameters, input image size itp.
    img = cv2.resize(img, (128, 128))
    hog = cv2.HOGDescriptor()
    return hog.compute(img)
