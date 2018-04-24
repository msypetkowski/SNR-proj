"""
Utilities for reading and preparing data.
"""
import random
from itertools import islice, groupby, chain, repeat, cycle, starmap

import cv2
import numpy as np
from skimage.feature import hog

from config import config
from utils import transposed_group


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
    # TODO: downsample the image initially, because subsampling later takes too long
    # remember also about bboxes
    return [(p.name, cv2.imread(str(p)), classes[p.name], bboxes[p.name])
            for p in islice(config.images_paths, count)]


def get_class_mapping(raw_data):
    """ Returns dict int -> int.
    Classes in raw_data may have values from non-continuous range.
    """
    _, _, classes, _ = zip(*raw_data)
    return {cls: i for i, cls in enumerate(np.unique(classes))}


def map_classes(raw_data):
    """ Changes classes, so that they are subsequent numbers
    from 0 to config.classes_count
    """
    mapping = get_class_mapping(raw_data)
    return [(fn, img, mapping[cls], bbox) for fn, img, cls, bbox in raw_data]


def get_hog_features(img, feat_size=config.hog_feature_size):
    """ Returns features vector of size feat_size.
    """
    # TODO: check and adjust hog parameters, input image size itp.

    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(8, 8), visualise=True)

    # img = cv2.resize(img, (128, 128))
    # hog = cv2.HOGDescriptor()
    # return hog.compute(img)

    # TODO: find a good way for adjusting feature size
    # (but this doesn't seem to be that bad)
    rand = np.random.RandomState(config.random_seed)
    return rand.choice(hog_image.flatten(), feat_size)


def process_one_example(name, image, cls, bbox, rand=random.Random()):
    x, y, w, h = bbox
    img = image[y:y + w, x:x + w]
    # TODO: data augmentation
    label = np.zeros(config.classes_count)
    assert 0 <= cls < config.classes_count
    label[cls] = 1.0
    assert label.shape == (config.classes_count,)
    return get_hog_features(img), label


def example_generator(raw_data, random_seed=123):
    """ Yields examples infinitely - pairs (img_feature_vec, cls_vec)
    Does data augmentation.
    """
    rand = random.Random(random_seed)
    rand.shuffle(raw_data)
    for ex in cycle(raw_data):
        yield process_one_example(*ex)


def get_train_validation_raw(validation_ratio):
    """ Returns pair (train_raw_data, validation_raw_data)
    """
    raw_data = read_raw()
    raw_data = map_classes(raw_data)
    rand = random.Random(123)
    rand.shuffle(raw_data)
    raw_data.sort(key=lambda x: x[2])
    by_class = [list(g) for _, g in groupby(raw_data, lambda x: x[2])]
    assert len(by_class) == config.classes_count
    bucket_size = len(raw_data) / config.classes_count
    assert all(len(c) == bucket_size for c in by_class)

    # split maintaining same examples count for each class
    split_index = round(len(by_class[0]) * validation_ratio)
    return (sum([c[split_index:] for c in by_class], []),
            sum([c[:split_index] for c in by_class], []))


def batch_generator(raw_data, batch_size):
    """ Yields pairs of lists (images_features, labels)
    """
    return starmap(lambda x, y: (np.array(x), np.array(y)), transposed_group(
        example_generator(raw_data),
        batch_size))
