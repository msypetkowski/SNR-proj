"""
Utilities for reading and preparing data.
"""
import random
from itertools import islice, groupby, starmap

from config import BaseConfig as config
from img_process import transform_img
from utils import transposed_group
# from skimage.feature import hog
from visualization import *


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


def crop_image(image, crop_bbox, orig_bbox):
    """ Returns pair: new_image, new_bbox
    """
    x, y, w, h = crop_bbox
    y1 = max(y, 0)
    x1 = max(x, 0)
    img = image[y1:y + h, x1:x + w]

    ox, oy, ow, oh = orig_bbox
    new_bbox = (ox - x1, oy - y1, ow, oh)
    return img, new_bbox


def scale_image(image, target_size, orig_bbox):
    """ Returns pair: new_image, new_bbox
    """
    min_dim = min(image.shape[:2])
    ratio = target_size / min_dim
    if ratio >= 1:
        return image, orig_bbox
    img = cv2.resize(image, tuple([round(i * ratio) for i in image.shape[1::-1]]))
    new_bbox = tuple(map(lambda x: round(float(x)), (np.array(orig_bbox) * ratio)))
    return img, new_bbox


def preprocess_raw_image(img, bbox, extend_ratio=0.6):
    """ Crop with extended bbox and downscale too large images.
    Returns pair: new_image, new_bbox
    """
    x, y, w, h = bbox
    dx, dy = [round(i * extend_ratio) for i in (w, h)]
    x, y = x - dx, y - dy
    w, h = w + dx * 2, h + dy * 2
    img, bbox = crop_image(img, (x, y, w, h), bbox)
    img, bbox = scale_image(img, config.raw_image_size, bbox)
    return img, bbox


def map_classes(raw_data):
    """ Changes classes, so that they are subsequent numbers
    from 0 to config.classes_count
    """
    mapping = get_class_mapping(raw_data)
    return [(fn, img, mapping[cls], bbox) for fn, img, cls, bbox in raw_data]


def read_raw(count=-1):
    """ Returns list of tuples (image_filename, tensor, class, bounding_box)
    """
    if count == -1:
        count = len(config.images_paths)
    bboxes = read_bboxes()
    classes = read_classes()
    # TODO: consider not including bboxes, since they are not used
    # nor are proper for the preprocessedd images
    ret = [(p.name, cv2.imread(str(p)), classes[p.name], bboxes[p.name])
           for p in islice(config.images_paths, count)]
    ret = [[(name, new_img, cls, new_bbox)  # where:
            for new_img, new_bbox in (preprocess_raw_image(img, bbox),)][0]
           for name, img, cls, bbox in ret]
    return map_classes(ret)


def get_class_mapping(raw_data):
    """ Returns dict int -> int.
    Classes in raw_data may have values from non-continuous range.
    """
    _, _, classes, _ = zip(*raw_data)
    return {cls: i for i, cls in enumerate(np.unique(classes))}


def get_hog_features(img, feat_size=config.hog_feature_size):
    """ Returns features vector of size feat_size.
    """
    # TODO: check and adjust hog parameters, input image size itp.

    img = cv2.resize(img, (64, 128), cv2.INTER_LINEAR)

    # TODO: maybe consider experimenting and making visualizations with this
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fd, hog_image = hog(img, orientations=4, pixels_per_cell=(8, 8),
    #                    cells_per_block=(8, 8), visualise=True)
    # img = cv2.resize(img, (128, 128))

    # this is standard/default/well known/frequently used hog fd for images 128x64x3
    # output size is 3780
    hog = cv2.HOGDescriptor()
    fd = hog.compute(img)
    fd = fd.flatten()
    assert fd.shape == (config.hog_feature_size,)

    if len(fd) != feat_size:
        # TODO: find a good way for adjusting feature size
        # (but this doesn't seem to be that bad)
        # or maybe just assume that this len(fd) == feat_size
        raise NotImplementedError
        # rand = np.random.RandomState(config.random_seed)
        # return rand.choice(fd, feat_size)
    else:
        return fd


def process_one_example(_name, img, cls, _bbox, rand, use_hog=True, augment=True):
    """ Returns feed-ready pair (img_feature_vec, cls_vec).
    """
    if augment:
        img = transform_img(img, rand)
    # TODO: data augmentation
    label = np.zeros(config.classes_count)
    assert 0 <= cls < config.classes_count
    label[cls] = 1.0
    assert label.shape == (config.classes_count,)
    return (get_hog_features(img) if use_hog else cv2.resize(img, tuple([config.feed_ready_image_size] * 2))), label


def example_generator(raw_data, random_seed=config.random_seed, use_hog=True, augment=True):
    """ Yields feed-ready examples infinitely - pairs (img_feature_vec, cls_vec)
    Does data augmentation.
    """
    rand = random.Random(random_seed)
    while True:
        rand.shuffle(raw_data)
        for ex in raw_data:
            yield process_one_example(*ex, use_hog=use_hog, rand=rand, augment=augment)


def get_train_validation_raw(validation_ratio):
    """ Returns pair (train_raw_data, validation_raw_data)
    """
    raw_data = read_raw()
    rand = random.Random(config.random_seed)
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


def batch_generator(raw_data, batch_size, use_hog=True, augment=True):
    """ Yields pairs of lists (images_features, labels)
    """
    return starmap(lambda x, y: (np.array(x), np.array(y)), transposed_group(
        example_generator(raw_data, random_seed=config.random_seed, use_hog=use_hog, augment=augment),
        batch_size))


def get_unaugmented(raw_data, use_hog=True):
    """ Returns pair of list (images_features, labels) set of cropped with bboxes images.
    """
    ret = []
    for filename, img, cls, bbox in raw_data:
        cropped_img, new_bbox = crop_image(img, bbox, bbox)
        ret.append((process_one_example(
            filename, cropped_img, cls, new_bbox, use_hog=use_hog, augment=False, rand=None)))
    return list(zip(*ret))
