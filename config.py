"""
Data and model configuration.
"""

from pathlib import Path
import glob


class BaseConfig:
    data_root_path = Path.home().joinpath('snrdata')
    data_bboxes = data_root_path.joinpath('bounding_boxes.txt')
    data_classes = data_root_path.joinpath('classes.txt')
    data_labels = data_root_path.joinpath('image_class_labels.txt')
    images_paths = list(map(Path, glob.glob(
        str(data_root_path.joinpath('SET_C')) + '/**/*.jpg')))


config = BaseConfig
