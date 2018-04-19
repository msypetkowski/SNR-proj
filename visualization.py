import cv2
import numpy as np


def merge_images(images, scale=1):
    """ Concatenate same-sized images so that resulting image
    is a square with side of length:
    ceil(sqrt(len(images))) * (one image side length)
    """
    images = list(images)
    dim = int(np.ceil(np.sqrt(len(images))))
    for _ in range(dim * dim - len(images)):
        blank_image = images[0].copy()
        blank_image = blank_image * 0
        images.append(blank_image)
    rows = []

    img_iter = iter(images)
    for _ in range(dim):
        rows.append(np.concatenate([next(img_iter)
                                    for _ in range(dim)], axis=1))
    try:
        next(img_iter)
        assert 0
    except StopIteration:
        pass
    vis = np.concatenate(rows, axis=0)
    return cv2.resize(vis, (int(vis.shape[1] * scale),
                            int(vis.shape[0] * scale)))


def show_image(img):
    """ Show image and wait for key.
    Returns pressed key code.
    """
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key
