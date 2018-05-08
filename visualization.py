import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import spline


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


def make_border(img, thickness=10, color=(0, 0, 0)):
    return cv2.copyMakeBorder(img, *[thickness] * 4, cv2.BORDER_CONSTANT, value=color)


def merge_images_with_border(images):
    return merge_images(list(map(make_border, images)))


def draw_bbox(img, bbox):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=3)
    return img


def show_image(img, bbox=None):
    """ Show image and wait for key.
    Returns pressed key code.
    """
    if bbox is not None:
        img = img.copy()
        draw_bbox(img, bbox)
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def draw_plots(plot_name, plot_values, plot_valid_values, legend_name_prefix):
    all_keys = set(list(plot_values.keys()) + list(plot_valid_values.keys()))
    scale = 3
    plt.figure(figsize=(3 * scale, len(all_keys) * scale))
    for i, key in enumerate(all_keys):
        plt.subplot(311 + i)
        plt.title(key)
        plt.xlabel('Training iteration')
        plt.ylabel(key)
        plt.grid(True)

        lines = []
        for plt_values, lbl_suffix in zip((plot_values, plot_valid_values), ("_training", "_validation")):
            if key in plt_values:
                label = legend_name_prefix + lbl_suffix
                values = plt_values[key]
                x, y = zip(*values)
                xnew = np.linspace(min(x), max(x), 300)
                power_smooth = spline(x, y, xnew)
                lines.append(plt.plot(xnew, power_smooth, label=label)[0])
                plt.plot(x, y, label=label, alpha=0.3, color=lines[-1].get_color())[0]
        if len(lines) > 1:
            plt.legend(handler_map={line: HandlerLine2D() for line in lines})

    # plt.tight_layout()
    plt.savefig(plot_name + '.png')
    plt.savefig(plot_name + '.eps')
    plt.show()
