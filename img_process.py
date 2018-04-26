import cv2
import imutils
from random import random


def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


def crop_img(img, random_tab):
    """ Returns cropped image (with expanded bbox)
    """
    dy, dx = (i/6 for i in img.shape[:2])
    x1 = int(random_tab[0]*dx)
    x2 = int((random_tab[1] + 5) * dx)
    y1 = int(random_tab[2]*dy)
    y2 = int((random_tab[1] + 5) * dy)
    img = img[y1:y2, x1:x2]
    return img


def flip_img(img, random):
    """ Returns rotated image:
    random <  0.5 -> horizontal
    random >= 0.5 -> vertical
    """
    img = cv2.flip(img, round(random), 1)
    return img


def rotate_img(img, random):
    """ Returns rotated and cropped image
    random = 1 -> 45 degrees clockwise
    random = 0.5 -> no rotate
    random = 0 -> 45 degrees counterclockwise
    """
    degrees = 45 * (random - 0.5)
    h, w = (int(2*i/3) for i in img.shape[:2])
    img = imutils.rotate_bound(img, degrees)
    h2, w2 = img.shape[:2]
    x = int((w2-w)/2)
    y = int((h2-h)/2)
    img = img[y:y+h, x:x+w]
    return img


def transform_img(img, rand):
    """ Randomly transforms image
    :param img: image to be transformed
    :param rand: random generator
    """
    if (rand.random() < 0.5):
        img = flip_img(img, 1)
    r = [rand.random() for _ in range(4)]
    # TODO: consider doing sometching to avoid cutting bird's head
    img = crop_img(img, r)
    return rotate_img(img, rand.random())


# TODO: remove/move shomewhere later
if __name__ == '__main__':
    # testing data
    img = cv2.imread('bird.jpg', 0)
    h, w = img.shape[:2]
    x, y = int(w/6), int(h/6)
    cv2.rectangle(img,(x,y),(5*x,5*y), (0,191,255), 3)
    show(img)

    fl_img = flip_img(img, random())
    show(fl_img)

    random_tab = [random(),random(),random(),random()]
    cr_img = crop_img(img, random_tab)
    show(cr_img)

    rotated = rotate_img(img, random())
    show(rotated)
