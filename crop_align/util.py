#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt


def gjz_read_img(filename='', flag=cv2.IMREAD_COLOR):
    # for dlib face landmark detection, must return uint8 type ndarray
    img = cv2.imread(filename, flag)
    # return img.astype(np.float32)
    return img


def gjz_write_img(filename=None, img=None):
    """A wrapper of `cv2.imwrite`, but considering some situations"""
    assert (filename is not None and img is not None)

    cv2.imwrite(filename, img)


def gjz_show_img(img=None, title='', flag=0, **options):
    if flag == 0:
        plt.title(title)
        if 'invert' in options and options['invert']:
            img_inv = img.max() - img
            plt.imshow(img_inv, cmap='gray', interpolation='nearest')
        else:
            plt.imshow(img[:, :, ::-1], cmap='gray', interpolation='nearest')
        plt.show()

    elif flag == 1:
        cv2.imshow(title, img)
        cv2.waitKey(0)