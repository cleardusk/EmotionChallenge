#!/usr/bin/env python
# coding: utf-8

import dlib
import os
import os.path as osp
import json
from multiprocessing import Pool
import cv2

LANDMAEK_NUM = 68
PREDICTOR_PATH = '../models/shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
face_regressor = dlib.shape_predictor(PREDICTOR_PATH)


def get_bbox(img):
    rects = face_detector(img, 1)
    # bug, fix face  may not detected
    if len(rects) == 0:
        # print('Not detected')
        h, w, _ = img.shape
        return 0, 0, w, h
    rect = rects[0]
    bbox = l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
    return bbox


def get_id_landmark(param):
    fp, dsize = param  # dsize: (w, h)
    img = cv2.imread(fp, cv2.IMREAD_COLOR)

    # resize image
    img_scale = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)

    bbox = get_bbox(img_scale)
    # for test
    # if bbox[0] == 0:
    #     print('{} NOT DETECTED'.format(fp))
    rect = dlib.rectangle(*bbox)
    res = face_regressor(img, rect).parts()
    landmark = []
    for pt in res:
        landmark.append(pt.x)
        landmark.append(pt.y)

    id = osp.split(fp)[-1]
    return {id: landmark}


def get_fps(data_dir):
    fps = []
    for root, dirs, files in os.walk(data_dir):
        for fl in files:
            fps.append(osp.join(root, fl))
    return sorted(fps)


def get_id_landmark_pool(data_dir, wfp, dsize=(224, 224)):
    pool = Pool()

    fps = get_fps(data_dir)
    scales = [dsize] * len(fps)
    res = pool.map(get_id_landmark, zip(fps, scales))

    pool.close()
    pool.join()

    id_landmark = {}
    for r in res:
        id_landmark.update(r)
    json.dump(id_landmark, open(wfp, 'w'), sort_keys=True)


def main():
    get_id_landmark_pool(data_dir='../data/face_224', wfp='../data/landmark_224.json')


if __name__ == '__main__':
    main()
