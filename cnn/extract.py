#!/usr/bin/env python
# coding: utf-8

import os

os.environ['GLOG_minloglevel'] = '2'  # not output caffe log
import cv2
import caffe
import time
from util import num_to_label


def foo():
    gt = open('../validation.txt').read().strip().split()

    predicts = open('predictions.txt').read().split()
    predicts = map(num_to_label, map(int, predicts))
    assert len(predicts) == len(gt)
    n = sum(map(lambda x, y: x == y, predicts, gt))
    tn = len(predicts)
    print('{}/{}({})'.format(n, tn, float(n) / tn))


class Classifier:
    def __init__(self, model_file, deploy_file, device_id=3):
        # caffe.set_device(device_id)
        caffe.set_mode_gpu()
        self.net = caffe.Net(deploy_file, model_file, caffe.TEST)
        # caffe.Net('deploy.prototxt', 1, weights='snapshot/with_landmark_03032013_test_iter_4400.caffemodel')

    def predict(self, image_array, landmark):
        self.net.blobs['data'].data[0, ...] = image_array.transpose((2, 0, 1))
        self.net.blobs['landmark_offset'].data[0, ...] = landmark
        s_c = time.time()
        self.net.forward()
        e_c = time.time()
        # fw_time = e_c - s_c

        prob = self.net.blobs['fc8'].data[0].argmax()
        return num_to_label(prob)
        # prob = self.net.blobs['fc8'].data[0]
        # return prob


def extract(model_file):
    classifier = Classifier(model_file=model_file,
                            deploy_file='deploy.prototxt')
    records = open('../data/test_ld.txt').read().strip().split('\n')
    # print(records[:1])
    predicts = []
    for rec in records:
        rec = rec.split()
        fp = rec[0]
        # print(fp)
        landmark = rec[1:]
        landmark = map(float, landmark)
        # print(landmark)
        assert len(landmark) == 136

        img = cv2.imread(fp)
        pred = classifier.predict(img, landmark)
        predicts.append(pred)
    # print(predicts)

    fn = 'predictions.txt'
    open(fn, 'w').write('\n'.join(map(str, predicts)))

    cmd = 'zip {} {}'.format(fn.replace('txt', 'zip'), fn)
    os.system(cmd)


def submit():
    # extract('snapshot/model_iter_50000.caffemodel')
    extract('../models/final.caffemodel')


if __name__ == '__main__':
    submit()
