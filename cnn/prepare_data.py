#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os
import os.path as osp
from random import shuffle
from util import label_to_num, label_to_num_dom, label_to_num_com

LANDMAEK_NUM = 68


def parse_id_label(ground_truth_fp='../data/training_new.txt'):
    # parse {id: label}
    lines = open(ground_truth_fp).read().replace('\r\n', '\n').strip().split('\n')
    id_label = {}

    for l in lines:
        # print(l)
        id, label = l.split()
        id_label.update({id: label_to_num(label)})
    return id_label


def load_train_data_mean(data_dir='../data/face_224', landmark_fp='../data/landmark_224.json', ground_truth_fp='../data/training_new.txt',
                         norm_flag=False):
    id_label = parse_id_label(ground_truth_fp=ground_truth_fp)

    id_landmark = json.load(open(landmark_fp))

    # parse face_id_landmarks
    face_id_landmarks = {}
    for id, _ in id_label.iteritems():
        face_id = id.split('_')[1]
        face_id_landmarks.update({face_id: []})

    for id, landmark in id_landmark.iteritems():
        if not id_label.has_key(id):
            continue
        face_id = id.split('_')[1]
        face_id_landmarks[face_id].append(landmark)

    face_id_landmarks_mean = {}
    for face_id, landmarks in face_id_landmarks.iteritems():
        mean = np.zeros(LANDMAEK_NUM * 2, dtype=np.float32)
        cnt = 0
        for landmark in landmarks:
            mean += np.array(landmark, dtype=np.float32)
            cnt += 1
        mean /= cnt
        face_id_landmarks_mean.update({face_id: mean.flatten()})

    # parse mean landmark
    X, y = [], []
    for id, label in sorted(id_label.iteritems()):  # if add sorted?
        face_id = id.split('_')[1]
        x = np.array(id_landmark[id], dtype=np.float32).flatten() - face_id_landmarks_mean[face_id]
        if norm_flag:
            x = (x - x.min()) / (x.max() - x.min())
        X.append(x)
        y.append(label)

    fps = [osp.join(data_dir, id.split('_')[1], id) for id, label in sorted(id_label.iteritems())]
    y = np.array(y, dtype=np.uint32)
    return fps, X, y


def load_val_data_mean(data_dir='../data/face_224', landmark_fp='landmark_224.json', val_data_order='order_of_validation.txt',
                       norm_flag=False):
    lines = open(val_data_order).read().replace('\r\n', '\n').strip().split('\n')
    ids = []
    for l in lines:
        id = l.strip()
        ids.append(id)

    # print(ids)
    id_landmark = json.load(open(landmark_fp))

    face_id_landmarks = {}
    for id in ids:
        face_id = id.split('_')[1]
        face_id_landmarks.update({face_id: []})

    # print(face_id_landmarks)
    for id in ids:
        face_id = id.split('_')[1]
        landmark = id_landmark[id]
        face_id_landmarks[face_id].append(landmark)

    face_id_landmarks_mean = {}
    for face_id, landmarks in face_id_landmarks.iteritems():
        mean = np.zeros(LANDMAEK_NUM * 2, dtype=np.float32)
        cnt = 0
        for landmark in landmarks:
            mean += np.array(landmark, dtype=np.float32)
            cnt += 1
        mean /= cnt
        face_id_landmarks_mean.update({face_id: mean.flatten()})

    # parse mean landmark
    X = []
    for id in sorted(ids):  # if add sorted?
        face_id = id.split('_')[1]
        x = np.array(id_landmark[id], dtype=np.float32).flatten() - face_id_landmarks_mean[face_id]
        if norm_flag:
            x = (x - x.min()) / (x.max() - x.min())  # pay attention to this
        X.append(x)
    fps = [osp.join(data_dir, id.split('_')[1], id) for id in sorted(ids)]
    return fps, X


def load_train_mean(fp):
    data = json.load(open(fp))
    ids = {}
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        # print(id)
        ids.update({id: []})
    # print(ids)
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        for feature, label in data_dict.values():
            ids[id].append((np.array(feature, dtype=np.float32).flatten(), label))
    # print(ids)

    # calc id mean pts
    ids_mean = {}
    for id in ids.iterkeys():
        feature_mean = np.zeros(136, dtype=np.float32)
        cnt = 0
        for feature, label in ids[id]:
            cnt += 1
            feature_mean += feature
        feature_mean /= float(cnt)
        ids_mean.update({id: feature_mean.flatten()})
    # print(ids_mean)

    # substract mean pts
    X, y = [], []
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        for feature, label in data_dict.values():
            X.append(np.array(feature, dtype=np.float32).flatten() - ids_mean[id])
            y.append(label)

    fps = [data_dict.keys()[0] for data_dict in data]
    return fps, X, np.array(y, dtype=np.uint32)


def load_val_mean(fp):
    # load data
    data = json.load(open(fp))
    ids = {}
    # get all id
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        # print(id)
        ids.update({id: []})
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        for feature in data_dict.values():
            ids[id].append(np.array(feature, dtype=np.float32).flatten())

    # calc id mean pts
    ids_mean = {}
    for id in ids.iterkeys():
        feature_mean = np.zeros(136, dtype=np.float32)
        cnt = 0
        for feature in ids[id]:
            cnt += 1
            feature_mean += feature
        feature_mean /= float(cnt)
        ids_mean.update({id: feature_mean.flatten()})

    X = []
    for data_dict in data:
        id = data_dict.keys()[0].split('/')[1]
        for feature in data_dict.values():
            X.append(np.array(feature, dtype=np.float32).flatten() - ids_mean[id])

    fps = [data_dict.keys()[0] for data_dict in data]
    return fps, X


def create_list_train():
    ground_truth_fp = '../data/training_validation_new.txt'
    train_image_fps, train_landmarks, train_labels = load_train_data_mean(ground_truth_fp=ground_truth_fp,
                                                                          landmark_fp='../data/landmark_224.json',
                                                                          norm_flag=False)

    data = zip(train_image_fps, train_landmarks, train_labels)
    train_fp = '../data/train_ld_shuffle.txt'
    val_fp = '../data/val_ld_shuffle.txt'
    train_data, val_data = data[:20719], data[20719:]

    def write(wfp, data):
        records = []
        for fp, lm, lb in data:
            l = ''
            l += '{} '.format(fp)
            l += '{} '.format(lb)
            l += ' '.join(map(lambda s: '{:0.6f}'.format(s), lm.tolist()))
            records.append(l)
        shuffle(records)
        with open(wfp, 'w') as f:
            f.write('\n'.join(records))

    write(train_fp, train_data)
    write(val_fp, val_data)


def create_list_test():
    test_image_fps, test_landmarks = load_val_data_mean(val_data_order='../data/order_of_test.txt',
                                                        landmark_fp='../data/landmark_224.json',
                                                        norm_flag=False)
    # val_image_fps, val_landmarks = load_val_data_mean(val_data_order='../order_of_validation.txt',
    #                                                   landmark_fp='../landmark_224.json')
    data = zip(test_image_fps, test_landmarks)
    fp = '../data/test_ld.txt'

    def write(wfp, data):
        records = []
        for fp, lm in data:
            l = ''
            l += '{} '.format(fp)
            l += ' '.join(map(lambda s: '{:0.6f}'.format(s), lm.tolist()))
            records.append(l)
        with open(wfp, 'w') as f:
            f.write('\n'.join(records))

    write(fp, data)


if __name__ == '__main__':
    create_list_train()
    create_list_test()