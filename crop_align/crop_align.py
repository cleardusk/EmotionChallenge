#!/usr/bin/env python
# coding: utf-8

import json
import os
import os.path as osp
from glob import glob
from subprocess import check_call
from multiprocessing import Pool


def mkdir(d):
    if not osp.exists(d):
        os.mkdir(d)


def trans_crop(data_dir, dst_data_dir, lms_fp, bin='./build/crop_align', processes=32, tgt_h=224, tgt_w=224):
    """Select the 28th and 52th point"""
    cmds = []
    image_paths = sorted(glob(osp.join(data_dir, '*.JPG')))
    landmark = json.load(open(lms_fp))
    for image_path in image_paths:
        fn = osp.split(image_path)[-1]
        id = fn.split('_')[1]
        sub_dir = osp.join(dst_data_dir, id)
        mkdir(sub_dir)
        save_path = osp.join(sub_dir, fn)

        if osp.exists(save_path):
            continue

        lms = landmark.get(fn)
        lms = map(float, lms)
        ecx, ecy = lms[54], lms[55]
        ulx, uly = lms[102], lms[103]
        # tgt_h, tgt_w = 256., 256.
        tgt_ecy, tgt_uly = tgt_h / 55. * 20, tgt_h / 55. * 38

        cmd = ['{}'.format(bin), image_path, save_path]
        cmd.extend(map(str, [ecx, ecy, ulx, uly]))
        cmd.extend(map(str, [tgt_h, tgt_w, tgt_ecy, tgt_uly]))

        # print(' '.join(cmd))
        # check_call(cmd)
        cmds.append(cmd)

    pool = Pool(processes=processes)  # Pool() seems faster than Pool(16)
    pool.map(check_call, cmds)
    pool.close()
    pool.join()


def main():
    # trans_crop(data_dir='../data/train_data', dst_data_dir='../data/face_224', lms_fp='../data/train_landmark.json',
    #            tgt_h=224, tgt_w=224)
    # trans_crop(data_dir='../data/val_data', dst_data_dir='../data/face_224', lms_fp='../data/val_landmark.json', tgt_h=224, tgt_w=224)
    trans_crop(data_dir='../data/test_data', dst_data_dir='../data/face_224', lms_fp='../data/test_landmark.json', tgt_h=224, tgt_w=224)
    # for t in ['train', 'val', 'test']:
    #     trans_crop(data_dir='{}_data'.format(t), lms_fp='{}_landmark.json'.format(t), tgt_h=224, tgt_w=224)


if __name__ == '__main__':
    main()
