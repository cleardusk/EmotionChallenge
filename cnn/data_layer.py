#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import caffe
import cv2


class DataLayerWithLandmark(caffe.Layer):
    """
    This is a python data layer that can do some image perturbations prior to feeding the net
    """

    def setup(self, bottom, top, **kwargs):
        sys.stderr.write("Set up DataLayerWithLandmark\n")
        self.top_names = ['data', 'label', 'landmarks']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        self.params = eval(self.param_str)

        # Check the paramameters for validity.
        # check_params(params)

        # # store input as class variables
        # self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(self.params)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        if len(top) != 3:
            raise AssertionError("Number of top blobs should be 3.")
        top[0].reshape(self.params['batch_size'], 3, self.params['im_h'], self.params['im_w'])
        top[1].reshape(self.params['batch_size'], 1)
        top[2].reshape(self.params['batch_size'], self.params['landmark_dim'])

    def forward(self, bottom, top):
        """
        Load data.
        """
        ims, labels, landmarks = self.batch_loader.load_next_batch()
        top[0].data[...] = ims
        top[1].data[...] = labels
        top[2].data[...] = landmarks

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params):
        self.params = params

        self.data_list = []

        # load images into memory
        print "Loading images"
        with open(self.params['list_file']) as list_f:
            for line in list_f:
                data_info = line.split()
                im_path = data_info[0]
                label = data_info[1]
                landmarks = data_info[2:]
                assert len(landmarks) == self.params['landmark_dim'], "landmark blob shape mismatch"

                # full_im_path = os.path.join(self.params['root'], im_path) if self.params['root'] else im_path
                # full_im_path = im_path
                # print(im_path)
                im = cv2.imread(im_path, cv2.IMREAD_COLOR).astype(np.float32)
                im = im.transpose((2, 0, 1))
                assert im.shape == (3, self.params['im_h'], self.params['im_w']), "image shape mismatch"

                label = float(label)
                landmarks = np.array(map(float, landmarks), dtype=np.float32)

                self.data_list.append((im, label, landmarks))
        self._cur = 0  # current image

        print "BatchLoader initialized with {} images".format(len(self.data_list))

    def load_next_batch(self):
        ims = np.zeros((self.params['batch_size'], 3, self.params['im_h'], self.params['im_w']), dtype=np.float32)
        labels = np.zeros((self.params['batch_size'], 1), dtype=np.float32)
        landmarks = np.zeros((self.params['batch_size'], self.params['landmark_dim']), dtype=np.float32)
        for itt in xrange(self.params['batch_size']):
            # Use the batch loader to load the next image.
            im, label, landmark = self.load_next_image()
            # Add directly to the caffe data layer
            ims[itt, ...] = im
            labels[itt, ...] = label
            landmarks[itt, ...] = landmark
        return ims, labels, landmarks

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.data_list):
            self._cur = 0

        # Load data
        im, label, landmark = self.data_list[self._cur]  # Get the image index
        # im = im[:, :, ::choice((-1, 1))]
        self._cur += 1
        return im, label, landmark
