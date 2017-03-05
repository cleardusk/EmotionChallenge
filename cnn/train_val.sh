#!/usr/bin/env sh

caffe train -solver solver.prototxt -gpu 3 2>&1 | tee final_03051346.log