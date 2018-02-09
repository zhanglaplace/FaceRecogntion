#!/bin/bash
# Usage:
# ./start_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3
mkdir -p face_snapshot
mkdir -p face_snapshot_64
../../../../build/tools/caffe train -solver face_solver.prototxt -gpu all 2>&1 | tee result/amsoftmax_64_train_20180208_vgg_1.log
