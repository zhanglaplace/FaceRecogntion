#!/bin/bash
# Usage:
# ./start_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3
mkdir -p face_snapshot
../../../../../build/tools/caffe train -solver face_solver.prototxt  -weights ./face_snapshot/center_model_iter_8000.caffemodel -gpu 0 2>&1 | tee result/center_train_20180112_shuffle0.log
