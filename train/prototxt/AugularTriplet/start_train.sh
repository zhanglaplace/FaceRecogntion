#!/bin/bash
# Usage:
# ./start_train.sh GPU
#
# Example:
# ./code/sphereface_train.sh 0,1,2,3
mkdir -p face_snapshot
mkdir -p result
../../../build/tools/caffe train -solver face_solver.prototxt -gpu all 2>&1 | tee result/angularTriplet_20180209_vgg_1.log
