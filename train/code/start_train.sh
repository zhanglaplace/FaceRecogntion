#!/bin/bash

GPU_ID=$1
./../tools/caffe-sphereface/build/tools/caffe train -solver code/face_solver.prototxt -gpu ${GPU_ID} 2>&1 | tee result/train.log