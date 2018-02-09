#!/bin/bash
mkdir -p face_snapshot
../../../../build/tools/caffe train -solver face_solver.prototxt -gpu all 2>&1 | tee result/COCO_softmax_20180204_vgg.log
