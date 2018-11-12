#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6,7 sh class_aware_train14_res50.sh 2>&1 | tee class_aware_train14_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh class_aware_train15_res50.sh 2>&1 | tee class_aware_train15_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train13_res50.sh 2>&1 | tee pyramid_class_aware_train13_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train14_res50.sh 2>&1 | tee pyramid_class_aware_train14_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train15_res50.sh 2>&1 | tee pyramid_class_aware_train15_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train16_res50.sh 2>&1 | tee pyramid_class_aware_train16_res50.log
