#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train13.sh 2>&1 | tee pyramid_class_aware_train13.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train13_res50.sh 2>&1 | tee pyramid_class_aware_train13_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh pyramid_class_aware_train13_res101.sh 2>&1 | tee pyramid_class_aware_train13_res101.log