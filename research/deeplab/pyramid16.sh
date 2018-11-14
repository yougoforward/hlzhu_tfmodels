#!/usr/bin/env bash

sh class_aware_train15.sh 2>&1 | tee class_aware_train15.log

sh class_aware_train15_res101.sh 2>&1 | tee class_aware_train16_res101.log

#sh pyramid_class_aware_train17_res50.sh 2>&1 | tee pyramid_class_aware_train17_res50.log
#
#sh pyramid_class_aware_train16_res50.sh 2>&1 | tee pyramid_class_aware_train16_res50.log

sh pyramid_class_aware_train16.sh 2>&1 | tee pyramid_class_aware_train16.log


#sh pyramid_class_aware_train16_res101.sh 2>&1 | tee pyramid_class_aware_train16_res101.log

