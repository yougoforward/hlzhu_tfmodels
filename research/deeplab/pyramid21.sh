#!/usr/bin/env bash
sh pyramid_class_aware_train23_res50.sh 2>&1 | tee pyramid_class_aware_train23_res50.log

sh pyramid_class_aware_train21_res50.sh 2>&1 | tee pyramid_class_aware_train21_res50.log

sh pyramid_class_aware_train21.sh 2>&1 | tee pyramid_class_aware_train21.log