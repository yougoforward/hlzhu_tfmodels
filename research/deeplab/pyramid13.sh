#!/usr/bin/env bash

sh pyramid_class_aware_train13.sh 2>&1 | tee pyramid_class_aware_train13.log

sh pyramid_class_aware_train13_res50.sh 2>&1 | tee pyramid_class_aware_train13_res50.log

sh pyramid_class_aware_train13_res101.sh 2>&1 | tee pyramid_class_aware_train13_res101.log