#!/usr/bin/env bash
#sh class_aware_train15_3_res50.sh 2>&1 |tee class_aware_train15_nosoftmax1loss_res50.log

sh class_aware_train15_4_res50.sh 2>&1 |tee class_aware_train15_nosigmoidsoftmax1loss_res50.log

sh class_aware_train15_5_res50.sh 2>&1 |tee class_aware_train15_nosigmoidloss_res50.log