#!/usr/bin/env bash
sh class_aware_train15_3_res50.sh 2>&1 |tee class_aware_train15_nosoftmax1loss_res50.log
sh dual_pyramid_train_with_boundary.sh 2>&1 | tee dual_pyramid_train_with_boundary9.log
